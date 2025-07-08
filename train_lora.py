#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for CLI Agent
Fine-tunes a small language model on CLI Q&A data using LoRA.
Updated to work without quantization for CPU/compatibility issues.
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import os

def load_cli_data(file_path):
    """Load CLI Q&A data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            # Format as instruction-response pairs
            formatted = {
                "text": f"Instruction: {item['question']}\nResponse: {item['answer']}"
            }
            data.append(formatted)
    return data

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the dataset for causal language modeling."""
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  # Use max_length padding
        max_length=max_length,
        return_tensors=None  # Return lists, not tensors for datasets
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def check_gpu_availability():
    """Check if CUDA is available and properly set up."""
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available, using CPU")
        return False

def download_model_safely(model_name):
    """Safely download and verify model availability."""
    print(f"Checking model availability: {model_name}")
    
    try:
        # Try to load tokenizer first (smaller download)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer loaded successfully")
        
        # Try to load model info
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Model config loaded successfully")
        
        return True, tokenizer
    except Exception as e:
        print(f"❌ Error with {model_name}: {e}")
        return False, None

def main():
    # Configuration - try multiple models in order of preference
    MODELS_TO_TRY = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/DialoGPT-small", 
        "distilgpt2",
        "gpt2"
    ]
    
    DATA_PATH = "data/cli_qa.jsonl"
    OUTPUT_DIR = "checkpoints/lora_adapter"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Check GPU availability
    use_gpu = check_gpu_availability()
    
    # Try to find a working model
    MODEL_NAME = None
    tokenizer = None
    
    for model_name in MODELS_TO_TRY:
        print(f"\n🔄 Trying model: {model_name}")
        success, tok = download_model_safely(model_name)
        if success:
            MODEL_NAME = model_name
            tokenizer = tok
            print(f"✅ Successfully using model: {MODEL_NAME}")
            break
        else:
            print(f"⚠️  Failed to load {model_name}, trying next...")
    
    if MODEL_NAME is None:
        print("❌ Could not load any model. Please check your internet connection.")
        print("You can also try running this first to cache a model:")
        print("  python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')\"")
        return
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate configuration
    print(f"\n🔄 Loading model: {MODEL_NAME}")
    
    if use_gpu:
        try:
            # Try with quantization if GPU is available
            print("Attempting to load with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                cache_dir="./model_cache"  # Use local cache
            )
            print("✅ Successfully loaded with quantization")
            
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
            print("Loading without quantization...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    cache_dir="./model_cache"
                )
                print("✅ Successfully loaded without quantization")
            except Exception as e2:
                print(f"❌ GPU loading failed: {e2}")
                print("Falling back to CPU...")
                use_gpu = False
    
    if not use_gpu:
        # CPU loading without quantization
        print("Loading model for CPU use...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map=None,  # Let PyTorch handle device placement
                trust_remote_code=True,
                cache_dir="./model_cache",
                low_cpu_mem_usage=True  # Help with large models on CPU
            )
            print("✅ Successfully loaded on CPU")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("This might be a network issue. Try these steps:")
            print("1. Check your internet connection")
            print("2. Clear transformers cache: rm -rf ~/.cache/huggingface/")
            print("3. Try running: huggingface-cli login")
            return
    
    # LoRA configuration - adjusted for TinyLlama
    print("Configuring LoRA...")
    
    # Get the model's target modules (this varies by model architecture)
    target_modules = []
    for name, module in model.named_modules():
        if 'q_proj' in name or 'v_proj' in name or 'k_proj' in name or 'o_proj' in name:
            target_modules.extend(['q_proj', 'v_proj', 'k_proj', 'o_proj'])
            break
    
    # Fallback target modules for different architectures
    if not target_modules:
        # Common alternatives for different model types
        possible_targets = [
            ['query_key_value'],  # Some models use this
            ['qkv_proj'],         # Another common name
            ['attention.wq', 'attention.wv'],  # Llama-style
            ['self_attn.q_proj', 'self_attn.v_proj'],  # Transformer style
        ]
        
        for target_set in possible_targets:
            for name, _ in model.named_modules():
                if any(target in name for target in target_set):
                    target_modules = target_set
                    break
            if target_modules:
                break
    
    # If still no target modules found, use a general approach
    if not target_modules:
        print("Using general Linear layer targeting...")
        target_modules = ["q_proj", "v_proj"]  # Most common names
    
    print(f"Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Reduced rank for stability
        lora_alpha=16,  # Reduced alpha
        lora_dropout=0.1,
        target_modules=target_modules
    )
    
    # Apply LoRA
    try:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("✅ LoRA applied successfully")
    except Exception as e:
        print(f"❌ Error applying LoRA: {e}")
        print("This might be due to module name mismatch. Let's check available modules:")
        for name, _ in model.named_modules():
            if 'proj' in name or 'attention' in name:
                print(f"  - {name}")
        return
    
    # Load and prepare data
    print(f"Loading data from: {DATA_PATH}")
    try:
        raw_data = load_cli_data(DATA_PATH)
        print(f"Loaded {len(raw_data)} examples")
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_PATH}")
        print("Creating sample data first...")
        # Create sample data if it doesn't exist
        sample_data = [
            {"question": "How do I create a new Git branch?", "answer": "Use: git checkout -b <branch-name>"},
            {"question": "How to list files including hidden ones?", "answer": "Use: ls -la"},
            {"question": "How do I search for text in files?", "answer": "Use: grep 'pattern' filename"},
            {"question": "How to create a directory?", "answer": "Use: mkdir <directory-name>"},
            {"question": "How do I compress files with tar?", "answer": "Use: tar -czf archive.tar.gz files/"}
        ]
        
        os.makedirs("data", exist_ok=True)
        with open(DATA_PATH, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        raw_data = load_cli_data(DATA_PATH)
        print(f"Created and loaded {len(raw_data)} sample examples")
    
    # Create dataset
    dataset = Dataset.from_list(raw_data)
    
    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"📊 Dataset info:")
    print(f"   Total examples: {len(tokenized_dataset)}")
    print(f"   Sample keys: {list(tokenized_dataset[0].keys())}")
    print(f"   Input length: {len(tokenized_dataset[0]['input_ids'])}")
    print(f"   Labels length: {len(tokenized_dataset[0]['labels'])}")
    
    # Show a sample of the tokenized data
    sample_text = tokenized_dataset[0]
    decoded = tokenizer.decode(sample_text['input_ids'][:50], skip_special_tokens=True)
    print(f"   Sample text: {decoded}...")
    
    # Training arguments - adjusted for CPU/limited resources
    batch_size = 1 if not use_gpu else 2
    grad_accum_steps = 8 if not use_gpu else 4
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_steps=10,
        logging_steps=5,
        save_steps=100,
        eval_strategy="no",  # Changed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=use_gpu,  # Only use fp16 if GPU is available
        optim="adamw_torch",  # Use PyTorch's AdamW
        lr_scheduler_type="linear",
        learning_rate=5e-5,
        weight_decay=0.01,
    )
    
    # Create trainer with data collator for language modeling
    from transformers import DataCollatorForLanguageModeling
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8 if use_gpu else None,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )
    
    print("Starting training...")
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("This might be due to memory constraints or configuration issues.")
        return
    
    # Save the LoRA adapter
    print(f"Saving LoRA adapter to: {OUTPUT_DIR}")
    try:
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("✅ Model saved successfully!")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return
    
    print("\n" + "="*50)
    print("🎉 Training completed successfully!")
    print(f"📁 LoRA adapter saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Test the agent: python agent.py")
    print("2. Evaluate performance: python evaluate.py")
    print("="*50)

if __name__ == "__main__":
    main()