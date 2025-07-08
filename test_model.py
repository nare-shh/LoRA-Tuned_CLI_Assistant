#!/usr/bin/env python3
"""
Test script to check model downloading and availability
Run this first to verify your setup works.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os

def test_model_download(model_name):
    """Test if we can download and load a model."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    try:
        print("1. Testing tokenizer download...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir="./model_cache"
        )
        print(f"   ✅ Tokenizer loaded successfully")
        print(f"   📝 Vocab size: {tokenizer.vocab_size}")
        
        print("2. Testing model config...")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="./model_cache"
        )
        print(f"   ✅ Config loaded successfully")
        print(f"   🧠 Model type: {config.model_type}")
        
        print("3. Testing model download (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
            cache_dir="./model_cache",
            low_cpu_mem_usage=True
        )
        print(f"   ✅ Model loaded successfully")
        
        # Test inference
        print("4. Testing inference...")
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   ✅ Test generation: {response}")
        
        return True, model_name
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}...")
        return False, str(e)

def main():
    print("🔍 Testing Model Download and Availability")
    print("This script will test if we can download and use different models.")
    
    # List of models to try (ordered by preference and size)
    models_to_test = [
        "gpt2",  # Small, reliable baseline
        "distilgpt2",  # Even smaller
        "microsoft/DialoGPT-small",  # Good for chat
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Original target
        "microsoft/phi-2",  # Alternative small model
    ]
    
    working_models = []
    failed_models = []
    
    # Create cache directory
    os.makedirs("model_cache", exist_ok=True)
    
    for model_name in models_to_test:
        success, result = test_model_download(model_name)
        if success:
            working_models.append(model_name)
            print(f"✅ {model_name} - WORKING")
        else:
            failed_models.append((model_name, result))
            print(f"❌ {model_name} - FAILED")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    if working_models:
        print(f"✅ Working models ({len(working_models)}):")
        for model in working_models:
            print(f"   - {model}")
        
        print(f"\n🎯 RECOMMENDATION:")
        print(f"Use this model for training: {working_models[0]}")
        print(f"Edit train_lora.py and set: MODEL_NAME = \"{working_models[0]}\"")
        
    else:
        print("❌ No models worked. Possible issues:")
        print("   1. Internet connection problems")
        print("   2. Hugging Face hub access issues")
        print("   3. Disk space problems")
        print("\n🔧 Try these fixes:")
        print("   1. Check internet: ping huggingface.co")
        print("   2. Clear cache: rm -rf ~/.cache/huggingface/")
        print("   3. Login to HF: pip install huggingface_hub && huggingface-cli login")
        print("   4. Check disk space: df -h")
    
    if failed_models:
        print(f"\n❌ Failed models ({len(failed_models)}):")
        for model, error in failed_models:
            print(f"   - {model}: {error[:50]}...")

if __name__ == "__main__":
    main()