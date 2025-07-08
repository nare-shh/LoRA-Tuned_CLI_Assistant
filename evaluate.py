

import json
import torch
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ModelEvaluator:
    def __init__(self, base_model_name, lora_adapter_path=None):
        """Initialize evaluator with base and fine-tuned models."""
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        
        print("Loading models for evaluation...")
        self.load_models()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def load_models(self):
        """Load base model and tokenizer, optionally load fine-tuned version."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load fine-tuned model if adapter exists
        if self.lora_adapter_path and os.path.exists(self.lora_adapter_path):
            print(f"Loading fine-tuned model with LoRA adapter: {self.lora_adapter_path}")
            self.finetuned_model = PeftModel.from_pretrained(self.base_model, self.lora_adapter_path)
        else:
            print("No LoRA adapter found, using base model for both comparisons")
            self.finetuned_model = self.base_model
    
    def generate_response(self, model, instruction, max_length=256):
        """Generate response from a model."""
        prompt = f"Instruction: {instruction}\nResponse:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()
        
        return response
    
    def calculate_bleu(self, reference, candidate):
        """Calculate BLEU score between reference and candidate."""
        smoothing = SmoothingFunction().method1
        
        # Tokenize
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        # Calculate BLEU score
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
        return score
    
    def calculate_rouge(self, reference, candidate):
        """Calculate ROUGE scores."""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def manual_score_plan(self, response):
        """Manually score the quality of CLI plan (0, 1, or 2)."""
        print(f"\nResponse to score: {response}")
        print("\nScoring criteria:")
        print("0 = Wrong/incomplete/unhelpful")
        print("1 = Partially useful but missing details")
        print("2 = Clear, complete, and actionable")
        
        while True:
            try:
                score = int(input("Enter score (0-2): "))
                if score in [0, 1, 2]:
                    return score
                else:
                    print("Please enter 0, 1, or 2")
            except ValueError:
                print("Please enter a valid number")
    
    def evaluate_test_set(self, test_prompts, reference_answers=None, manual_scoring=False):
        """Evaluate both models on test set."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'base_model': self.base_model_name,
            'lora_adapter': self.lora_adapter_path,
            'test_results': []
        }
        
        total_bleu_base = 0
        total_bleu_ft = 0
        total_rouge_base = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        total_rouge_ft = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        total_manual_base = 0
        total_manual_ft = 0
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n{'='*60}")
            print(f"Test {i+1}/{len(test_prompts)}: {prompt}")
            print(f"{'='*60}")
            
            # Generate responses
            base_response = self.generate_response(self.base_model, prompt)
            ft_response = self.generate_response(self.finetuned_model, prompt)
            
            print(f"\n🔵 Base Model Response:\n{base_response}")
            print(f"\n🟢 Fine-tuned Model Response:\n{ft_response}")
            
            test_result = {
                'prompt': prompt,
                'base_response': base_response,
                'finetuned_response': ft_response
            }
            
            # Calculate automatic metrics if reference is provided
            if reference_answers and i < len(reference_answers):
                reference = reference_answers[i]
                
                # BLEU scores
                bleu_base = self.calculate_bleu(reference, base_response)
                bleu_ft = self.calculate_bleu(reference, ft_response)
                
                # ROUGE scores
                rouge_base = self.calculate_rouge(reference, base_response)
                rouge_ft = self.calculate_rouge(reference, ft_response)
                
                test_result.update({
                    'reference': reference,
                    'bleu_base': bleu_base,
                    'bleu_finetuned': bleu_ft,
                    'rouge_base': rouge_base,
                    'rouge_finetuned': rouge_ft
                })
                
                total_bleu_base += bleu_base
                total_bleu_ft += bleu_ft
                for key in total_rouge_base:
                    total_rouge_base[key] += rouge_base[key]
                    total_rouge_ft[key] += rouge_ft[key]
                
                print(f"\n📊 Automatic Metrics:")
                print(f"BLEU - Base: {bleu_base:.3f}, Fine-tuned: {bleu_ft:.3f}")
                print(f"ROUGE-L - Base: {rouge_base['rougeL']:.3f}, Fine-tuned: {rouge_ft['rougeL']:.3f}")
            
            # Manual scoring
            if manual_scoring:
                print(f"\n👤 Manual Scoring for Test {i+1}")
                print("Score the BASE model response:")
                manual_base = self.manual_score_plan(base_response)
                print("Score the FINE-TUNED model response:")
                manual_ft = self.manual_score_plan(ft_response)
                
                test_result.update({
                    'manual_score_base': manual_base,
                    'manual_score_finetuned': manual_ft
                })
                
                total_manual_base += manual_base
                total_manual_ft += manual_ft
            
            results['test_results'].append(test_result)
        
        # Calculate averages
        n_tests = len(test_prompts)
        
        if reference_answers:
            results['average_metrics'] = {
                'bleu_base': total_bleu_base / n_tests,
                'bleu_finetuned': total_bleu_ft / n_tests,
                'rouge_base': {k: v / n_tests for k, v in total_rouge_base.items()},
                'rouge_finetuned': {k: v / n_tests for k, v in total_rouge_ft.items()}
            }
        
        if manual_scoring:
            results['manual_averages'] = {
                'base_model': total_manual_base / n_tests,
                'finetuned_model': total_manual_ft / n_tests
            }
        
        return results
    
    def print_summary(self, results):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print(" EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        if 'average_metrics' in results:
            avg = results['average_metrics']
            print(f" Average BLEU Scores:")
            print(f"  Base Model: {avg['bleu_base']:.3f}")
            print(f"  Fine-tuned: {avg['bleu_finetuned']:.3f}")
            print(f"  Improvement: {avg['bleu_finetuned'] - avg['bleu_base']:+.3f}")
            
            print(f"\n📈Average ROUGE-L Scores:")
            print(f"  Base Model: {avg['rouge_base']['rougeL']:.3f}")
            print(f"  Fine-tuned: {avg['rouge_finetuned']['rougeL']:.3f}")
            print(f"  Improvement: {avg['rouge_finetuned']['rougeL'] - avg['rouge_base']['rougeL']:+.3f}")
        
        if 'manual_averages' in results:
            manual = results['manual_averages']
            print(f"\n👤 Average Manual Scores (0-2):")
            print(f"  Base Model: {manual['base_model']:.2f}")
            print(f"  Fine-tuned: {manual['finetuned_model']:.2f}")
            print(f"  Improvement: {manual['finetuned_model'] - manual['base_model']:+.2f}")

def main():
    # Test prompts (from section 7 + additional)
    test_prompts = [
        "How do I create a new Git branch and switch to it?",
        "What's the command to compress a directory using tar?",
        "How can I search for a specific text pattern in multiple files?",
        "What's the command to create a Python virtual environment?",
        "How do I list all files including hidden ones?",
        "How do I undo the last Git commit but keep changes?",  # Edge case
        "What command shows disk usage of current directory?"    # Edge case
    ]
    
    # Optional reference answers for automatic evaluation
    reference_answers = [
        "Use: git checkout -b <branch-name>",
        "Use: tar -czf archive.tar.gz directory/",
        "Use: grep -r 'pattern' .",
        "Use: python -m venv venv_name",
        "Use: ls -la",
        "Use: git reset --soft HEAD~1",
        "Use: du -sh ."
    ]
    
    # Configuration
    BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LORA_ADAPTER = "checkpoints/lora_adapter"
    OUTPUT_FILE = "evaluation/results.json"
    
    # Create output directory
    os.makedirs("evaluation", exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(BASE_MODEL, LORA_ADAPTER)
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluator.evaluate_test_set(
        test_prompts, 
        reference_answers, 
        manual_scoring=True  # Set to False to skip manual scoring
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()