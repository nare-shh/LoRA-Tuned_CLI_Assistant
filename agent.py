#!/usr/bin/env python3
"""
CLI Agent Script
Uses fine-tuned model to generate CLI command plans and simulate execution.
"""

import json
import torch
import re
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

class CLIAgent:
    def __init__(self, base_model_name, lora_adapter_path, logs_dir="logs"):
        """Initialize the CLI Agent with fine-tuned model."""
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.logs_dir = logs_dir
        
        # Create logs directory
        os.makedirs(logs_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(logs_dir, "trace.jsonl")
        
        print("Loading model and tokenizer...")
        self.load_model()
        print("CLI Agent ready!")
    
    def load_model(self):
        """Load the base model and LoRA adapter."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA adapter if it exists
        if os.path.exists(self.lora_adapter_path):
            print(f"Loading LoRA adapter from: {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
        else:
            print(f"Warning: LoRA adapter not found at {self.lora_adapter_path}")
            print("Using base model without fine-tuning.")
    
    def log_step(self, step_type, content, metadata=None):
        """Log execution steps to JSONL file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step_type": step_type,
            "content": content,
            "metadata": metadata or {}
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def generate_response(self, instruction, max_length=256, temperature=0.7):
        """Generate response using the fine-tuned model."""
        # Format prompt
        prompt = f"Instruction: {instruction}\nResponse:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after "Response:")
        if "Response:" in response:
            response = response.split("Response:")[-1].strip()
        
        return response
    
    def extract_commands(self, text):
        """Extract shell commands from generated text."""
        commands = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for lines that start with common CLI patterns
            if (line.startswith(('git ', 'cd ', 'ls ', 'mkdir ', 'rm ', 'cp ', 'mv ', 
                              'grep ', 'find ', 'tar ', 'curl ', 'wget ', 'ssh ',
                              'chmod ', 'chown ', 'sudo ', 'pip ', 'npm ', 'docker ',
                              'python ', 'node ', 'java ', 'gcc ', 'make ')) or
                line.startswith('$') or
                (line.startswith('`') and line.endswith('`'))):
                
                # Clean up the command
                command = line.replace('`', '').replace('$', '').strip()
                if command:
                    commands.append(command)
        
        return commands
    
    def simulate_command(self, command):
        """Simulate command execution (dry run)."""
        print(f"🔧 Dry-run: {command}")
        
        # Log the simulated execution
        self.log_step("command_simulation", command, {
            "status": "dry_run",
            "safe_mode": True
        })
        
        # Provide some context about what the command would do
        if command.startswith('git '):
            return f"Git operation: {command}"
        elif command.startswith('cd '):
            return f"Change directory: {command}"
        elif command.startswith(('ls', 'find')):
            return f"List/search operation: {command}"
        elif command.startswith(('mkdir', 'touch')):
            return f"Create operation: {command}"
        elif command.startswith(('rm', 'rmdir')):
            return f"⚠️  Delete operation: {command}"
        else:
            return f"Shell command: {command}"
    
    def process_instruction(self, instruction):
        """Process a user instruction and generate CLI plan."""
        print(f"\n📋 Instruction: {instruction}")
        
        # Log the instruction
        self.log_step("user_instruction", instruction)
        
        # Generate response
        response = self.generate_response(instruction)
        print(f"🤖 Generated Plan:\n{response}\n")
        
        # Log the generated response
        self.log_step("generated_response", response)
        
        # Extract and simulate commands
        commands = self.extract_commands(response)
        
        if commands:
            print("🚀 Extracted Commands:")
            for i, cmd in enumerate(commands, 1):
                print(f"{i}. {self.simulate_command(cmd)}")
        else:
            print("ℹ️  No executable commands found in the response.")
        
        return response, commands
    
    def interactive_mode(self):
        """Run the agent in interactive mode."""
        print("\n" + "="*50)
        print("🤖 CLI Agent Interactive Mode")
        print("Type 'exit' or 'quit' to stop")
        print("="*50)
        
        while True:
            try:
                instruction = input("\n💬 Enter your CLI instruction: ").strip()
                
                if instruction.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not instruction:
                    continue
                
                self.process_instruction(instruction)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="CLI Agent with Fine-tuned Model")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="Base model name")
    parser.add_argument("--lora-adapter", default="checkpoints/lora_adapter",
                       help="Path to LoRA adapter")
    parser.add_argument("--instruction", help="Single instruction to process")
    parser.add_argument("--interactive", action="store_true", default=True,
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = CLIAgent(
        base_model_name=args.base_model,
        lora_adapter_path=args.lora_adapter
    )
    
    # Process single instruction or run interactive mode
    if args.instruction:
        agent.process_instruction(args.instruction)
    elif args.interactive:
        agent.interactive_mode()

if __name__ == "__main__":
    main()