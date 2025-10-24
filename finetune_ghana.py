"""
Simple fine-tuning script for Ghana knowledge - focused approach
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import os

# Configuration
MODEL_NAME = "google/flan-t5-small"
DATASET_FILE = "ghana_qa.json"
OUTPUT_DIR = "./ghana-finetuned-model"
MAX_LENGTH = 128  # Reduced for faster training

def load_dataset(file_path):
    """Load the question-answer dataset from JSON file"""
    data = []
    print(f"Loading dataset from {file_path}...")
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                json_obj = json.loads(line.strip())
                data.append({
                    'input_text': json_obj['question'],
                    'target_text': json_obj['answer']
                })
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(data)} question-answer pairs")
    return Dataset.from_list(data)

def preprocess_function(examples, tokenizer):
    """Preprocess the data for training - simplified approach"""
    inputs = examples['input_text']
    targets = examples['target_text']
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    # Replace padding token id's of the labels by -100 so it's ignored by loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    print("=== Simple Fine-tuning FLAN-T5 with Ghana Knowledge ===")
    print()
    
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,  # Important: set to False for training
        r=16,  # Increased rank for potentially better learning
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v", "k", "o"]  # Target more modules
    )
    
    # Apply PEFT to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load and preprocess dataset
    dataset = load_dataset(DATASET_FILE)
    
    # Use smaller dataset for focused training (first 20 examples)
    small_dataset = dataset.select(range(min(20, len(dataset))))
    print(f"Using {len(small_dataset)} examples for focused training")
    
    # Preprocess
    tokenized_dataset = small_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=small_dataset.column_names
    )
    
    # Training arguments - more aggressive training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,  # More epochs
        per_device_train_batch_size=2,  # Smaller batch
        per_device_eval_batch_size=2,
        learning_rate=1e-3,  # Higher learning rate
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=None
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting focused fine-tuning...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training completed!")
    
    # Quick test
    print("\nQuick test:")
    test_question = "When did Ghana gain independence?"
    inputs = tokenizer(test_question, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Q: {test_question}")
    print(f"A: {answer}")

if __name__ == "__main__":
    main()
