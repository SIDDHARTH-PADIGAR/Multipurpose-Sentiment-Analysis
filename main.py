"""
Complete End-to-End NLP Pipeline for Sentiment Analysis
========================================================

Installation:
pip install transformers datasets torch torchvision torchaudio evaluate scikit-learn accelerate

Running:
python train_and_infer.py

This will train a BERT model on IMDb sentiment analysis, evaluate it, and run inference examples.
GPU recommended but not required. Training takes ~10-15 minutes on GPU, longer on CPU.
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
import logging

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # ========================================
    # 1. DATA LOADING AND PREPROCESSING
    # ========================================
    logger.info("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    
    # Optimized subset for student laptop - balanced to avoid overfitting
    train_dataset = dataset["train"].shuffle(seed=42).select(range(2000))  # Reduced from 5000
    test_dataset = dataset["test"].shuffle(seed=42).select(range(500))     # Reduced from 1000
    
    logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # ========================================
    # 2. TOKENIZER SETUP
    # ========================================
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding=True, 
            max_length=512
        )
    
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # ========================================
    # 3. MODEL SETUP
    # ========================================
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # ========================================
    # 4. METRICS FUNCTION
    # ========================================
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }
    
    # ========================================
    # 5. TRAINING SETUP
    # ========================================
    # Check if GPU is available and supports mixed precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16_enabled = torch.cuda.is_available()
    
    logger.info(f"Using device: {device}")
    logger.info(f"Mixed precision (fp16): {fp16_enabled}")
    
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=3e-5,  # Slightly higher LR for faster convergence
        per_device_train_batch_size=8,   # Reduced batch size for laptop
        per_device_eval_batch_size=8,    # Reduced batch size
        num_train_epochs=2,  # Reduced epochs to prevent overfitting with small data
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,    # More frequent logging for smaller dataset
        eval_strategy="steps",
        eval_steps=100,      # More frequent evaluation
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=fp16_enabled,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        warmup_steps=100,    # Added warmup for better training stability
    )
    
    # ========================================
    # 6. TRAINER SETUP AND TRAINING
    # ========================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Increased patience for small dataset
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # ========================================
    # 7. EVALUATION
    # ========================================
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    
    logger.info("Final Evaluation Results:")
    for key, value in eval_results.items():
        logger.info(f"{key}: {value:.4f}")
    
    # ========================================
    # 8. SAVE MODEL
    # ========================================
    logger.info("Saving model...")
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    
    # ========================================
    # 9. INFERENCE DEMO
    # ========================================
    logger.info("Running inference demo...")
    
    # Load the trained model for inference
    model = AutoModelForSequenceClassification.from_pretrained("./final_model")
    tokenizer = AutoTokenizer.from_pretrained("./final_model")
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Test sentences
    test_sentences = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film, waste of time. Acting was horrible.",
        "It was okay, nothing special but not bad either.",
        "One of the best movies I've ever seen. Highly recommend!",
        "Boring and predictable. Could barely stay awake."
    ]
    
    label_names = ["Negative", "Positive"]
    
    logger.info("\n" + "="*50)
    logger.info("INFERENCE RESULTS")
    logger.info("="*50)
    
    with torch.no_grad():
        for sentence in test_sentences:
            # Tokenize
            inputs = tokenizer(
                sentence, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
            
            logger.info(f"Text: {sentence}")
            logger.info(f"Prediction: {label_names[predicted_class]} (confidence: {confidence:.4f})")
            logger.info("-" * 50)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()