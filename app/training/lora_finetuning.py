from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import torch
from typing import List, Dict
import json

class RecipeModelTrainer:
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-2-7b-hf",
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)
        
    def _prepare_recipe_data(self, recipes: List[Dict]) -> Dataset:
        """Convert recipes to training format"""
        def format_recipe(recipe: Dict) -> str:
            return f"""Title: {recipe['title']}
Ingredients: {', '.join(recipe['ingredients'])}
Instructions: {' '.join(recipe['instructions'])}
"""
        
        texts = [format_recipe(recipe) for recipe in recipes]
        return Dataset.from_dict({
            "text": texts
        })
        
    def train(
        self,
        train_recipes: List[Dict],
        val_recipes: List[Dict],
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4
    ):
        """Fine-tune the model using LoRA"""
        # Prepare datasets
        train_dataset = self._prepare_recipe_data(train_recipes)
        val_dataset = self._prepare_recipe_data(val_recipes)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model(f"{output_dir}/final")
        
    def generate_recipe(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate a recipe using the fine-tuned model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True) 