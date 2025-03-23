from typing import List, Dict, Optional
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import json

class ContextAwareRecipeGenerator:
    def __init__(self, model_path: str):
        # Initialize the pipeline with the fine-tuned model
        self.pipeline = pipeline(
            "text-generation",
            model=model_path,
            device_map="auto",
            torch_dtype="auto"
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        
        # Define the chain-of-thought prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["ingredients", "dietary_restrictions", "cuisine_type", "cooking_time"],
            template="""Let's think about this step by step:

1. First, let's analyze the available ingredients:
{ingredients}

2. Consider the dietary restrictions:
{dietary_restrictions}

3. Taking into account the desired cuisine type:
{cuisine_type}

4. And the available cooking time:
{cooking_time} minutes

Now, let's create a recipe that:
- Uses the available ingredients efficiently
- Respects all dietary restrictions
- Matches the cuisine style
- Can be prepared within the time limit

Recipe:
Title: 

Ingredients:

Step-by-step Instructions:

Cooking Tips:

Nutritional Notes:
"""
        )
        
        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
        
    def _validate_ingredients(self, ingredients: List[str]) -> List[str]:
        """Validate and normalize ingredients"""
        return [ing.strip().lower() for ing in ingredients if ing.strip()]
        
    def _format_cooking_time(self, time_minutes: int) -> str:
        """Format cooking time with appropriate units"""
        if time_minutes < 60:
            return f"{time_minutes} minutes"
        hours = time_minutes // 60
        minutes = time_minutes % 60
        return f"{hours} hours and {minutes} minutes" if minutes else f"{hours} hours"
        
    def generate_recipe(
        self,
        ingredients: List[str],
        dietary_restrictions: Optional[List[str]] = None,
        cuisine_type: Optional[str] = None,
        cooking_time: Optional[int] = None
    ) -> Dict:
        """
        Generate a recipe considering multiple contexts
        
        Args:
            ingredients: List of available ingredients
            dietary_restrictions: List of dietary restrictions (e.g., ["vegetarian", "gluten-free"])
            cuisine_type: Desired cuisine type (e.g., "Italian", "Indian")
            cooking_time: Maximum cooking time in minutes
            
        Returns:
            Dictionary containing the generated recipe
        """
        # Prepare inputs
        validated_ingredients = self._validate_ingredients(ingredients)
        formatted_ingredients = "\n".join(f"- {ing}" for ing in validated_ingredients)
        
        # Format contexts
        dietary_context = "\n".join(dietary_restrictions) if dietary_restrictions else "None specified"
        cuisine_context = cuisine_type if cuisine_type else "Any cuisine"
        time_context = self._format_cooking_time(cooking_time) if cooking_time else "No time limit"
        
        # Generate recipe using chain-of-thought
        recipe_text = self.llm_chain.run({
            "ingredients": formatted_ingredients,
            "dietary_restrictions": dietary_context,
            "cuisine_type": cuisine_context,
            "cooking_time": time_context
        })
        
        # Parse the generated recipe
        recipe_parts = recipe_text.split("\n\n")
        recipe_dict = {}
        
        for part in recipe_parts:
            if part.startswith("Title:"):
                recipe_dict["title"] = part.replace("Title:", "").strip()
            elif part.startswith("Ingredients:"):
                recipe_dict["ingredients"] = [
                    ing.strip() for ing in 
                    part.replace("Ingredients:", "").strip().split("\n")
                    if ing.strip()
                ]
            elif part.startswith("Step-by-step Instructions:"):
                recipe_dict["instructions"] = [
                    step.strip() for step in 
                    part.replace("Step-by-step Instructions:", "").strip().split("\n")
                    if step.strip()
                ]
            elif part.startswith("Cooking Tips:"):
                recipe_dict["tips"] = part.replace("Cooking Tips:", "").strip()
            elif part.startswith("Nutritional Notes:"):
                recipe_dict["nutrition"] = part.replace("Nutritional Notes:", "").strip()
        
        return recipe_dict 