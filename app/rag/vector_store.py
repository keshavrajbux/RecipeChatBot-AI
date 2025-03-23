import os
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import pickle
import json

load_dotenv()

class RecipeVectorStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = None
        self.recipes = []
        self.initialize_store()

    def initialize_store(self):
        """Initialize or load existing vector store"""
        if os.path.exists(f"{self.vector_store_path}/faiss_index.bin"):
            self.index = faiss.read_index(f"{self.vector_store_path}/faiss_index.bin")
            with open(f"{self.vector_store_path}/recipes.pkl", 'rb') as f:
                self.recipes = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            os.makedirs(self.vector_store_path, exist_ok=True)

    def add_recipes(self, recipes: List[Dict]):
        """Add new recipes to the vector store"""
        texts = [self._prepare_text(recipe) for recipe in recipes]
        embeddings = self.embedding_model.encode(texts)
        
        self.index.add(np.array(embeddings))
        self.recipes.extend(recipes)
        
        # Save the updated index and recipes
        self._save_store()

    def search_similar_recipes(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar recipes using the query"""
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(query_embedding, k)
        
        similar_recipes = []
        for idx in I[0]:
            if idx < len(self.recipes):
                similar_recipes.append(self.recipes[idx])
        
        return similar_recipes

    def _prepare_text(self, recipe: Dict) -> str:
        """Prepare recipe text for embedding"""
        return f"{recipe.get('title', '')} {recipe.get('ingredients', '')} {recipe.get('instructions', '')}"

    def _save_store(self):
        """Save the current state of the vector store"""
        faiss.write_index(self.index, f"{self.vector_store_path}/faiss_index.bin")
        with open(f"{self.vector_store_path}/recipes.pkl", 'wb') as f:
            pickle.dump(self.recipes, f)

    def get_recipe_count(self) -> int:
        """Get the total number of recipes in the store"""
        return len(self.recipes) 