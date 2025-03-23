from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize

class HybridRecipeSearch:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.dimension)
        self.recipes = []
        self.bm25 = None
        self.tokenized_corpus = []
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Tokenize and preprocess text for BM25"""
        return text.lower().split()
        
    def _compute_hybrid_scores(
        self, 
        semantic_scores: np.ndarray, 
        keyword_scores: np.ndarray,
        alpha: float = 0.7
    ) -> np.ndarray:
        """Combine semantic and keyword scores with weighted average"""
        # Normalize scores to [0, 1] range
        semantic_scores_norm = normalize(semantic_scores.reshape(1, -1))[0]
        keyword_scores_norm = normalize(keyword_scores.reshape(1, -1))[0]
        
        # Weighted combination
        hybrid_scores = (alpha * semantic_scores_norm + 
                        (1 - alpha) * keyword_scores_norm)
        return hybrid_scores
        
    def add_recipes(self, recipes: List[Dict]):
        """Add recipes to both semantic and keyword indices"""
        # Prepare texts for embedding
        texts = [
            f"{recipe['title']} {' '.join(recipe['ingredients'])} {' '.join(recipe['instructions'])}"
            for recipe in recipes
        ]
        
        # Update semantic search
        embeddings = self.embedding_model.encode(texts)
        self.index.add(np.array(embeddings))
        
        # Update keyword search
        self.tokenized_corpus.extend([self._preprocess_text(text) for text in texts])
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Store recipes
        self.recipes.extend(recipes)
        
    def search(
        self, 
        query: str, 
        k: int = 5, 
        alpha: float = 0.7
    ) -> List[Tuple[Dict, float]]:
        """
        Perform hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for semantic search (1-alpha for keyword search)
            
        Returns:
            List of (recipe, score) tuples
        """
        # Semantic search
        query_embedding = self.embedding_model.encode([query])
        semantic_distances, semantic_indices = self.index.search(query_embedding, len(self.recipes))
        semantic_scores = 1 / (1 + semantic_distances[0])  # Convert distances to similarities
        
        # Keyword search
        tokenized_query = self._preprocess_text(query)
        keyword_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Combine scores
        hybrid_scores = self._compute_hybrid_scores(semantic_scores, keyword_scores, alpha)
        
        # Get top k results
        top_k_indices = np.argsort(hybrid_scores)[-k:][::-1]
        
        return [
            (self.recipes[idx], hybrid_scores[idx])
            for idx in top_k_indices
        ] 