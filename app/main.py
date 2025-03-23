from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import timedelta
from typing import List, Dict, Optional
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

from .security.auth import (
    Token, create_access_token, get_current_user,
    verify_password, get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES
)
from .security.rate_limiter import rate_limit
from .rag.hybrid_search import HybridRecipeSearch
from .generation.recipe_generator import ContextAwareRecipeGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'logs/app.log',
            maxBytes=10000000,
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

app = FastAPI(
    title="Modern Recipe ChatBot API",
    description="An intelligent recipe recommendation system using RAG and LLMs",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
recipe_search = HybridRecipeSearch()
recipe_generator = ContextAwareRecipeGenerator(os.getenv("MODEL_PATH"))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"}
    )

@app.post("/token", response_model=Token)
@rate_limit()
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    try:
        if form_data.username != "demo" or not verify_password(form_data.password, get_password_hash("demo")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": form_data.username}, expires_delta=access_token_expires
        )
        logger.info(f"User {form_data.username} successfully logged in")
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        raise

@app.post("/recipes/search")
@rate_limit()
async def search_recipes(
    request: Request,
    query: str,
    k: Optional[int] = 5,
    alpha: Optional[float] = 0.7,
    current_user = Depends(get_current_user)
) -> List[Dict]:
    """Search for recipes using hybrid RAG approach"""
    try:
        logger.info(f"Searching recipes with query: {query}")
        results = recipe_search.search(query, k=k, alpha=alpha)
        return [{"recipe": recipe, "score": float(score)} for recipe, score in results]
    except Exception as e:
        logger.error(f"Recipe search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search recipes"
        )

@app.post("/recipes/generate")
@rate_limit()
async def generate_recipe(
    request: Request,
    ingredients: List[str],
    dietary_restrictions: Optional[List[str]] = None,
    cuisine_type: Optional[str] = None,
    cooking_time: Optional[int] = None,
    current_user = Depends(get_current_user)
) -> Dict:
    """Generate a recipe using context-aware LLM"""
    try:
        logger.info(f"Generating recipe with ingredients: {ingredients}")
        recipe = recipe_generator.generate_recipe(
            ingredients=ingredients,
            dietary_restrictions=dietary_restrictions,
            cuisine_type=cuisine_type,
            cooking_time=cooking_time
        )
        return recipe
    except Exception as e:
        logger.error(f"Recipe generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recipe"
        )

@app.post("/recipes/add")
@rate_limit()
async def add_recipes(
    request: Request,
    recipes: List[Dict],
    current_user = Depends(get_current_user)
):
    """Add new recipes to the vector store"""
    try:
        logger.info(f"Adding {len(recipes)} new recipes")
        recipe_search.add_recipes(recipes)
        return {"message": f"Successfully added {len(recipes)} recipes"}
    except Exception as e:
        logger.error(f"Recipe addition error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add recipes"
        ) 