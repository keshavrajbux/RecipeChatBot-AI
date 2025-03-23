# Modern Recipe ChatBot

A secure and intelligent recipe recommendation system using modern AI techniques and best practices.

## Features

- **Advanced AI Integration**:
  - RAG (Retrieval Augmented Generation) for accurate recipe recommendations
  - Vector embeddings using FAISS for semantic recipe search
  - Sentence transformers for natural language understanding

- **Security Features**:
  - JWT-based authentication
  - Rate limiting with Redis
  - Environment-based configuration
  - CORS protection
  - Input validation

- **Modern Architecture**:
  - FastAPI for high-performance async API
  - Redis for rate limiting and caching
  - FAISS for efficient vector similarity search
  - Modular and maintainable codebase

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RecipeChatBot-AI.git
cd RecipeChatBot-AI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

5. Start Redis server (required for rate limiting)

6. Run the application:
```bash
uvicorn app.main:app --reload
```

## API Documentation

Once running, visit:
- API documentation: http://localhost:8000/docs
- Alternative documentation: http://localhost:8000/redoc

## Security Notes

- Update `SECRET_KEY` in production
- Configure proper CORS settings
- Use proper user authentication
- Set appropriate rate limits
- Secure Redis instance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
