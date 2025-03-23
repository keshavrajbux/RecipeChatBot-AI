from fastapi import HTTPException, Request
from redis import Redis
import os
from dotenv import load_dotenv
import time
from typing import Callable
from functools import wraps

load_dotenv()

class RateLimiter:
    def __init__(self):
        self.redis_client = Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
        self.rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        self.window = 60  # 1 minute window

    async def check_rate_limit(self, request: Request):
        client_ip = request.client.host
        current = time.time()
        key = f"rate_limit:{client_ip}"
        
        # Create a pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # Remove old requests
        pipe.zremrangebyscore(key, 0, current - self.window)
        
        # Count requests in current window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current): current})
        
        # Set expiry
        pipe.expire(key, self.window)
        
        # Execute pipeline
        _, request_count, *_ = pipe.execute()
        
        if request_count > self.rate_limit:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )

def rate_limit():
    limiter = RateLimiter()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get('request')
            if request:
                await limiter.check_rate_limit(request)
            return await func(*args, **kwargs)
        return wrapper
    return decorator 