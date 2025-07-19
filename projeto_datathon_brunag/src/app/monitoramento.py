"""
app/monitoramento.py

Middleware de logging de requisições HTTP.
"""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("monitoramento")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs_api.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class LogRequestMiddleware(BaseHTTPMiddleware):
    """Registra método, endpoint, tempo e corpo da requisição."""

    async def dispatch(self, request: Request, call_next):
        inicio = time.time()
        body = await request.body()
        response = await call_next(request)
        duracao = time.time() - inicio

        logger.info(
            f"{request.method} {request.url.path} | "
            f"Tempo: {duracao:.3f}s | Status: {response.status_code} | "
            f"Body: {body.decode('utf-8')}"
        )
        return response
