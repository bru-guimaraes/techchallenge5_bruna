import time
import logging
from fastapi import Request

# Configurar o logger
logger = logging.getLogger("monitoramento")
logger.setLevel(logging.INFO)

# Arquivo de log
handler = logging.FileHandler("logs_api.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Middleware de monitoramento
async def log_request(request: Request, call_next):
    inicio = time.time()
    body = await request.body()
    response = await call_next(request)
    duracao = time.time() - inicio

    logger.info(
        f"{request.method} {request.url.path} | Tempo: {duracao:.3f}s | Status: {response.status_code} | Body: {body.decode('utf-8')}"
    )

    return response
