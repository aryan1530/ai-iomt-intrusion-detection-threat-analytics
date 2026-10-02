import json
import logging
import sys
import time
from typing import Callable

from fastapi import Request

from .config import settings


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def request_logging_middleware(request: Request, call_next: Callable):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logging.getLogger("api").info(
            json.dumps(
                {
                    "path": request.url.path,
                    "method": request.method,
                    "status": response.status_code,
                    "elapsed_ms": round(elapsed_ms, 2),
                }
            )
        )
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logging.getLogger("api").exception(
            json.dumps(
                {
                    "path": request.url.path,
                    "method": request.method,
                    "status": 500,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "error": True,
                }
            )
        )
        raise
