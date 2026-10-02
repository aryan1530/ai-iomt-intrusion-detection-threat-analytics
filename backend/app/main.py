from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .logging_setup import request_logging_middleware, setup_logging
from .routers import router

setup_logging()
init_db()

app = FastAPI(title="IoMT Intrusion Detection & Threat Analytics", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(request_logging_middleware)
app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"service": "iomt-ids", "docs": "/docs", "feature_mode": settings.feature_mode}
