"""
main.py â€” Project Aether FastAPI Application Entry Point
=========================================================
Wires together the FastAPI app, CORS, logging, and the ML routes.
Run with: uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s â€” %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("aether")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Project Aether â€” Liquid Neural CDE API",
    description=(
        "Production ML API for continuous-time healthcare claim-denial "
        "prediction using Liquid Neural CDEs (Neural Controlled Differential "
        "Equations with Liquid Time-Constant vector fields)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS â€” allow local dev by default, with production origins from env
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = settings.parsed_cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
    return response

# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )

# ---------------------------------------------------------------------------
# Health & root
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {
        "project":     "Project Aether",
        "description": "Liquid Neural CDE â€” Healthcare Claim Denial Prediction",
        "version":     "1.0.0",
        "docs":        "/docs",
        "status":      "online",
    }


@app.get("/health", tags=["System"])
async def health():
    """Quick liveness probe for orchestrators (Kubernetes, Docker, etc.)."""
    import torch
    return {
        "status":       "ok",
        "torch":        torch.__version__,
        "cuda":         torch.cuda.is_available(),
        "device":       "cuda" if torch.cuda.is_available() else "cpu",
    }

# ---------------------------------------------------------------------------
# Mount ML router
# ---------------------------------------------------------------------------

app.include_router(router)

logger.info("Project Aether backend started â€” visit http://localhost:8000/docs")
