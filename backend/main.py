"""
Simplex – FastAPI Backend
=========================
Serves the shift-covering solver as a REST API with SSE streaming.
"""

import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path so we can import solver.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api import solve, codes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown logic."""
    yield


app = FastAPI(
    title="Simplex API",
    description="Weekly Shift-Covering Optimiser",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local development; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────
app.include_router(solve.router, prefix="/api", tags=["solve"])
app.include_router(codes.router, prefix="/api", tags=["codes"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}