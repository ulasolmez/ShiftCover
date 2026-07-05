# ── Simplex – Multi-stage Docker build ───────────────────────────────────
# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# Stage 2: Production image
FROM python:3.11-slim
WORKDIR /app

# Install system deps for ortools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install backend deps
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r backend-requirements.txt

# Copy backend + solver
COPY backend/ ./backend/
COPY solver.py .

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./static

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]