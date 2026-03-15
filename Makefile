.PHONY: help dev-frontend dev-backend dev

help:
	@echo "Available commands:"
	@echo "  make dev-frontend    - Starts the frontend development server (Vite)"
	@echo "  make dev-backend     - Starts the backend development server (Uvicorn with reload)"
	@echo "  make dev             - Starts both frontend and backend development servers"

dev-frontend:
	@echo "Starting frontend development server..."
	@cd frontend && npm run dev

dev-backend:
	@echo "Starting backend development server..."
	@cd backend && uv run uvicorn research_engine.app:app --host 0.0.0.0 --port 2024 --reload

# Run frontend and backend concurrently
dev:
	@echo "Starting both frontend and backend development servers..."
	@powershell -Command "Start-Process cmd -ArgumentList '/c cd frontend && npm run dev'; cd backend; uv run uvicorn research_engine.app:app --host 0.0.0.0 --port 2024 --reload"
