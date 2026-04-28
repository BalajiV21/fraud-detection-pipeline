.PHONY: help build up down logs train test clean ps restart

help:
	@echo "Available commands:"
	@echo "  make build     - Build all Docker images"
	@echo "  make up        - Start all services in background"
	@echo "  make down      - Stop all services"
	@echo "  make logs      - Tail logs from all services"
	@echo "  make ps        - Show running containers"
	@echo "  make train     - Trigger model training"
	@echo "  make test      - Run pytest test suite"
	@echo "  make restart   - Restart all services"
	@echo "  make clean     - Stop + remove volumes (WARNING: deletes models)"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

train:
	curl -X POST http://localhost:8001/train

test:
	pytest tests/ -v

restart:
	docker compose restart

clean:
	docker compose down -v
