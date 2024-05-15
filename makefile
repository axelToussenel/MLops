# Variables
COMPOSE_FILE = docker-compose.yml
AIRFLOW_IMAGE = apache/airflow:2.8.3
MLFLOW_IMAGE = python:3.9.18-slim

# Phony targets
.PHONY: help build build-airflow build-mlflow up start stop down clean logs

# Default target
help:
	@echo "Usage:"
	@echo "  make build          - Build all Docker images"
	@echo "  make build-airflow  - Build the Airflow Docker image"
	@echo "  make build-mlflow   - Build the MLflow Docker image"
	@echo "  make up             - Start all services"
	@echo "  make start          - Start services in the background"
	@echo "  make stop           - Stop running services"
	@echo "  make down           - Stop and remove all services"
	@echo "  make clean          - Remove all containers, networks, and volumes"
	@echo "  make logs           - Tail logs of all services"

# Build all Docker images
build: build-airflow build-mlflow

# Build the Airflow Docker image
build-airflow:
	docker-compose -f $(COMPOSE_FILE) build airflow-webserver airflow-scheduler airflow-worker airflow-triggerer airflow-init airflow-cli

# Build the MLflow Docker image
build-mlflow:
	docker-compose -f $(COMPOSE_FILE) build mlflow

# Start all services in the foreground
up: build
	docker-compose -f $(COMPOSE_FILE) up

# Start all services in the background
start: build
	docker-compose -f $(COMPOSE_FILE) up -d

# Stop running services
stop:
	docker-compose -f $(COMPOSE_FILE) stop

# Stop and remove all services
down:
	docker-compose -f $(COMPOSE_FILE) down

# Remove all containers, networks, and volumes
clean: down
	docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans

# Tail logs of all services
logs:
	docker-compose -f $(COMPOSE_FILE) logs -f
