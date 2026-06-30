Phase 10: Deployment
Dockerfile (new file, project root):
dockerfileFROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
docker-compose.yml (new file, project root):
yamlversion: "3.9"

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - OLLAMA_MODEL=qwen2.5:7b
    volumes:
      - ./data:/app/data
      - ./chroma_db:/app/chroma_db
    depends_on:
      - ollama

volumes:
  ollama_data:
.dockerignore (new file, project root):
venv/
__pycache__/
*.pyc
.env
data/chat.db
chroma_db/
.git/
.gitignore
*.md
Run command:
bashdocker-compose up --build
docker exec -it <ollama_container_name> ollama pull qwen2.5:7b
README addition — add near the top, under the title:
markdown![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.35-red)
Commit:
bashgit add Dockerfile docker-compose.yml .dockerignore README.md
git commit -m "feat: add Docker deployment with docker-compose and Ollama service"