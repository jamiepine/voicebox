# Run with Docker

Quick steps to build and run the project using Docker Compose:

Build and start both services:

```bash
docker-compose up --build
```

This will:
- Build the Python backend (exposes port 8000)
- Build the frontend and serve it via nginx (exposes port 3000 -> nginx:80)

Backend data is persisted to a Docker volume mounted at `/data` in the container.

Notes:
- The backend installs large ML dependencies (torch, transformers) and may result in a large image.
- You can pass a custom data directory by mounting a host folder to the `data` volume in `docker-compose.yml`.
