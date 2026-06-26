# GPU image with PyTorch + Ultralytics preinstalled.
# Requires NVIDIA Container Toolkit on the host for --gpus.
FROM ultralytics/ultralytics:latest

WORKDIR /app

COPY pyproject.toml README.md cv_agent.yaml coco128.yaml dataset.yaml.example ./
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Persist runs, Optuna DB, and logs on a mounted volume (see README).
VOLUME ["/app/runs"]

ENTRYPOINT ["cv_agent"]
CMD ["run"]
