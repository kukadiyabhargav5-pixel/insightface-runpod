# ==============================================================================
# InsightFace RunPod Serverless Worker - Production Dockerfile
# Optimized for NVIDIA CUDA GPU Inference & Sub-Second Worker Cold Start
# ==============================================================================

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV INSIGHTFACE_HOME=/root/.insightface

# Install system dependencies & Python runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt

# Pre-download and cache InsightFace 'buffalo_l' weights into Docker image
# This eliminates cold-start network downloads during serverless worker spin-up
RUN python3 -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1, det_size=(640, 640))"

# Copy application source code
COPY app /workspace/app
COPY handler.py /workspace/handler.py
COPY src /workspace/src

# Verify files exist
RUN ls -la /workspace && ls -la /workspace/src

# Entrypoint: Start official RunPod Serverless handler loop via src/handler.py
CMD ["python3", "-u", "src/handler.py"]

