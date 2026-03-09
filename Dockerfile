# Eärendil - Kerr Black Hole Visualizer
# ======================================
# Multi-stage Docker build with CUDA support

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-cursor0 \
    libxcb-xfixes0 \
    libegl1 \
    libxkbcommon0 \
    libfontconfig1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements-docker.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY main.py .
COPY kerr_tracer.py .
COPY sky_data.py .

# Create cache directory for sky data
RUN mkdir -p /app/sky_cache

# Pre-download sky texture (optional - makes first run faster)
# RUN python -c "from sky_data import load_or_build_sky_texture; load_or_build_sky_texture()"

# Expose for potential web interface (future Gradio version)
EXPOSE 7860

# Default command
CMD ["python", "main.py"]
