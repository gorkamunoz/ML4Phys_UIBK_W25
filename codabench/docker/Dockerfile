FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git wget curl tini \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    Cython==0.29.30 \
    numpy==1.22.4 \
    scipy==1.8.1 \
    scikit-learn==1.1.1 \
    pandas==1.4.2 \
    pyyaml==6.0 \
    imutils==0.5.4 \
    numba==0.55.2 \
    fastai

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu


