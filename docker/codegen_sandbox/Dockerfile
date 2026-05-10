FROM python:3.11-slim

# System dependencies for GDAL, GEOS, shapely, rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgdal-dev \
        gdal-bin \
        libgeos-dev \
        libproj-dev \
        libspatialindex-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python scientific / geo libraries
# No torch / GPU libs — this is a lightweight code sandbox
RUN pip install --no-cache-dir \
        numpy \
        scipy \
        pandas \
        pillow \
        opencv-python-headless \
        shapely \
        pyproj \
        affine \
        rasterio \
        "GDAL==$(gdal-config --version)" \
        scikit-learn

WORKDIR /sandbox
CMD ["python3"]
