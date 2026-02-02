#
# This example Dockerfile illustrates a method to install
# additional packages on top of NVIDIA's TensorFlow container image.
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

#RUN apt-get update && apt-get install -y --no-install-recommends \
#        my-package1 \
#        my-package2 \
#      && \
#    rm -rf /var/lib/apt/lists/

# Python packages from PyPI
RUN python3 -m pip install --no-cache-dir \
    "opencv-python-headless==4.8.1.78" \
    "pycocotools" \
    "pillow"
