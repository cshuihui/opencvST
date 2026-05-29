FROM tensorflow/tensorflow:latest-gpu

WORKDIR /workspace

RUN pip install opencv-contrib-python-headless openpyxl scikit-learn pandas