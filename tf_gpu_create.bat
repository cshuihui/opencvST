@echo off
chcp 65001

cd /d %~dp0
REM docker build -t tf_gpu .
REM                   当前目录
docker run -it --gpus all -v D:\py_product\dl:/workspace -w /workspace --shm-size=6g --name tf_gpu tensorflow/tensorflow:latest-gpu bash -c "pip install opencv-contrib-python-headless openpyxl scikit-learn pandas && exec bash"


cmd /k
