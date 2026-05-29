@echo off
chcp 65001

docker start tf_gpu
docker exec -it tf_gpu bash

cmd /k
