# 独立 OCR 服务镜像（GPU 版，含 paddlepaddle-gpu）。
# 完全自包含：构建 context = 本仓库根，只 COPY 本仓库自带的 common/ ocr/ ocr_service/，
#   不依赖任何外部仓库（与「合同雷达」前后端零耦合）。
# 基础镜像：CUDA 11.8 + cuDNN8 runtime（Ubuntu 22.04，Python 3.10）。
#   选 11.8 而非 12.x：目标服务器驱动 535.x 原生支持到 CUDA 12.2，11.8（<12.2）零兼容风险。
# 宿主须装 NVIDIA 驱动 + nvidia-container-toolkit，容器才能拿到 GPU。
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
# python + opencv/paddle 运行时系统库。
# - libgl1 / libglib2.0-0：opencv-contrib 运行时。
# - libgomp1：paddle 算子用 OpenMP。
RUN apt-get update -o Acquire::Retries=8 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=8 \
        python3 python3-pip \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 22.04 自带 pip 偏旧，先升级（paddle 大轮子对新 pip 更友好）。
RUN python3 -m pip install --no-cache-dir --upgrade pip

# GPU 版 paddle（CUDA 11.8 轮子，paddle 官方源）。
# 若该源无 3.2.2/cu118：改 cu126 源 + 基础镜像换 12.6/24.04，靠 CUDA 12 次版本兼容（535>=525 基线）。
# 纯 CPU 部署：删本行，requirements.txt 加回 paddlepaddle==3.2.2，基础镜像可换 python:3.12。
RUN python3 -m pip install --no-cache-dir \
        paddlepaddle-gpu==3.2.2 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 其余 OCR 依赖（不含 paddlepaddle，避免覆盖 GPU 版）
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# 本仓库自带源码（通用解析管线 + 服务壳 + tar/id 工具）
COPY common/ ./common/
COPY ocr/ ./ocr/
COPY ocr_service/ ./ocr_service/

ENV OCR_HOST=0.0.0.0 \
    OCR_PORT=8001
EXPOSE 8001
# 模型经卷挂载：./models→/app/models（PP-DocLayoutV2）、命名卷→/root/.paddlex（PP-OCRv6 等首跑下载）。
CMD ["python3", "-m", "ocr_service"]
