# OCR 服务（独立部署，Ubuntu + GPU）

通用文档解析 HTTP 服务：上传 PDF → **版面检测 / 文字识别 / 表格 / 印章文字** → 打包 tar 返回。
**完全独立**：自带全部源码，构建期不依赖任何外部仓库；只暴露 HTTP，调用方（如「合同雷达」前后端）经 `OCR_SERVICE_URL` 接入。

```
┌─────────────────────────┐        HTTP :8001          ┌──────────────────────────┐
│  本服务（OCR，独立仓）   │  ←───────────────────────  │  调用方（合同前后端等）    │
│  GPU · paddle 全栈       │   /ingest /reocr /health   │  经 OCR_SERVICE_URL 接入  │
└─────────────────────────┘                            └──────────────────────────┘
        可同机、可异机。本服务自管生命周期，与调用方各自 up / down。
```

接口：
- `GET  /health` → `{"ok": true}`
- `POST /ingest` （form：`file`=PDF，可选 `contract_id`）→ 通用解析，返回 `derived/<id>/` 的 tar（`application/x-tar`）
- `POST /reocr` （form：`file`=旧 derived tar，`contract_id`）→ 重跑文本，返回新 document

---

## 一、前置（服务器一次性装好）

### 1.1 Docker + compose v2
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER        # 免 sudo；执行后重新登录
docker --version && docker compose version
```

### 1.2 NVIDIA 驱动
```bash
nvidia-smi
```
- 没装：`sudo ubuntu-drivers autoinstall` 后重启。
- 镜像用 **CUDA 11.8**（驱动 535.x 原生支持到 12.2，11.8 零兼容风险），无需追新驱动。

### 1.3 nvidia-container-toolkit（容器拿 GPU 的关键，别漏）
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
**先验容器能看到 GPU**（过了再谈构建）：
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 二、取代码 + 准备模型

```bash
git clone <本仓库地址> OCR
cd OCR
```

**版面模型不入库**（`.gitignore` 排除 `models/`），需单独放到 `./models/pp_doclayout_v2/`（PP-DocLayoutV2，约 204 MB）：
```bash
# 从已有的合同雷达开发机推过来（示例）
rsync -avz <开发机>:~/coding/contract_radar/models/pp_doclayout_v2/  ./models/pp_doclayout_v2/
# 应有 inference.json / inference.pdiparams / config.json 等
```
> PP-OCRv6 / 印章 / 表格模型**不必手动传**：首次入库会自动联网下载到 `paddlex-cache` 命名卷，之后复用。

---

## 三、构建并启动

```bash
docker compose up -d --build      # 首次较慢：拉 CUDA 基础镜像 + paddle GPU 轮子（多 GB），需联网
docker compose ps
docker compose logs -f ocr-service
```

---

## 四、验证

```bash
# 1) 服务活着
curl http://localhost:8001/health                 # 期望 {"ok": true}

# 2) GPU 在容器里可见（已钉 1 号卡，应只列出一张卡；GPU 0 留给宿主的 Qwen）
docker compose exec ocr-service nvidia-smi

# 3) paddle 走 GPU
docker compose exec ocr-service python3 -c "import paddle; paddle.utils.run_check()"
#   末尾 'PaddlePaddle is installed successfully!' 且提示用到 GPU 即成功

# 4) 端到端：一份 PDF 走通解析
curl -F file=@sample.pdf http://localhost:8001/ingest -o out.tar && tar tf out.tar | head
```

> **GPU 利用范围（如实说明）**：文字识别 / 表格 / 印章（PaddleOCR、PaddleX pipeline）会自动用 GPU；
> **版面检测 `ocr/probe_layout.py` 目前写死 `device="cpu"`**。要让版面也走 GPU，把该文件里
> `create_model(..., device="cpu")` 改成 `device="gpu"` 重新 `--build` 即可（本仓库自持源码，可放心改）。

---

## 五、调用方接入（合同前后端示例）

调用方容器里设 `OCR_SERVICE_URL` 指到本服务：
```dotenv
# 同机：调用方容器经 host-gateway 访问宿主发布的 8001
OCR_SERVICE_URL=http://host.docker.internal:8001
# 异机：填本服务所在机器的 IP/域名（确认防火墙放行 8001）
# OCR_SERVICE_URL=http://192.168.1.50:8001
```

---

## 六、独立部署到「无源码」服务器（可选）

要在 OCR 服务器上**连本仓库源码都不放**，构建一次后只搬镜像：
```bash
# 有源码的机器
docker compose build
docker save cr-ocr-gpu | gzip > cr-ocr-gpu.tar.gz
# 拷到 OCR 服务器，只需：镜像 + docker-compose.yml + models/
docker load < cr-ocr-gpu.tar.gz
docker compose up -d            # build 段不会重复跑（镜像已在）
```

---

## 七、常见问题

| 现象 | 排查 |
|---|---|
| 报 `could not select device driver ... gpu` | nvidia-container-toolkit 没装好/docker 没重启；先过 §1.3 的 `docker run --gpus all ... nvidia-smi`。 |
| `run_check()` 显示用 CPU | toolkit 未配默认 runtime，或装成了 CPU 版 paddle；`docker compose exec ocr-service nvidia-smi` 看容器有没有卡。 |
| 构建拉 paddle 轮子慢/失败 | paddle GPU 栈多 GB；网络受限就配 pip 镜像源，或在能联网机器预构建 `docker save`/`load`（见 §六）。 |
| paddle 源里没有 3.2.2/cu118 | 改 cu126 源 + 基础镜像换 `nvidia/cuda:12.6.x-cudnn-runtime-ubuntu24.04`（需 `--break-system-packages`），靠 CUDA 12 次版本兼容在 535 驱动上跑。 |
| 容器里看到 GPU 0（带 Qwen）而非 1 号卡 | `docker-compose.yml` 的 `device_ids: ['1']` 没生效；确认 compose 支持 `deploy.resources`，或临时改 `count: all`。 |
| 首次入库很慢/报模型源错 | 首跑下载 paddlex 模型到 `paddlex-cache` 卷需外网；离线要预置该卷。 |
| `/ingest` 报找不到版面模型 | `./models/pp_doclayout_v2/` 没放对；见 §二。 |

---

## 源码来源与维护边界

本仓库的 `ocr_service/`、`ocr/`（4 个通用 stage）、`common/`（bundle/ids）是从「合同雷达」(`contract_radar`) **搬出独立**的解析核心，与那边解耦自管。
两边各持一份的 `common/bundle.py`（tar 打包格式）、`common/ids.py`（合同号规整）、`ocr/build_document.py` 是**线协议级契约**——改动其中的打包格式/ID 规则时，须与调用方保持兼容。
