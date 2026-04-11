FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（Milvus/FastEmbed 可能需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml* requirements.txt* ./

# 安装 Python 依赖
RUN pip install --no-cache-dir -e .

# 复制应用代码
COPY . .

# 默认命令
CMD ["wbot", "agent"]
