项目简介
这是一个从零开始实现的Transformer模型，专为教学和学习目的设计。项目完整实现了Transformer的核心组件，包括多头自注意力机制、位置编码、前馈网络等，并在Tiny Shakespeare数据集上进行了语言建模任务的训练。
项目结构
transformer-from-scratch/
├── src/                    # 源代码目录
│   ├── main.py             # 主训练脚本
│   ├── model.py            # Transformer模型定义
│   ├── data_loader.py      # 数据加载和预处理
│   └── utils.py            # 工具函数
├── configs/                # 配置文件目录
│   └── base.yaml           # 基础配置文件
├── scripts/                # 可执行脚本
│   ├── train.py            # 训练脚本
│   └── generate.py         # 文本生成脚本
├── checkpoints/            # 模型检查点保存目录
├── requirements.txt         # Python依赖包列表
└── README.md               # 项目说明文档
环境要求
硬件要求
GPU: NVIDIA GPU (推荐GTX 1060 6GB或以上)
显存: 至少4GB
内存: 8GB或以上
软件要求
Python 3.8+
PyTorch 1.9+
其他依赖见requirements.txt
快速开始
1. 环境配置
# 克隆项目
git clone https://github.com/your-username/transformer-from-scratch.git
cd transformer-from-scratch

# 创建conda环境
conda create -n transformer python=3.9
conda activate transformer

# 安装依赖
pip install -r requirements.txt
2. 训练模型
基础训练命令：
cd src
python main.py
使用配置文件训练：
python scripts/train.py --config configs/base.yaml --seed 42
完整训练命令（推荐）：
python scripts/train.py \
    --config configs/base.yaml \
    --seed 42 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --num_epochs 20
3. 文本生成
训练完成后，可以使用训练好的模型生成文本：
python scripts/generate.py \
    --model_path checkpoints/best_model.pth \
    --prompt "ROMEO:" \
    --max_len 100 \
    --temperature 0.8
    详细配置说明
模型配置 (configs/base.yaml)
model:
  d_model: 128          # 模型维度
  n_head: 4             # 注意力头数
  num_encoder_layers: 2 # 编码器层数
  num_decoder_layers: 2 # 解码器层数
  d_ff: 512             # 前馈网络维度
  dropout: 0.1          # Dropout比率
  max_seq_len: 1024     # 最大序列长度

training:
  batch_size: 64        # 批大小
  learning_rate: 3e-4   # 学习率
  num_epochs: 20        # 训练轮数
  warmup_steps: 1000    # 学习率预热步数
  grad_clip: 1.0        # 梯度裁剪阈值
  Transformer from Scratch - 中文README
项目简介
这是一个从零开始实现的Transformer模型，专为教学和学习目的设计。项目完整实现了Transformer的核心组件，包括多头自注意力机制、位置编码、前馈网络等，并在Tiny Shakespeare数据集上进行了语言建模任务的训练。
项目结构
transformer-from-scratch/
├── src/                    # 源代码目录
│   ├── main.py             # 主训练脚本
│   ├── model.py            # Transformer模型定义
│   ├── data_loader.py      # 数据加载和预处理
│   └── utils.py            # 工具函数
├── configs/                # 配置文件目录
│   └── base.yaml           # 基础配置文件
├── scripts/                # 可执行脚本
│   ├── train.py            # 训练脚本
│   └── generate.py         # 文本生成脚本
├── checkpoints/            # 模型检查点保存目录
├── requirements.txt         # Python依赖包列表
└── README.md               # 项目说明文档
环境要求
硬件要求
GPU: NVIDIA GPU (推荐GTX 1060 6GB或以上)
显存: 至少4GB
内存: 8GB或以上
软件要求
Python 3.8+
PyTorch 1.9+
其他依赖见requirements.txt
快速开始
1. 环境配置
# 克隆项目
git clone https://github.com/your-username/transformer-from-scratch.git
cd transformer-from-scratch

# 创建conda环境
conda create -n transformer python=3.9
conda activate transformer

# 安装依赖
pip install -r requirements.txt
2. 训练模型
基础训练命令：
cd src
python main.py
使用配置文件训练：
python scripts/train.py --config configs/base.yaml --seed 42
完整训练命令（推荐）：
python scripts/train.py \
    --config configs/base.yaml \
    --seed 42 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --num_epochs 20
3. 文本生成
训练完成后，可以使用训练好的模型生成文本：
python scripts/generate.py \
    --model_path checkpoints/best_model.pth \
    --prompt "ROMEO:" \
    --max_len 100 \
    --temperature 0.8
详细配置说明
模型配置 (configs/base.yaml)
model:
  d_model: 128          # 模型维度
  n_head: 4             # 注意力头数
  num_encoder_layers: 2 # 编码器层数
  num_decoder_layers: 2 # 解码器层数
  d_ff: 512             # 前馈网络维度
  dropout: 0.1          # Dropout比率
  max_seq_len: 1024     # 最大序列长度

training:
  batch_size: 64        # 批大小
  learning_rate: 3e-4   # 学习率
  num_epochs: 20        # 训练轮数
  warmup_steps: 1000    # 学习率预热步数
  grad_clip: 1.0        # 梯度裁剪阈值
数据配置
项目默认使用Tiny Shakespeare数据集，自动从网络下载。如需使用其他数据集，可修改data_loader.py中的配置。
核心功能
1. 模型架构
✅ 缩放点积注意力 (Scaled Dot-Product Attention)
✅ 多头自注意力 (Multi-Head Attention)
✅ 位置编码 (Positional Encoding)
✅ 位置前馈网络 (Position-wise FFN)
✅ 残差连接 + 层归一化 (Residual + LayerNorm)
✅ 编码器-解码器架构 (Encoder-Decoder)
2. 训练特性
✅ 学习率调度 (Learning Rate Scheduling)
✅ 梯度裁剪 (Gradient Clipping)
✅ 模型保存/加载 (Model Checkpointing)
✅ 训练可视化 (Training Visualization)
✅ 早停机制 (Early Stopping)
3. 评估指标
训练/验证损失 (Training/Validation Loss)
困惑度 (Perplexity)
文本生成质量 (Text Generation Quality)
