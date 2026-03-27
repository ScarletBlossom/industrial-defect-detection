# 🔍 Industrial Defect Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/ScarletBlossom/industrial-defect-detection?style=social)](https://github.com/ScarletBlossom/industrial-defect-detection)

基于深度学习的工业产品表面缺陷检测系统，使用 **ResNet18** 进行图像分类，提供 **Gradio** Web 界面。

![Demo](https://img.shields.io/badge/Status-Ready-brightgreen)

## ✨ 功能特点

- 🎯 **ResNet18 预训练模型** + 迁移学习
- 📊 **模拟数据生成** - 无需真实数据即可快速验证
- 🖼️ **Gradio Web 界面** - 拖拽上传，实时推理
- 📈 **训练可视化** - 损失曲线、准确率曲线
- 🔄 **数据增强** - 翻转、旋转、颜色抖动
- ⚙️ **可配置训练** - 轮数、批次、学习率均可调

## 📁 项目结构

```
industrial-defect-detection/
├── src/
│   ├── config.py              # 配置参数
│   ├── data/
│   │   ├── dataset.py         # 数据集加载
│   │   └── generate_data.py   # 模拟数据生成
│   ├── models/
│   │   └── resnet_model.py    # ResNet18 模型
│   ├── train/
│   │   ├── trainer.py         # 训练逻辑
│   │   └── train.py           # 训练入口
│   ├── inference/
│   │   ├── predictor.py       # 推理模块
│   │   └── gradio_app.py      # Web 界面
│   └── utils/
│       └── visualization.py   # 可视化工具
├── data/                      # 数据目录
├── checkpoints/               # 模型保存
├── logs/                      # 训练日志
├── run.py                     # 主入口脚本
├── LICENSE                    # MIT 协议
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision gradio pillow matplotlib numpy
```

### 2. 生成模拟数据

```bash
python src/data/generate_data.py
```

### 3. 训练模型

```bash
python src/train/train.py
```

### 4. 启动 Web 界面

```bash
python src/inference/gradio_app.py
```

然后浏览器打开 **http://localhost:7860**

![Gradio Interface](https://via.placeholder.com/800x400?text=Gradio+Web+Interface)

## 🎮 交互式运行

直接运行主脚本，选择功能：

```bash
python run.py
```

菜单选项：
- `1` - 生成模拟数据集
- `2` - 训练模型
- `3` - 可视化训练结果
- `4` - 启动 Web 界面
- `5` - 运行完整流程

## ⚙️ 配置参数

在 `src/config.py` 中修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EPOCHS` | 20 | 训练轮数 |
| `BATCH_SIZE` | 32 | 批大小 |
| `LEARNING_RATE` | 0.001 | 学习率 |
| `IMAGE_SIZE` | 224 | 输入图像尺寸 |
| `NUM_CLASSES` | 2 | 分类数量 |

## 📊 数据集

系统自动生成两类模拟图像：

| 类别 | 说明 |
|------|------|
| **Normal** | 正常产品表面 |
| **Defect** | 有缺陷的产品表面 |

缺陷类型：划痕、凹坑、斑点、裂纹、变色等

## 🔬 模型架构

```
ResNet18 (预训练权重)
    ↓
修改最后一层 fc
    ↓
2 分类输出 (Normal / Defect)
```

## 💻 API 使用

```python
from src.inference.predictor import DefectDetector

# 加载模型
detector = DefectDetector('checkpoints/best_model.pth')

# 推理
result = detector.predict('product_image.jpg')
# {'class': 'Normal', 'confidence': 0.95, 'probabilities': [0.95, 0.05]}
```

## 📈 训练可视化

训练完成后查看：
- `logs/training_history.json` - 训练历史
- `logs/training_curves.png` - 训练曲线

## 🛠️ 技术栈

- **Python** 3.8+
- **PyTorch** 2.0+ - 深度学习框架
- **TorchVision** - 预训练模型与数据增强
- **Gradio** - Web 界面
- **Pillow** - 图像处理
- **Matplotlib** - 可视化

## 📝 License

MIT License - 详见 [LICENSE](LICENSE)

---

⭐ 如果对你有帮助，欢迎 Star！
