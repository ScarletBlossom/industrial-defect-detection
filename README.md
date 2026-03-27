# Industrial Defect Detection System

基于深度学习的工业产品表面缺陷检测系统，使用 ResNet18 进行图像分类。

## 项目结构

```
industrial-defect-detection/
├── src/
│   ├── config.py              # 配置文件
│   ├── data/
│   │   ├── dataset.py         # 数据集加载
│   │   └── generate_data.py  # 模拟数据生成
│   ├── models/
│   │   └── resnet_model.py    # ResNet18 模型
│   ├── train/
│   │   ├── trainer.py        # 训练逻辑
│   │   └── train.py          # 训练入口
│   ├── inference/
│   │   ├── predictor.py      # 推理模块
│   │   └── gradio_app.py     # Web 界面
│   └── utils/
│       └── visualization.py  # 可视化工具
├── data/                      # 数据目录
│   ├── train/                # 训练集
│   ├── val/                  # 验证集
│   └── test/                 # 测试集
├── checkpoints/               # 模型保存
├── logs/                     # 训练日志
├── run.py                    # 主入口脚本
└── README.md
```

## 环境要求

- Python 3.8+
- PyTorch
- TorchVision
- Gradio
- Pillow
- Matplotlib

安装依赖：
```bash
pip install torch torchvision gradio pillow matplotlib numpy
```

## 快速开始

### 方式一：使用命令行

```bash
# 运行交互式菜单
python run.py
```

选择对应选项：
- 1: 生成模拟数据集
- 2: 训练模型
- 3: 可视化训练结果
- 4: 启动 Gradio Web 界面
- 5: 运行完整流程

### 方式二：分步运行

```bash
# 1. 生成模拟数据集
python src/data/generate_data.py

# 2. 训练模型
python src/train/train.py

# 3. 可视化训练结果
python src/utils/visualization.py

# 4. 启动 Web 界面
python src/inference/gradio_app.py
```

## 训练配置

在 `src/config.py` 中可以修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| EPOCHS | 20 | 训练轮数 |
| BATCH_SIZE | 32 | 批大小 |
| LEARNING_RATE | 0.001 | 学习率 |
| IMAGE_SIZE | 224 | 图像尺寸 |

## 数据集

系统会自动生成模拟数据集，包含两类：
- **Normal**: 正常产品表面
- **Defect**: 有缺陷的产品表面

缺陷类型包括：划痕、凹坑、斑点、裂纹、变色等。

## 模型

- **架构**: ResNet18 (预训练)
- **输入**: 224x224 RGB 图像
- **输出**: 2 分类 (Normal/Defect)

## Web 界面

启动后访问 `http://localhost:7860`，上传图片即可进行推理。

界面功能：
- 图片上传
- 缺陷检测
- 置信度显示

## 训练可视化

训练完成后查看：
- `logs/training_history.json` - 训练历史数据
- `logs/training_curves.png` - 训练曲线图

## 推理

```python
from src.inference.predictor import DefectDetector

detector = DefectDetector('checkpoints/best_model.pth')
result = detector.predict('test_image.jpg')
print(result)
```

## License

MIT License