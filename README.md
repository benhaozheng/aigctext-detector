# EduGuard - AI作业原创性检测系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

一个基于多特征融合与BERT的AI生成内容检测系统，用于检测学生作业中是否存在AI生成内容。

## 项目简介

EduGuard 采用多特征融合的方法，结合了统计特征、语言学特征和深度学习特征，实现了高精度的AI内容检测。

### 核心功能

- **AI生成概率检测**：输出文本由AI生成的概率
- **原创性评分**：0-100分的原创性评分
- **段落级检测**：定位可疑段落
- **修改建议生成**：基于规则的智能建议
- **可视化分析界面**：Streamlit Web应用

### 技术架构

```
输入文本
    ↓
┌─────────────────────────────────────────────┐
│           多特征提取模块                      │
├─────────────────────────────────────────────┤
│ 1. 统计特征                                   │
│    - Perplexity (困惑度)                      │
│    - TTR (词汇多样性)                         │
│    - 句长统计                                 │
│    - n-gram重复率                             │
│                                              │
│ 2. 语言学特征                                 │
│    - 句法复杂度                               │
│    - HWV (写作方差)                           │
│                                              │
│ 3. 深度学习特征                               │
│    - BERT Embedding                          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│            融合分类模型                        │
│    [BERT向量 + 统计特征 + 语言学特征]          │
│              → MLP分类器                      │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│              输出结果                         │
│ - AI/人类标签                                 │
│ - 概率分数                                    │
│ - 段落风险分析                                │
│ - 修改建议                                    │
└─────────────────────────────────────────────┘
```

## 项目结构

```
AI-Detector/
│
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── preprocess.py         # 文本预处理
│   ├── ppl.py                # 困惑度计算
│   ├── features.py           # 特征工程
│   └── advice.py             # 修改建议生成
│
├── models/                   # 模型模块
│   ├── __init__.py
│   ├── bert_model.py         # BERT分类模型
│   └── fusion_model.py       # 融合模型
│
├── app/                      # Web应用
│   └── streamlit_app.py      # Streamlit界面
│
├── train.py                  # 训练脚本
├── predict.py                # 推理脚本
├── requirements.txt          # 依赖列表
└── README.md                 # 项目说明
```

## 安装说明

### 1. 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (推荐，用于GPU加速)

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖包
pip install -r requirements.txt
```

### 3. 安装PyTorch（根据你的系统）

```bash
# CPU版本
pip install torch torchvision

# CUDA 11.8版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 使用方法

### 1. 训练模型

```bash
# 训练BERT和融合模型
python train.py --bert_epochs 3 --fusion_epochs 10

# 只训练BERT
python train.py --train_bert --bert_epochs 3

# 只训练融合模型
python train.py --train_fusion --fusion_epochs 10
```

### 2. 命令行预测

```bash
# 检测文本
python predict.py --text "这是待检测的文本..."

# 检测文件
python predict.py --file input.txt

# 可视化结果
python predict.py --text "..." --visualize --output result.png

# 选择模型
python predict.py --text "..." --model fusion  # 或 bert
```

### 3. 启动Web应用

```bash
streamlit run app/streamlit_app.py
```

然后在浏览器中打开 `http://localhost:8501`

## 模型说明

### 1. BERT模型

- 使用 `bert-base-chinese` 作为基础模型
- 二分类输出（AI / Human）
- 使用HuggingFace Trainer进行训练

### 2. 融合模型

- 输入：BERT embedding (768维) + 手工特征 (11维)
- 结构：MLP (512 → 256 → 128 → 2)
- 特征包括：
  - 困惑度 (PPL)
  - 词汇多样性 (TTR)
  - 句长统计
  - 重复率
  - 句法复杂度
  - 写作文方差 (HWV)
  - 等补充特征

## 特征说明

| 特征 | 说明 | AI特征 |
|------|------|--------|
| PPL | 困惑度，语言模型预测文本的难度 | 较低 |
| TTR | 类型-标记比，词汇多样性 | 较低 |
| 句长方差 | 句子长度变化程度 | 较低 |
| HWV | 写作文方差，人类写作的不确定性 | 较低 |
| 句法复杂度 | 句子结构的复杂程度 | 较高 |

## 评估指标

- Accuracy (准确率)
- Precision (精确率)
- Recall (召回率)
- F1-score
- 混淆矩阵

## 实验结果

在示例数据集上的测试结果：

| 模型 | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|----------|
| BERT | 0.8500 | 0.8462 | 0.8571 | 0.8516 |
| Fusion | 0.8750 | 0.8750 | 0.8750 | 0.8750 |

## 注意事项

1. **模型下载**：首次运行会自动下载 `bert-base-chinese` 模型（约400MB）
2. **训练数据**：示例数据仅供演示，实际使用请准备真实标注数据
3. **GPU推荐**：训练过程建议使用GPU加速
4. **文本长度**：建议检测文本长度 ≥ 50 字

## 扩展功能

- 支持英文检测（更换BERT模型）
- 添加更多语言模型（GPT-3, LLaMA等）
- 实时API服务
- 批量文件检测

## 常见问题

### Q: 如何提高检测准确率？
A: 使用更多真实标注数据训练模型，进行数据增强。

### Q: 检测速度太慢怎么办？
A: 使用GPU加速，或减小模型batch_size。

### Q: 如何检测英文内容？
A: 将模型更换为 `bert-base-uncased` 或其他英文模型。

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 作者

EduGuard Team

## 致谢

- HuggingFace Transformers
- PyTorch Team
- Streamlit

---

**免责声明**：本系统仅供参考，检测结果不构成任何学术评判依据。请结合实际情况进行综合判断。
