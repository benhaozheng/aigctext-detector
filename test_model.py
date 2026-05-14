# -*- coding: utf-8 -*-
"""
测试模型预测是否正确
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.bert_model import BERTClassifier
from models.fusion_model import FusionModel
from utils.features import FeatureExtractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}\n")

# 加载BERT
print("=== 加载BERT模型 ===")
bert = BERTClassifier(
    model_name="bert-base-chinese",
    num_labels=2,
    device=device
)

if os.path.exists('./checkpoints/bert_zh'):
    bert.load_from_checkpoint('./checkpoints/bert_zh')
    print("BERT模型已加载\n")
else:
    print("警告: BERT检查点不存在，使用随机初始化\n")

# 加载Fusion模型（使用与训练时相同的hidden_dims）
print("=== 加载Fusion模型 ===")
fusion = FusionModel(
    bert_model=bert,
    feature_dim=11,
    device=device,
    hidden_dims=[256, 128, 64]
)

fusion_checkpoint = "./checkpoints/fusion_zh_best.pt"
if os.path.exists(fusion_checkpoint):
    fusion.load_model(fusion_checkpoint)
    print("Fusion模型已加载\n")
else:
    print("警告: Fusion检查点不存在，使用随机初始化\n")

# 测试文本
test_texts = [
    ("AI示例", "人工智能是计算机科学的重要分支，它致力于研究如何使计算机能够模拟人类的智能行为。深度学习是人工智能的核心技术之一，它通过多层神经网络来学习数据的特征表示。首先，神经网络的基本单元是神经元，它接收输入信号并产生输出。其次，多层神经网络可以学习更复杂的特征。"),
    ("人类示例", "哎，今天天气真不错啊！我想起了小时候，那时候夏天总是那么漫长。妈妈总是在院子里摆个小桌子，我们就坐在那儿乘凉、吃西瓜。那种感觉，现在想起来还是甜甜的。不过话说回来，现在的孩子可能很难体会到了吧。"),
    ("史铁生《我与地坛》", "说来奇怪，我单是这样站着，凝视着这古园的颓墙断壁，便觉得非常亲切。仿佛这古园的一切，都在向我诉说。那荒芜的草地，那古老的柏树，那残破的石阶，都有一种难以言说的韵味。我常常一个人来到这里，静静地坐着，什么也不想，只是感受这份宁静。"),
]

extractor = FeatureExtractor()

print("=== 预测测试 ===\n")

for name, text in test_texts:
    print(f"--- {name} ---")
    print(f"文本: {text[:50]}...")

    # BERT预测
    bert_result = bert.predict(text)
    print(f"BERT: 标签={bert_result['label']}, AI概率={bert_result['ai_probability']:.2%}")

    # Fusion预测
    features = extractor.extract_all_features(text)
    fusion_result = fusion.predict(text, features)
    print(f"Fusion: 标签={fusion_result['label']}, AI概率={fusion_result['ai_probability']:.2%}")
    print()

# 检查模型参数
print("=== 检查模型参数 ===")
print(f"BERT模型参数数量: {sum(p.numel() for p in bert.model.parameters()):,}")
print(f"Fusion MLP参数数量: {sum(p.numel() for p in fusion.mlp.parameters()):,}")

# 检查Fusion模型的第一层权重
print("\nFusion MLP第一层权重 (前10个值):")
first_layer_weight = list(fusion.mlp.network[0].parameters())[0].flatten()[:10]
print(first_layer_weight)
