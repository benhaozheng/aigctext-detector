# -*- coding: utf-8 -*-
"""
训练脚本
功能：训练BERT模型和融合模型，并进行评估
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
# 设置中文字体（自动检测）
from utils.font_config import setup_chinese_font
setup_chinese_font()

from models.bert_model import BERTClassifier
from models.fusion_model import FusionModel
from utils.preprocess import TextPreprocessor
from utils.features import FeatureExtractor


def load_sample_data(csv_path: str = "./data/sample_data.csv"):
    """
    加载训练数据
    优先从CSV文件加载，如果文件不存在则使用内置示例数据

    Args:
        csv_path: CSV文件路径（支持多个文件，用逗号分隔）

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    # 支持多个CSV文件
    csv_paths = [p.strip() for p in csv_path.split(',')]
    dfs = []

    for path in csv_paths:
        if os.path.exists(path):
            print(f"从CSV文件加载数据: {path}")
            try:
                # 处理BOM编码问题和格式问题
                if 'hc3' in path:
                    df = pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='skip')
                else:
                    df = pd.read_csv(path, on_bad_lines='skip')

                # 验证数据格式
                if 'text' not in df.columns or 'label' not in df.columns:
                    print(f"⚠️ 跳过 {path}：格式错误，需要 'text' 和 'label' 列")
                    continue

                # 清理数据
                df = df.dropna(subset=['text', 'label'])
                df['text'] = df['text'].astype(str)

                # 🔧 统一label格式（处理大小写不一致问题）
                df['label'] = df['label'].str.strip().str.lower()
                df['label'] = df['label'].replace({
                    'ai': 'AI',
                    'human': 'Human'
                })

                # 过滤空文本和无效标签
                df = df[df['text'].str.len() > 10]
                df = df[df['label'].isin(['AI', 'Human'])]

                if len(df) > 0:
                    dfs.append(df)
                    print(f"✅ {path}: 加载 {len(df)} 条数据")
                    print(f"   AI: {sum(df['label']=='AI')}, Human: {sum(df['label']=='Human')}")
                else:
                    print(f"⚠️ {path}: 没有有效数据")

            except Exception as e:
                print(f"⚠️ 加载 {path} 失败: {e}")

    # 合并所有数据
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\n✅ 总计加载 {len(combined_df)} 条数据")

        # 检查数据平衡性
        ai_count = sum(combined_df['label'] == 'AI')
        human_count = sum(combined_df['label'] == 'Human')
        print(f"   AI: {ai_count}, Human: {human_count}")

        # 数据平衡：如果比例超过2:1，则进行下采样
        if ai_count > 0 and human_count > 0:
            ratio = max(ai_count, human_count) / min(ai_count, human_count)
            if ratio > 2.0:
                print(f"\n⚠️ 数据不平衡 (比例 {ratio:.2f}:1)，进行平衡采样...")
                # 对多数类进行下采样
                min_count = min(ai_count, human_count)
                ai_df = combined_df[combined_df['label'] == 'AI'].sample(n=min(ai_count, min_count), random_state=42)
                human_df = combined_df[combined_df['label'] == 'Human'].sample(n=min(human_count, min_count), random_state=42)
                combined_df = pd.concat([ai_df, human_df], ignore_index=True)
                print(f"✅ 平衡后: {len(combined_df)} 条数据 (AI: {sum(combined_df['label']=='AI')}, Human: {sum(combined_df['label']=='Human')})")

        return combined_df
    else:
        print("\n⚠️ 没有成功加载任何CSV文件")
        print("使用内置示例数据")
        return _load_builtin_data()


def _load_builtin_data():
    """
    加载内置示例数据（当CSV文件不可用时使用）
    """
    # AI生成风格的文本示例
    ai_texts = [
        "人工智能是计算机科学的重要分支，它致力于研究如何使计算机能够模拟人类的智能行为。",
        "深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的特征表示。",
        "自然语言处理是人工智能的重要应用领域，它致力于让计算机能够理解和生成人类语言。",
        "机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习规律。",
        "神经网络是深度学习的基础，它由多个神经元层次组成，能够学习复杂的特征。",
        "卷积神经网络是一种专门用于处理图像数据的神经网络结构。",
        "循环神经网络是一种适合处理序列数据的神经网络结构。",
        "强化学习是机器学习的一个重要分支，它通过与环境交互来学习最优策略。",
        "计算机视觉是人工智能的重要应用领域，它致力于让计算机能够理解和分析图像。",
        "语音识别是自然语言处理的重要应用，它致力于将语音信号转换为文本。",
        "首先，人工智能技术的发展为各个行业带来了新的机遇。",
        "其次，深度学习技术的突破使得许多复杂问题得到了有效解决。",
        "此外，自然语言处理技术的进步使得人机交互变得更加自然。",
        "总的来说，人工智能技术正在深刻改变我们的生活方式。",
        "值得注意的是，人工智能技术的发展也带来了一些挑战和问题。",
        "综上所述，人工智能技术的发展前景广阔，但也需要谨慎对待。",
        "一方面，人工智能技术可以提高工作效率；另一方面，它也可能替代部分人类工作。",
        "数据是人工智能的基础，没有高质量的数据，就无法训练出好的模型。",
        "算法是人工智能的核心，好的算法能够更好地从数据中学习规律。",
        "算力是人工智能的支撑，强大的算力能够加速模型的训练和推理过程。",
        "人工智能技术的发展可以分为几个阶段：符号主义、连接主义和行为主义。",
        "机器学习算法可以分为监督学习、无监督学习和强化学习三大类。",
        "深度学习模型主要包括卷积神经网络、循环神经网络和Transformer等。",
        "自然语言处理任务包括文本分类、命名实体识别、情感分析等。",
        "计算机视觉任务包括图像分类、目标检测、语义分割等。",
    ]

    # 人类写作风格的文本示例
    human_texts = [
        "哎，今天天气真不错啊！我想起了小时候，那时候夏天总是那么漫长。",
        "妈妈总是在院子里摆个小桌子，我们就坐在那儿乘凉、吃西瓜。",
        "那种感觉，现在想起来还是甜甜的。不过话说回来，现在的孩子可能很难体会到了吧。",
        "昨天我去菜市场买菜，看到卖西瓜的大爷，突然就想起奶奶了。",
        "她以前总说挑西瓜要看纹路，还要听听声音，咚咚的才好。",
        "其实我也不懂，但每次买回来的西瓜确实挺甜的，可能是运气好吧。",
        "说真的，现在的生活节奏太快了，有时候真想慢下来，好好感受一下生活。",
        "前几天我去看了一场电影，挺感人的，看完之后想了很多。",
        "电影里的主人公经历了很多挫折，但最后还是坚持下来了，这让我很受启发。",
        "我觉得人生就是这样，总会遇到各种困难，但只要坚持下去，总会看到希望的。",
        "你知道吗，我最近开始学习做菜了，虽然做得不太好，但还是挺有成就感的。",
        "第一次做菜的时候，差点把厨房烧了，现在想想都后怕。",
        "不过慢慢地就熟练了，现在也能做出几道像样的菜了。",
        "我觉得做菜和做人一样，都需要耐心和用心，不能急于求成。",
        "昨天和朋友聊天，她说最近工作压力很大，每天都有做不完的事情。",
        "我告诉她，工作虽然重要，但身体更重要，要注意休息。",
        "其实我也经常熬夜，知道这样不好，但就是控制不住自己。",
        "可能这就是现代人的通病吧，总是忍不住刷手机，一刷就到深夜。",
        "我最近在尝试养成早睡早起的习惯，希望能坚持下去。",
        "早起的感觉真的很好，空气清新，心情也特别好。",
        "我特别喜欢早上的阳光，照在身上暖暖的，感觉一整天都会很美好。",
        "说起来，我小时候特别喜欢画画，虽然画得不怎么样，但就是喜欢。",
        "现在工作了，很少有时间画画了，有时候还挺怀念的。",
        "可能每个人都会有这样的遗憾吧，小时候喜欢做的事情，长大后却没有时间做了。",
        "我觉得人还是要有点爱好的，不然生活会很单调。",
        "我喜欢听音乐，特别是轻音乐，听了之后心情会变得很平静。",
    ]

    # 创建标签
    ai_labels = ['AI'] * len(ai_texts)
    human_labels = ['Human'] * len(human_texts)

    # 合并数据
    texts = ai_texts + human_texts
    labels = ai_labels + human_labels

    # 创建DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })

    return df


def prepare_training_data(df: pd.DataFrame, test_size: float = 0.2,
                          random_state: int = 42) -> tuple:
    """
    准备训练数据

    Returns:
        (train_texts, train_labels, val_texts, val_labels)
    """
    # 分割训练集和验证集
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()

    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()

    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(val_texts)}")
    print(f"训练集类别分布: {pd.Series(train_labels).value_counts().to_dict()}")
    print(f"验证集类别分布: {pd.Series(val_labels).value_counts().to_dict()}")

    return train_texts, train_labels, val_texts, val_labels


def train_bert_model(train_texts, train_labels, val_texts, val_labels, args):
    """训练BERT模型"""
    print("\n" + "="*50)
    print("开始训练BERT模型")
    print("="*50)

    # 创建模型
    bert = BERTClassifier(
        model_name="bert-base-chinese",
        num_labels=2,
        max_length=128
    )

    # 训练
    history = bert.train_model(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir="./checkpoints/bert",
        num_epochs=args.bert_epochs,
        batch_size=args.batch_size,
        learning_rate=args.bert_lr
    )

    # 评估
    print("\n=== BERT模型验证集评估 ===")
    predictions = bert.predict_batch(val_texts)
    pred_labels = [p['label'] for p in predictions]

    print_results(val_labels, pred_labels, "BERT")

    return bert


def train_fusion_model(train_texts, train_labels, val_texts, val_labels, args):
    """训练融合模型"""
    print("\n" + "="*50)
    print("开始训练融合模型")
    print("="*50)

    # 首先训练BERT获取embedding
    bert = BERTClassifier(model_name="bert-base-chinese", max_length=128)

    # 提取特征
    print("\n正在提取训练集特征...")
    preprocessor = TextPreprocessor()
    extractor = FeatureExtractor()

    train_features = []
    for text in train_texts:
        features = extractor.extract_all_features(text)
        train_features.append(features)

    val_features = []
    for text in val_texts:
        features = extractor.extract_all_features(text)
        val_features.append(features)

    # 创建融合模型
    fusion = FusionModel(
        bert_model=bert,
        feature_dim=11,
        hidden_dims=[256, 128, 64]
    )

    # 训练
    history = fusion.train(
        train_texts=train_texts,
        train_labels=train_labels,
        train_features=train_features,
        val_texts=val_texts,
        val_labels=val_labels,
        val_features=val_features,
        num_epochs=args.fusion_epochs,
        batch_size=args.batch_size,
        learning_rate=args.fusion_lr
    )

    # 评估
    print("\n=== 融合模型验证集评估 ===")
    predictions = fusion.predict_batch(val_texts, val_features)
    pred_labels = [p['label'] for p in predictions]

    print_results(val_labels, pred_labels, "Fusion")

    return fusion


def print_results(true_labels, pred_labels, model_name):
    """打印评估结果"""
    # 转换为数值
    label_map = {'Human': 0, 'AI': 1}
    true_nums = [label_map[l] for l in true_labels]
    pred_nums = [label_map[l] for l in pred_labels]

    # 计算指标
    accuracy = accuracy_score(true_nums, pred_nums)
    precision = precision_score(true_nums, pred_nums, average='binary', zero_division=0)
    recall = recall_score(true_nums, pred_nums, average='binary', zero_division=0)
    f1 = f1_score(true_nums, pred_nums, average='binary', zero_division=0)

    print(f"\n{model_name}模型评估结果:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels, target_names=['Human', 'AI']))

    # 绘制混淆矩阵
    plot_confusion_matrix(true_nums, pred_nums, model_name)


def plot_confusion_matrix(true_labels, pred_labels, model_name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'])
    plt.title(f'{model_name} - 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(f'./checkpoints/{model_name.lower()}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: ./checkpoints/{model_name.lower()}_confusion_matrix.png")


def compare_models(bert, fusion, val_texts, val_labels, val_features):
    """对比BERT和融合模型"""
    print("\n" + "="*50)
    print("模型对比分析")
    print("="*50)

    # BERT预测
    bert_preds = bert.predict_batch(val_texts)
    bert_probs = [p['ai_probability'] for p in bert_preds]
    bert_labels = [p['label'] for p in bert_preds]

    # 融合模型预测
    fusion_preds = fusion.predict_batch(val_texts, val_features)
    fusion_probs = [p['ai_probability'] for p in fusion_preds]
    fusion_labels = [p['label'] for p in fusion_preds]

    # 计算准确率
    label_map = {'Human': 0, 'AI': 1}
    true_nums = [label_map[l] for l in val_labels]
    bert_nums = [label_map[l] for l in bert_labels]
    fusion_nums = [label_map[l] for l in fusion_labels]

    bert_acc = accuracy_score(true_nums, bert_nums)
    fusion_acc = accuracy_score(true_nums, fusion_nums)

    print(f"\n准确率对比:")
    print(f"  BERT模型:   {bert_acc:.4f}")
    print(f"  融合模型:   {fusion_acc:.4f}")
    print(f"  提升:       {fusion_acc - bert_acc:+.4f}")

    # 绘制对比图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    models = ['BERT', 'Fusion']
    accs = [bert_acc, fusion_acc]
    bars = plt.bar(models, accs, color=['#3498db', '#e74c3c'])
    plt.ylabel('准确率')
    plt.title('模型准确率对比')
    plt.ylim(0, 1)

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.4f}', ha='center', fontsize=12)

    plt.subplot(1, 2, 2)
    plt.scatter(bert_probs, fusion_probs, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', label='y=x')
    plt.xlabel('BERT AI概率')
    plt.ylabel('Fusion AI概率')
    plt.title('AI预测概率对比')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./checkpoints/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n对比图已保存: ./checkpoints/model_comparison.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练AI检测模型')
    # 默认使用中文HC3数据集 + 示例数据
    parser.add_argument('--data', type=str,
                       default='./data/hc3_all_fixed.csv,./data/ai_vs_human_text_2026.csv,./data/sample_data.csv',
                       help='训练数据CSV文件路径（支持多个文件，用逗号分隔）')
    parser.add_argument('--bert_epochs', type=int, default=3, help='BERT训练轮数')
    parser.add_argument('--fusion_epochs', type=int, default=10, help='融合模型训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批大小')
    parser.add_argument('--bert_lr', type=float, default=2e-5, help='BERT学习率')
    parser.add_argument('--fusion_lr', type=float, default=1e-4, help='融合模型学习率')
    parser.add_argument('--train_bert', action='store_true', help='是否训练BERT')
    parser.add_argument('--train_fusion', action='store_true', help='是否训练融合模型')

    args = parser.parse_args()

    # 如果没有指定，默认都训练
    if not args.train_bert and not args.train_fusion:
        args.train_bert = True
        args.train_fusion = True

    # 创建必要的目录
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./data', exist_ok=True)

    # 加载数据
    print("加载训练数据...")
    df = load_sample_data(csv_path=args.data)
    print(f"数据集大小: {len(df)}")
    print(f"类别分布:\n{df['label'].value_counts()}")

    # 准备训练数据
    train_texts, train_labels, val_texts, val_labels = prepare_training_data(df)

    # 训练BERT
    bert = None
    if args.train_bert:
        bert = train_bert_model(train_texts, train_labels, val_texts, val_labels, args)

    # 训练融合模型
    fusion = None
    if args.train_fusion:
        fusion = train_fusion_model(train_texts, train_labels, val_texts, val_labels, args)

    # 模型对比
    if bert and fusion:
        # 提取验证集特征
        extractor = FeatureExtractor()
        val_features = [extractor.extract_all_features(text) for text in val_texts]
        compare_models(bert, fusion, val_texts, val_labels, val_features)

    print("\n" + "="*50)
    print("训练完成！")
    print("="*50)


if __name__ == '__main__':
    main()
