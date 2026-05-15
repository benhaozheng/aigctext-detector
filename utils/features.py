# -*- coding: utf-8 -*-
"""
特征工程模块
功能：提取统计特征和语言学特征
"""

import numpy as np
from typing import List, Dict, Any
import jieba
import jieba.posseg as pseg
from collections import Counter
import re


class FeatureExtractor:
    """特征提取器"""

    def __init__(self):
        """初始化特征提取器"""
        jieba.initialize()

        # 词性标签
        self.pos_tags = {
            'n': '名词', 'v': '动词', 'a': '形容词', 'd': '副词',
            'm': '数词', 'q': '量词', 'p': '介词', 'c': '连词',
            'u': '助词', 'r': '代词', 'w': '标点'
        }

    def extract_all_features(self, text: str, ppl: float = None, preprocessor=None) -> Dict[str, float]:
        """
        提取所有特征

        Args:
            text: 输入文本
            ppl: 已计算的困惑度（可选）
            preprocessor: 文本预处理器（可选）

        Returns:
            特征字典
        """
        if not text:
            return self._get_zero_features()

        # 如果没有提供预处理器，使用默认的
        if preprocessor is None:
            from .preprocess import TextPreprocessor
            preprocessor = TextPreprocessor()

        # 预处理
        tokens = preprocessor.tokenize(text)
        sentences = preprocessor.split_sentences(text)

        # 统计特征
        ttr = preprocessor.calculate_ttr(tokens)
        sent_mean, sent_var = preprocessor.get_sentence_length_stats(sentences)
        repeat_rate = preprocessor.count_ngram_repetition(tokens, 3) / max(len(tokens), 1)

        # 语言学特征
        syntactic_complexity = self.calculate_syntactic_complexity(text)
        hwv = self.calculate_hwv(text, sentences)

        # 组装特征字典
        features = {
            # 统计特征
            'ppl': ppl if (ppl is not None and ppl == ppl and ppl > 0) else 100.0,  # NaN检查
            'ttr': ttr if ttr == ttr else 0.0,  # NaN检查
            'sentence_length_mean': sent_mean if sent_mean == sent_mean else 0.0,
            'sentence_length_var': sent_var if sent_var == sent_var else 0.0,
            'repetition_rate': repeat_rate if repeat_rate == repeat_rate else 0.0,

            # 语言学特征
            'syntactic_complexity': syntactic_complexity if syntactic_complexity == syntactic_complexity else 0.0,
            'hwv': hwv if hwv == hwv else 0.0,

            # 补充特征
            'avg_word_length': self.calculate_avg_word_length(tokens),
            'punctuation_density': self.calculate_punctuation_density(text, sentences),
            'pos_diversity': self.calculate_pos_diversity(text),
            'function_word_ratio': self.calculate_function_word_ratio(text),
        }

        # 二次检查所有特征值，确保没有NaN或无穷大
        for key, value in features.items():
            if value != value or value > 1e10:  # NaN或过大值
                features[key] = 0.0
                print(f"警告: 特征 {key} 值异常: {value}，已设为0")

        return features

    def _get_zero_features(self) -> Dict[str, float]:
        """返回全零特征"""
        return {
            'ppl': 100.0,
            'ttr': 0.0,
            'sentence_length_mean': 0.0,
            'sentence_length_var': 0.0,
            'repetition_rate': 0.0,
            'syntactic_complexity': 0.0,
            'hwv': 0.0,
            'avg_word_length': 0.0,
            'punctuation_density': 0.0,
            'pos_diversity': 0.0,
            'function_word_ratio': 0.0,
        }

    def calculate_syntactic_complexity(self, text: str) -> float:
        """
        计算句法复杂度
        基于句长、从句数量、词性多样性等

        Args:
            text: 输入文本

        Returns:
            句法复杂度分数
        """
        if not text:
            return 0.0

        # 分句
        sentences = re.split(r'[。！？；\n]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        complexity_scores = []

        for sent in sentences:
            score = 0.0

            # 1. 句长因子
            score += min(len(sent) / 100, 1.0) * 0.3

            # 2. 逗号密度（从句指标）
            comma_count = sent.count('，') + sent.count(',')
            score += min(comma_count / 5, 1.0) * 0.3

            # 3. 词性多样性
            words = pseg.lcut(sent)
            pos_set = set([w.flag for w in words])
            score += min(len(pos_set) / 10, 1.0) * 0.2

            # 4. 特殊句式标记
            if any(mark in sent for mark in ['虽然', '尽管', '但是', '然而', '因此', '所以']):
                score += 0.1

            if any(mark in sent for mark in ['如果', '假如', '要是', '只有', '只要']):
                score += 0.1

            complexity_scores.append(min(score, 1.0))

        return float(np.mean(complexity_scores))

    def calculate_hwv(self, text: str, sentences: List[str] = None) -> float:
        """
        计算Human Writing Variance (HWV)
        人类写作方差指标 - AI生成的文本通常方差较小（更统一）

        Args:
            text: 输入文本
            sentences: 预先分好的句子列表

        Returns:
            HWV值 (0-1之间，值越大越像人类)
        """
        if not text:
            return 0.0

        if sentences is None:
            sentences = re.split(r'[。！？；\n]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 0.0

        # 1. 句长方差
        lengths = [len(s) for s in sentences]
        length_variance = np.var(lengths) if len(lengths) > 1 else 0

        # 2. 标点使用方差
        punct_ratios = []
        for sent in sentences:
            if len(sent) > 0:
                ratio = sum(1 for c in sent if c in '，。！？；：""''、') / len(sent)
                punct_ratios.append(ratio)

        punct_variance = np.var(punct_ratios) if len(punct_ratios) > 1 else 0

        # 3. 词性分布方差
        pos_distributions = []
        for sent in sentences:
            words = pseg.lcut(sent)
            pos_counts = Counter([w.flag for w in words])
            if words:
                pos_dist = {pos: count / len(words) for pos, count in pos_counts.items()}
                pos_distributions.append(pos_dist)

        # 计算分布之间的方差（简化版本）
        if pos_distributions:
            # 使用常见词性的方差
            common_pos = ['n', 'v', 'a', 'd']
            pos_variance = 0
            for pos in common_pos:
                ratios = [dist.get(pos, 0) for dist in pos_distributions]
                pos_variance += np.var(ratios) if len(ratios) > 1 else 0
            pos_variance /= len(common_pos)
        else:
            pos_variance = 0

        # 综合计算HWV
        hwv = (
            min(length_variance / 500, 1.0) * 0.4 +
            min(punct_variance * 100, 1.0) * 0.3 +
            min(pos_variance * 10, 1.0) * 0.3
        )

        return float(hwv)

    def calculate_avg_word_length(self, tokens: List[str]) -> float:
        """
        计算平均词长

        Args:
            tokens: 词语列表

        Returns:
            平均词长
        """
        if not tokens:
            return 0.0

        total_length = sum(len(token) for token in tokens)
        return total_length / len(tokens)

    def calculate_punctuation_density(self, text: str, sentences: List[str]) -> float:
        """
        计算标点符号密度

        Args:
            text: 输入文本
            sentences: 句子列表

        Returns:
            标点密度
        """
        if not text or not sentences:
            return 0.0

        punct_count = sum(1 for c in text if c in '，。！？；：""''、')
        char_count = len(text)

        if char_count == 0:
            return 0.0

        return punct_count / char_count

    def calculate_pos_diversity(self, text: str) -> float:
        """
        计算词性多样性

        Args:
            text: 输入文本

        Returns:
            词性多样性分数
        """
        if not text:
            return 0.0

        words = pseg.lcut(text)
        if not words:
            return 0.0

        pos_types = len(set([w.flag for w in words]))
        total_words = len(words)

        return pos_types / max(total_words, 1)

    def calculate_function_word_ratio(self, text: str) -> float:
        """
        计算功能词比例
        功能词：介词、连词、助词、代词等

        Args:
            text: 输入文本

        Returns:
            功能词比例
        """
        if not text:
            return 0.0

        words = pseg.lcut(text)
        if not words:
            return 0.0

        # 功能词词性
        function_pos = {'p', 'c', 'u', 'r', 'd'}  # 介、连、助、代、副

        function_count = sum(1 for w in words if w.flag in function_pos)

        return function_count / len(words)

    def features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        将特征字典转换为向量

        Args:
            features: 特征字典

        Returns:
            特征向量（numpy数组）
        """
        # 按固定顺序排列特征
        feature_order = [
            'ppl',
            'ttr',
            'sentence_length_mean',
            'sentence_length_var',
            'repetition_rate',
            'syntactic_complexity',
            'hwv',
            'avg_word_length',
            'punctuation_density',
            'pos_diversity',
            'function_word_ratio'
        ]

        vector = np.array([features.get(f, 0.0) for f in feature_order], dtype=np.float32)

        return vector

    def normalize_features(self, features: Dict[str, float],
                          stats: Dict[str, Dict[str, float]] = None) -> Dict[str, float]:
        """
        特征归一化

        Args:
            features: 原始特征
            stats: 统计信息 {feature_name: {'mean': x, 'std': y}}

        Returns:
            归一化后的特征
        """
        if stats is None:
            # 默认归一化参数
            stats = {
                'ppl': {'mean': 100, 'std': 50},
                'ttr': {'mean': 0.6, 'std': 0.15},
                'sentence_length_mean': {'mean': 30, 'std': 15},
                'sentence_length_var': {'mean': 200, 'std': 100},
                'repetition_rate': {'mean': 0.1, 'std': 0.05},
                'syntactic_complexity': {'mean': 0.5, 'std': 0.2},
                'hwv': {'mean': 0.3, 'std': 0.15},
                'avg_word_length': {'mean': 2.0, 'std': 0.5},
                'punctuation_density': {'mean': 0.1, 'std': 0.03},
                'pos_diversity': {'mean': 0.3, 'std': 0.1},
                'function_word_ratio': {'mean': 0.3, 'std': 0.1}
            }

        normalized = {}
        for key, value in features.items():
            if key in stats:
                s = stats[key]
                normalized[key] = (value - s['mean']) / (s['std'] + 1e-8)
            else:
                normalized[key] = value

        return normalized


if __name__ == '__main__':
    # 测试代码
    extractor = FeatureExtractor()

    test_text = """
    人工智能是计算机科学的重要分支。它致力于研究如何让计算机模拟人类智能。
    深度学习是一种重要的机器学习方法。神经网络是深度学习的基础。
    """

    features = extractor.extract_all_features(test_text)

    print("提取的特征:")
    for key, value in features.items():
        print(f"{key}: {value:.4f}")

    vector = extractor.features_to_vector(features)
    print(f"\n特征向量形状: {vector.shape}")
    print(f"特征向量: {vector}")
