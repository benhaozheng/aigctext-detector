# -*- coding: utf-8 -*-
"""
文本预处理模块
功能：文本清洗、分句、分词、段落切分
支持中英文
"""

import re
import jieba
import string
from typing import List, Tuple, Optional
import numpy as np


class TextPreprocessor:
    """文本预处理器（支持中英文）"""

    def __init__(self, language: str = 'zh'):
        """
        初始化预处理器

        Args:
            language: 语言类型 ('zh'=中文, 'en'=英文, 'auto'=自动检测)
        """
        self.language = language

        # 加载jieba分词词典（仅中文需要）
        if language in ['zh', 'auto']:
            jieba.initialize()

        # 中文标点符号
        self.chinese_punctuation = '，。！？；：""''、—…《》【】（）'

        # 英文标点符号
        self.english_punctuation = '.,!?;:"\'-—'

    def set_language(self, language: str):
        """设置语言类型"""
        self.language = language

    def _detect_language(self, text: str) -> str:
        """简单的语言检测"""
        # 检测中文字符
        chinese_chars = re.findall(r'[一-鿿]', text)
        # 检测英文字符
        english_chars = re.findall(r'[a-zA-Z]', text)

        total = len(chinese_chars) + len(english_chars)
        if total == 0:
            return 'zh'

        chinese_ratio = len(chinese_chars) / total

        return 'zh' if chinese_ratio > 0.3 else 'en'

    def clean_text(self, text: str) -> str:
        """
        清洗文本
        去除特殊字符、多余空格、HTML标签等

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)

        # 去除URL
        text = re.sub(r'http[s]?://\S+', '', text)

        # 去除邮箱
        text = re.sub(r'\S+@\S+', '', text)

        # 统一全角/半角标点
        text = text.replace('，', '，').replace('。', '。')
        text = text.replace('！', '！').replace('？', '？')

        # 去除多余空格（保留单词间单个空格）
        text = re.sub(r' +', ' ', text)

        # 去除行首行尾空格
        lines = [line.strip() for line in text.split('\n')]

        # 过滤空行
        lines = [line for line in lines if line]

        return '\n'.join(lines)

    def split_sentences(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        分句
        将文本分割成句子列表

        Args:
            text: 输入文本
            language: 语言类型（None表示使用self.language）

        Returns:
            句子列表
        """
        if not text:
            return []

        lang = language if language else self.language

        # 自动检测语言
        if lang == 'auto':
            lang = self._detect_language(text)

        # 根据语言选择分句模式
        if lang == 'en':
            # 英文分句：按 . ! ? 分割
            pattern = re.compile(r'[.!?]+\s+')
        else:
            # 中文分句：按 。！？； 分割
            pattern = re.compile(r'[。！？；\n]+')

        sentences = pattern.split(text)

        # 清理每个句子
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def split_paragraphs(self, text: str, sentences_per_paragraph: int = 6) -> List[str]:
        """
        分段
        将文本按段落分割，支持智能分段

        Args:
            text: 输入文本
            sentences_per_paragraph: 每段包含的最大句子数（用于智能分段）

        Returns:
            段落列表
        """
        if not text:
            return []

        # 先按换行符分段
        paragraphs = re.split(r'\n\s*\n', text)

        # 清理每个段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # 智能分段：处理过长的段落
        result = []
        for para in paragraphs:
            # 如果段落太长（超过300字或超过指定句子数），则切分
            if len(para) > 300:
                # 按句子分割
                sentences = self.split_sentences(para)
                # 每N个句子组成一段
                for i in range(0, len(sentences), sentences_per_paragraph):
                    chunk = ''.join(sentences[i:i + sentences_per_paragraph])
                    if chunk.strip():
                        result.append(chunk.strip())
            else:
                result.append(para)

        return result

    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        分词
        中文使用jieba，英文使用空格分词

        Args:
            text: 输入文本
            language: 语言类型（None表示使用self.language）

        Returns:
            词语列表
        """
        if not text:
            return []

        lang = language if language else self.language

        # 自动检测语言
        if lang == 'auto':
            lang = self._detect_language(text)

        if lang == 'en':
            # 英文分词：按空格分割，去除标点
            words = text.split()
            # 转换为小写并去除标点
            words = [word.lower().strip(string.punctuation) for word in words]
            # 过滤空词和纯数字
            words = [w for w in words if w and not w.isdigit()]
        else:
            # 中文分词：使用jieba
            words = jieba.lcut(text)

        # 过滤停用词
        stop_words = self._get_stop_words(lang)
        words = [w for w in words if w not in stop_words and len(w.strip()) > 0]

        return words

    def _get_stop_words(self, language: str = 'zh') -> set:
        """
        获取停用词表

        Args:
            language: 语言类型 ('zh' 或 'en')

        Returns:
            停用词集合
        """
        if language == 'en':
            # 英文停用词
            stop_words = {
                'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because',
                'as', 'what', 'which', 'this', 'that', 'these', 'those',
                'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'must', 'can',
                'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
                'my', 'your', 'his', 'its', 'our', 'their',
                'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of'
            }
            stop_words.update(set(string.punctuation))
        else:
            # 中文停用词
            stop_words = {
                '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
                '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
                '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '有',
                '个', '之', '为', '或是', '或者', '因为', '由于', '所以', '但是',
                '然而', '而且', '并且', '接着', '于是', '无论如何', '不管', '无论',
                '呀', '啊', '哦', '嗯', '哈', '嘿', '哎', '唉', '嘛', '呗', '呢'
            }
            stop_words.update(set(self.chinese_punctuation))
            stop_words.update(set(string.punctuation))

        return stop_words

    def get_sentence_length_stats(self, sentences: List[str]) -> Tuple[float, float]:
        """
        计算句长统计量
        返回句子长度的均值和方差

        Args:
            sentences: 句子列表

        Returns:
            (mean_length, variance_length)
        """
        if not sentences:
            return 0.0, 0.0

        lengths = [len(s) for s in sentences]

        mean_length = float(np.mean(lengths))
        var_length = float(np.var(lengths))

        return mean_length, var_length

    def count_ngram_repetition(self, tokens: List[str], n: int = 3) -> int:
        """
        计算n-gram重复次数
        统计重复出现的n-gram数量

        Args:
            tokens: 词语列表
            n: n-gram的n值

        Returns:
            重复的n-gram数量
        """
        if len(tokens) < n:
            return 0

        # 生成所有n-gram
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)

        # 统计每个n-gram出现次数
        from collections import Counter
        ngram_counts = Counter(ngrams)

        # 计算重复次数（出现次数>1的n-gram数量）
        repeated = sum(1 for count in ngram_counts.values() if count > 1)

        return repeated

    def calculate_ttr(self, tokens: List[str]) -> float:
        """
        计算类型-标记比（Type-Token Ratio, TTR）
        词汇多样性指标

        Args:
            tokens: 词语列表

        Returns:
            TTR值 (0-1之间)
        """
        if not tokens:
            return 0.0

        unique_types = len(set(tokens))
        total_tokens = len(tokens)

        ttr = unique_types / total_tokens if total_tokens > 0 else 0.0

        return ttr

    def preprocess_full(self, text: str) -> dict:
        """
        完整预处理流程
        返回预处理后的所有结果

        Args:
            text: 原始文本

        Returns:
            包含预处理结果的字典
        """
        # 清洗
        cleaned = self.clean_text(text)

        # 分句
        sentences = self.split_sentences(cleaned)

        # 分段
        paragraphs = self.split_paragraphs(cleaned)

        # 分词
        tokens = self.tokenize(cleaned)

        # 句长统计
        sent_mean, sent_var = self.get_sentence_length_stats(sentences)

        # TTR
        ttr = self.calculate_ttr(tokens)

        # n-gram重复率
        repeat_2gram = self.count_ngram_repetition(tokens, 2)
        repeat_3gram = self.count_ngram_repetition(tokens, 3)
        repeat_4gram = self.count_ngram_repetition(tokens, 4)

        return {
            'cleaned_text': cleaned,
            'sentences': sentences,
            'paragraphs': paragraphs,
            'tokens': tokens,
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'token_count': len(tokens),
            'sentence_length_mean': sent_mean,
            'sentence_length_var': sent_var,
            'ttr': ttr,
            'repeat_2gram': repeat_2gram,
            'repeat_3gram': repeat_3gram,
            'repeat_4gram': repeat_4gram
        }


if __name__ == '__main__':
    # 测试代码
    preprocessor = TextPreprocessor()

    test_text = """
    这是一个测试文本。它包含了多个句子！

    这是第二个段落，用于测试分段功能。

    这个句子是为了测试词汇多样性，词汇多样性是一个重要的指标。
    """

    result = preprocessor.preprocess_full(test_text)

    print(f"句子数量: {result['sentence_count']}")
    print(f"段落数量: {result['paragraph_count']}")
    print(f"词语数量: {result['token_count']}")
    print(f"句长均值: {result['sentence_length_mean']:.2f}")
    print(f"句长方差: {result['sentence_length_var']:.2f}")
    print(f"TTR: {result['ttr']:.4f}")
    print(f"2-gram重复数: {result['repeat_2gram']}")
