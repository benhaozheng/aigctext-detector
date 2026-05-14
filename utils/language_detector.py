# -*- coding: utf-8 -*-
"""
语言检测模块
功能：检测文本是中文还是英文
"""

import re
from typing import Tuple


class LanguageDetector:
    """语言检测器"""

    # 中文字符范围
    CHINESE_PATTERN = re.compile(
        r'[一-鿿㐀-䶿\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]'
    )

    # 英文字符范围（包含基本拉丁字母和常见标点）
    ENGLISH_PATTERN = re.compile(r'[a-zA-Z]')

    @staticmethod
    def detect(text: str) -> str:
        """
        检测文本的主要语言

        Args:
            text: 输入文本

        Returns:
            'zh' (中文), 'en' (英文), 或 'mixed' (混合)
        """
        if not text or len(text.strip()) == 0:
            return 'zh'  # 默认中文

        # 统计中文字符数
        chinese_chars = LanguageDetector.CHINESE_PATTERN.findall(text)
        chinese_count = len(chinese_chars)

        # 统计英文字符数
        english_chars = LanguageDetector.ENGLISH_PATTERN.findall(text)
        english_count = len(english_chars)

        # 计算比例
        total_chars = chinese_count + english_count

        if total_chars == 0:
            return 'zh'

        chinese_ratio = chinese_count / total_chars
        english_ratio = english_count / total_chars

        # 判断主要语言
        if chinese_ratio > 0.3:
            return 'zh'
        elif english_ratio > 0.7:
            return 'en'
        else:
            return 'mixed'

    @staticmethod
    def detect_with_confidence(text: str) -> Tuple[str, float]:
        """
        检测语言并返回置信度

        Args:
            text: 输入文本

        Returns:
            (语言代码, 置信度)
        """
        if not text or len(text.strip()) == 0:
            return 'zh', 0.0

        chinese_chars = LanguageDetector.CHINESE_PATTERN.findall(text)
        english_chars = LanguageDetector.ENGLISH_PATTERN.findall(text)

        chinese_count = len(chinese_chars)
        english_count = len(english_chars)

        total = chinese_count + english_count

        if total == 0:
            return 'zh', 0.0

        chinese_ratio = chinese_count / total
        english_ratio = english_count / total

        if chinese_ratio > 0.3:
            confidence = chinese_ratio
            return 'zh', confidence
        elif english_ratio > 0.7:
            confidence = english_ratio
            return 'en', confidence
        else:
            # 混合语言，返回比例更高的
            if chinese_ratio >= english_ratio:
                return 'zh', chinese_ratio
            else:
                return 'en', english_ratio

    @staticmethod
    def get_model_name(language: str) -> str:
        """
        根据语言返回对应的BERT模型名称

        Args:
            language: 语言代码 ('zh' 或 'en')

        Returns:
            BERT模型名称
        """
        models = {
            'zh': 'bert-base-chinese',
            'en': 'bert-base-uncased'
        }
        return models.get(language, 'bert-base-chinese')


if __name__ == '__main__':
    # 测试代码
    detector = LanguageDetector()

    test_cases = [
        "这是一个测试文本，用于检测中文。",
        "This is a test text for English detection.",
        "这是一个mixed文本with some English words.",
        "",
        "人工智能是计算机科学的重要分支。",
        "Artificial intelligence is a branch of computer science."
    ]

    print("语言检测测试:")
    print("=" * 60)
    for text in test_cases:
        lang, conf = detector.detect_with_confidence(text)
        model = detector.get_model_name(lang)
        print(f"文本: {text[:40]}")
        print(f"  语言: {lang}, 置信度: {conf:.2f}, 模型: {model}")
        print()
