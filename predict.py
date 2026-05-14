# -*- coding: utf-8 -*-
"""
推理脚本
功能：使用训练好的模型进行AI检测预测
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
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
from utils.advice import AdviceGenerator
from utils.language_detector import LanguageDetector


class AIDetector:
    """AI内容检测器（支持中英文自动检测）"""

    def __init__(self, use_fusion: bool = True, device: str = None, language: str = 'auto'):
        """
        初始化检测器

        Args:
            use_fusion: 是否使用融合模型
            device: 计算设备
            language: 语言类型 ('zh'=中文, 'en'=英文, 'auto'=自动检测)
        """
        self.use_fusion = use_fusion
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.language = language

        print(f"初始化AI检测器...")
        print(f"使用设备: {self.device}")
        print(f"模型类型: {'融合模型' if use_fusion else 'BERT模型'}")
        print(f"语言模式: {language}")

        # 语言检测器
        self.lang_detector = LanguageDetector()

        # 模型缓存
        self.models = {}

        # 加载默认模型（中文）
        self._load_model('zh')

        # 初始化工具
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.advice_generator = AdviceGenerator()

    def _load_model(self, language: str):
        """加载指定语言的模型"""
        if language in self.models:
            return

        model_name = LanguageDetector.get_model_name(language)

        print(f"正在加载{language}模型: {model_name}")

        # 加载BERT
        bert = BERTClassifier(
            model_name=model_name,
            num_labels=2,
            device=self.device
        )

        # 尝试加载检查点
        checkpoint = f"./checkpoints/bert_{language}"
        if os.path.exists(checkpoint):
            try:
                bert.load_from_checkpoint(checkpoint)
                print(f"已加载训练好的{language}模型")
            except:
                pass

        # 创建融合模型（使用与训练时相同的hidden_dims）
        fusion = FusionModel(
            bert_model=bert,
            feature_dim=11,
            device=self.device,
            hidden_dims=[256, 128, 64]
        )

        fusion_checkpoint = f"./checkpoints/fusion_{language}_best.pt"
        if os.path.exists(fusion_checkpoint):
            try:
                fusion.load_model(fusion_checkpoint)
                print(f"已加载训练好的{language}融合模型")
            except:
                pass

        self.models[language] = {
            'bert': bert,
            'fusion': fusion
        }

    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        if self.language == 'auto':
            lang, _ = self.lang_detector.detect_with_confidence(text)
            return lang
        return self.language

    def detect(self, text: str, return_details: bool = True, language: str = None) -> Dict:
        """
        检测单个文本（支持中英文自动检测）

        Args:
            text: 输入文本
            return_details: 是否返回详细信息
            language: 语言（None表示自动检测）

        Returns:
            检测结果
        """
        if not text or len(text.strip()) == 0:
            return {
                'error': '输入文本为空',
                'ai_probability': 0.0,
                'originality_score': 0.0,
                'label': 'Unknown'
            }

        # 检测语言
        if language is None:
            language = self._detect_language(text)

        # 加载对应语言的模型
        if language not in self.models:
            self._load_model(language)

        # 获取模型
        bert = self.models[language]['bert']
        fusion = self.models[language]['fusion']

        # 设置预处理器的语言
        self.preprocessor.set_language(language)

        # 预处理
        cleaned = self.preprocessor.clean_text(text)
        paragraphs = self.preprocessor.split_paragraphs(cleaned)

        # 提取特征
        features = self.feature_extractor.extract_all_features(cleaned)

        # 预测
        if self.use_fusion:
            result = fusion.predict(cleaned, features)
        else:
            result = bert.predict(cleaned)

        ai_prob = result['ai_probability']
        originality_score = (1 - ai_prob) * 100

        # 基础结果
        detection_result = {
            'text_length': len(text),
            'ai_probability': ai_prob,
            'originality_score': originality_score,
            'label': result['label'],
            'language': language,
        }

        if return_details:
            # 段落级检测
            paragraph_results = self._detect_paragraphs(paragraphs, language, bert, fusion)

        ai_prob = result['ai_probability']
        originality_score = (1 - ai_prob) * 100

        # 基础结果
        detection_result = {
            'text_length': len(text),
            'ai_probability': ai_prob,
            'originality_score': originality_score,
            'label': result['label'],
        }

        if return_details:
            # 段落级检测
            paragraph_results = self._detect_paragraphs(paragraphs)

            # 特征详情
            feature_details = {
                'perplexity': features.get('ppl', 0),
                'ttr': features.get('ttr', 0),
                'sentence_length_mean': features.get('sentence_length_mean', 0),
                'sentence_length_var': features.get('sentence_length_var', 0),
                'hwv': features.get('hwv', 0),
                'syntactic_complexity': features.get('syntactic_complexity', 0),
            }

            # 修改建议
            advices = self.advice_generator.generate_advice(
                cleaned, ai_prob, features
            )

            # 段落风险分析
            risk_analysis = self.advice_generator.analyze_paragraph_risk(
                paragraphs, [p['ai_probability'] for p in paragraph_results]
            )

            detection_result.update({
                'paragraphs': paragraph_results,
                'features': feature_details,
                'advices': advices,
                'risk_analysis': risk_analysis,
                'paragraph_count': len(paragraphs)
            })

        return detection_result

    def _detect_paragraphs(self, paragraphs: List[str], language: str, bert, fusion) -> List[Dict]:
        """检测每个段落"""
        results = []

        for para in paragraphs:
            if self.use_fusion:
                features = self.feature_extractor.extract_all_features(para)
                result = fusion.predict(para, features)
            else:
                result = bert.predict(para)

            results.append({
                'text': para,
                'ai_probability': result['ai_probability'],
                'label': result['label']
            })

        return results

    def detect_batch(self, texts: List[str]) -> List[Dict]:
        """批量检测"""
        results = []

        for i, text in enumerate(texts):
            print(f"正在处理第 {i+1}/{len(texts)} 个样本...")
            result = self.detect(text, return_details=False)
            results.append(result)

        return results

    def analyze_file(self, file_path: str) -> Dict:
        """
        分析文件

        Args:
            file_path: 文件路径

        Returns:
            分析结果
        """
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"已读取文件: {file_path}")
        print(f"文本长度: {len(text)} 字符")

        # 检测
        result = self.detect(text)

        return result

    def visualize_result(self, result: Dict, save_path: str = None):
        """
        可视化检测结果

        Args:
            result: 检测结果
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. AI概率仪表盘
        ax1 = axes[0, 0]
        ai_prob = result['ai_probability']
        colors = ['#2ecc71' if ai_prob < 0.3 else '#f39c12' if ai_prob < 0.7 else '#e74c3c']
        ax1.barh(['AI生成概率'], [ai_prob], color=colors, height=0.5)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('概率')
        ax1.set_title(f'AI生成概率: {ai_prob:.2%}', fontsize=14, fontweight='bold')
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

        # 2. 原创性评分
        ax2 = axes[0, 1]
        originality = result['originality_score']
        score_colors = ['#e74c3c' if originality < 30 else '#f39c12' if originality < 70 else '#2ecc71']
        ax2.barh(['原创性评分'], [originality], color=score_colors, height=0.5)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('分数')
        ax2.set_title(f'原创性评分: {originality:.1f}/100', fontsize=14, fontweight='bold')
        ax2.axvline(x=60, color='gray', linestyle='--', alpha=0.5, label='及格线')
        ax2.legend()

        # 3. 特征雷达图
        ax3 = axes[1, 0]
        if 'features' in result:
            features = result['features']
            feature_names = ['困惑度\n(反向)', 'TTR', '句长均值\n(归一化)', 'HWV', '句法复杂度']

            # 归一化特征值到0-1
            feature_values = [
                max(0, 1 - features.get('perplexity', 50) / 200),
                features.get('ttr', 0.5),
                min(1, features.get('sentence_length_mean', 30) / 50),
                features.get('hwv', 0.3),
                features.get('syntactic_complexity', 0.5)
            ]

            # 雷达图
            angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
            feature_values += feature_values[:1]
            angles += angles[:1]

            ax3 = plt.subplot(2, 2, 3, projection='polar')
            ax3.plot(angles, feature_values, 'o-', linewidth=2, color='#3498db')
            ax3.fill(angles, feature_values, alpha=0.25, color='#3498db')
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(feature_names)
            ax3.set_ylim(0, 1)
            ax3.set_title('文本特征分析', fontsize=12, fontweight='bold', pad=20)
            axes[1, 0] = ax3

        # 4. 段落风险分布
        ax4 = axes[1, 1]
        if 'paragraphs' in result and result['paragraphs']:
            paragraphs = result['paragraphs']
            ai_probs = [p['ai_probability'] for p in paragraphs]
            x_pos = range(1, len(paragraphs) + 1)

            colors_list = ['#2ecc71' if p < 0.3 else '#f39c12' if p < 0.7 else '#e74c3c' for p in ai_probs]
            ax4.bar(x_pos, ai_probs, color=colors_list)
            ax4.set_xlabel('段落编号')
            ax4.set_ylabel('AI概率')
            ax4.set_title('各段落AI生成概率', fontsize=12, fontweight='bold')
            ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax4.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化结果已保存: {save_path}")

        plt.show()

    def print_report(self, result: Dict):
        """打印检测报告"""
        print("\n" + "="*60)
        print("AI作业原创性检测报告".center(60))
        print("="*60)

        print(f"\n【检测结果】")
        print(f"  检测标签: {result['label']}")
        print(f"  AI生成概率: {result['ai_probability']:.2%}")
        print(f"  原创性评分: {result['originality_score']:.1f}/100")

        if 'paragraph_count' in result:
            print(f"\n【文本信息】")
            print(f"  文本长度: {result['text_length']} 字符")
            print(f"  段落数量: {result['paragraph_count']}")

        if 'features' in result:
            print(f"\n【特征分析】")
            for key, value in result['features'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")

        if 'risk_analysis' in result and result['risk_analysis']:
            print(f"\n【段落风险分析】")
            for item in result['risk_analysis']:
                print(f"  段落{item['paragraph_index']}: {item['risk_text']} (AI概率: {item['ai_probability']:.2%})")

        if 'advices' in result and result['advices']:
            print(f"\n【修改建议】")
            for i, advice in enumerate(result['advices'], 1):
                print(f"  {i}. [{advice['type']}] {advice['advice']}")

        print("\n" + "="*60)

        # 风险评估
        ai_prob = result['ai_probability']
        if ai_prob < 0.3:
            risk_level = "低风险"
            risk_desc = "文本具有较高的人类写作特征"
        elif ai_prob < 0.7:
            risk_level = "中风险"
            risk_desc = "文本存在部分AI生成特征"
        else:
            risk_level = "高风险"
            risk_desc = "文本很可能由AI生成"

        print(f"\n【综合评估】: {risk_level}")
        print(f"  {risk_desc}")
        print("="*60 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI内容检测')
    parser.add_argument('--text', type=str, help='待检测的文本')
    parser.add_argument('--file', type=str, help='待检测的文件路径')
    parser.add_argument('--model', type=str, default='fusion', choices=['bert', 'fusion'],
                       help='使用的模型')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--output', type=str, help='可视化结果保存路径')

    args = parser.parse_args()

    # 创建检测器
    detector = AIDetector(use_fusion=(args.model == 'fusion'))

    # 获取输入
    if args.file:
        result = detector.analyze_file(args.file)
    elif args.text:
        result = detector.detect(args.text)
    else:
        # 交互模式
        print("=== AI作业原创性检测系统 ===")
        print("请输入待检测的文本（输入 'quit' 退出）:\n")

        while True:
            text = input("\n请输入文本: ")
            if text.lower() in ['quit', 'exit', 'q']:
                print("退出系统...")
                break

            if text.strip():
                result = detector.detect(text)
                detector.print_report(result)
            else:
                print("输入为空，请重新输入")

        return

    # 打印报告
    detector.print_report(result)

    # 可视化
    if args.visualize:
        save_path = args.output or './checkpoints/detection_result.png'
        detector.visualize_result(result, save_path)


if __name__ == '__main__':
    main()
