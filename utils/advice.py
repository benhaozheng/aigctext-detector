# -*- coding: utf-8 -*-
"""
修改建议生成模块
功能：基于规则生成文本修改建议，降低AI生成特征
"""

import re
from typing import List, Dict, Tuple
import random


class AdviceGenerator:
    """修改建议生成器"""

    def __init__(self):
        """初始化建议生成器"""
        # AI常用表达模式
        self.ai_patterns = {
            '连接词': [
                '首先', '其次', '再次', '最后', '总之', '综上所述',
                '一方面', '另一方面', '此外', '而且', '另外', '同时'
            ],
            '总结词': [
                '总的来说', '总体而言', '从根本上说', '本质上',
                '值得注意的是', '需要强调的是'
            ],
            '过度结构化': [
                '第一', '第二', '第三', '第一点', '第二点',
                '首先...其次...最后', '一是...二是...三是'
            ],
            '说教语气': [
                '我们应该', '我们必须', '大家要', '需要我们',
                '让我们一起', '值得注意的是'
            ],
            '绝对化表达': [
                '完全', '彻底', '绝对', '必定', '必然',
                '毫无疑问', '毫无疑问地', '不可否认'
            ]
        }

        # 修改建议模板
        self.advice_templates = {
            'ppl_high': [
                "增加个人经历和具体事例，让内容更有说服力",
                "使用更多口语化表达，增加文章的自然感",
                "添加一些反问句和感叹句，增强情感表达",
                "适当使用方言或个人习惯用语"
            ],
            'ppl_low': [
                "简化句式结构，避免过于工整的排比",
                "减少连接词的使用，让句子更直接",
                "增加一些不完美表达，如倒装句或省略句",
                "适当增加句长变化，不要所有句子都差不多长"
            ],
            'ttr_low': [
                "使用同义词替换重复出现的词汇",
                "增加描述性词汇，减少名词和动词的重复使用",
                "使用更多具体的名称和专有名词"
            ],
            'structure_uniform': [
                "打乱段落顺序或调整句子位置",
                "改变句子开头方式，避免都用同一类词开头",
                "增加一些插入语或倒装句",
                "使用长短句交替，创造节奏变化"
            ],
            'lack_personal': [
                "添加个人观点和感受",
                "使用第一人称叙述，增加主观色彩",
                "分享相关经历或见闻",
                "表达情感色彩和态度倾向"
            ]
        }

    def generate_advice(self, text: str, ai_probability: float,
                       features: Dict = None) -> List[Dict[str, str]]:
        """
        根据检测结果生成修改建议

        Args:
            text: 输入文本
            ai_probability: AI生成概率
            features: 特征字典

        Returns:
            建议列表 [{'type': 类型, 'advice': 建议}]
        """
        advice_list = []

        # 1. 根据AI概率给出总体建议
        if ai_probability > 0.7:
            advice_list.append({
                'type': '总体建议',
                'advice': '本文具有较高AI生成特征，建议进行较大幅度修改'
            })
        elif ai_probability > 0.4:
            advice_list.append({
                'type': '总体建议',
                'advice': '本文存在一定AI生成特征，建议进行部分修改'
            })
        else:
            advice_list.append({
                'type': '总体建议',
                'advice': '本文AI生成特征较低，保持现有风格即可'
            })

        # 2. 根据特征给出具体建议
        if features:
            advice_list.extend(self._generate_feature_advice(features))

        # 3. 检测AI模式并给出建议
        pattern_advice = self._detect_ai_patterns(text)
        advice_list.extend(pattern_advice)

        # 4. 给出通用建议
        advice_list.extend(self._get_general_advice())

        return advice_list[:8]  # 限制返回数量

    def _generate_feature_advice(self, features: Dict) -> List[Dict[str, str]]:
        """根据特征生成建议"""
        advice_list = []

        # TTR低 - 词汇多样性不足
        if features.get('ttr', 0.7) < 0.5:
            advice = random.choice(self.advice_templates['ttr_low'])
            advice_list.append({'type': '词汇多样性', 'advice': advice})

        # 句长方差小 - 结构单一
        if features.get('sentence_length_var', 100) < 50:
            advice = random.choice(self.advice_templates['structure_uniform'])
            advice_list.append({'type': '句式结构', 'advice': advice})

        # HWV低 - 缺乏个人特色
        if features.get('hwv', 0.5) < 0.3:
            advice = random.choice(self.advice_templates['lack_personal'])
            advice_list.append({'type': '个人特色', 'advice': advice})

        return advice_list

    def _detect_ai_patterns(self, text: str) -> List[Dict[str, str]]:
        """检测文本中的AI常用模式"""
        advice_list = []

        # 检测过度使用的连接词
        connection_count = sum(1 for word in self.ai_patterns['连接词'] if word in text)
        if connection_count > 3:
            advice_list.append({
                'type': '连接词使用',
                'advice': f'检测到使用了{connection_count}个常见连接词，建议减少或替换'
            })

        # 检测绝对化表达
        absolute_count = sum(1 for word in self.ai_patterns['绝对化表达'] if word in text)
        if absolute_count > 0:
            advice_list.append({
                'type': '表达方式',
                'advice': '检测到绝对化表达，建议改为更委婉的说法'
            })

        # 检测说教语气
        preaching_count = sum(1 for word in self.ai_patterns['说教语气'] if word in text)
        if preaching_count > 1:
            advice_list.append({
                'type': '语气风格',
                'advice': '检测到说教语气，建议改为更亲切的口吻'
            })

        # 检测过度结构化
        if re.search(r'首先.*其次.*最后', text):
            advice_list.append({
                'type': '文章结构',
                'advice': '检测到"首先-其次-最后"的固定结构，建议打乱顺序'
            })

        return advice_list

    def _get_general_advice(self) -> List[Dict[str, str]]:
        """获取通用建议"""
        general_advice = [
            {
                'type': '增加细节',
                'advice': '添加具体的场景描述和细节，使内容更生动'
            },
            {
                'type': '情感表达',
                'advice': '增加情感色彩的表达，如喜欢、讨厌、惊讶等'
            },
            {
                'type': '句式变化',
                'advice': '使用疑问句、感叹句等多种句式'
            },
            {
                'type': '避免套话',
                'advice': '删除"众所周知"、"毫无疑问"等套话'
            }
        ]

        return random.sample(general_advice, 2)

    def get_specific_suggestions(self, text: str) -> List[Dict[str, str]]:
        """
        获取具体的修改位置建议

        Args:
            text: 输入文本

        Returns:
            包含位置和建议的列表
        """
        suggestions = []

        # 分句检测
        sentences = re.split(r'[。！？；\n]+', text)

        for i, sent in enumerate(sentences):
            if not sent.strip():
                continue

            # 检测每句的特征
            sent = sent.strip()

            # 以连接词开头的句子
            for conn in self.ai_patterns['连接词']:
                if sent.startswith(conn):
                    suggestions.append({
                        'position': f'第{i+1}句',
                        'original': sent[:50] + '...' if len(sent) > 50 else sent,
                        'suggestion': f'避免以"{conn}"开头，尝试直接表达内容'
                    })
                    break

            # 检测总结词
            for summary in self.ai_patterns['总结词']:
                if summary in sent:
                    suggestions.append({
                        'position': f'第{i+1}句',
                        'original': sent[:50] + '...' if len(sent) > 50 else sent,
                        'suggestion': f'将"{summary}"改为更自然的过渡'
                    })
                    break

        return suggestions[:5]  # 返回前5条

    def paraphrase_suggestion(self, sentence: str) -> str:
        """
        为单句提供改写建议（基于规则）

        Args:
            sentence: 输入句子

        Returns:
            改写建议
        """
        if not sentence:
            return ""

        suggestions = []

        # 规则1：去掉开头的连接词
        for conn in ['首先', '其次', '再次', '最后', '总之', '此外', '另外']:
            if sentence.startswith(conn):
                suggestions.append(f"去掉开头的「{conn}」")
                break

        # 规则2：替换绝对化表达
        replacements = {
            '完全': '很大程度上',
            '绝对': '基本',
            '必定': '很可能',
            '毫无疑问': '可以说',
            '不可否认': '值得注意的是'
        }
        for old, new in replacements.items():
            if old in sentence:
                suggestions.append(f"将「{old}」改为「{new}」")
                break

        # 规则3：增加口语化表达
        if len(sentence) > 50 and '，' not in sentence:
            suggestions.append("适当增加停顿，将长句拆分")

        # 规则4：增加情感词
        emotion_words = ['喜欢', '爱', '讨厌', '惊讶', '好奇', '担心']
        if not any(word in sentence for word in emotion_words):
            suggestions.append("增加个人感受或情感词")

        if suggestions:
            return "；".join(suggestions)
        else:
            return "保持原句，增加个人细节即可"

    def analyze_paragraph_risk(self, paragraphs: List[str],
                               ai_probs: List[float]) -> List[Dict]:
        """
        分析段落风险并给出针对性建议

        Args:
            paragraphs: 段落列表
            ai_probs: 每段的AI概率

        Returns:
            风险分析列表
        """
        risk_analysis = []

        for i, (para, prob) in enumerate(zip(paragraphs, ai_probs)):
            # 风险等级
            if prob > 0.7:
                risk_level = 'High'
                risk_text = '高风险'
            elif prob > 0.4:
                risk_level = 'Medium'
                risk_text = '中风险'
            else:
                risk_level = 'Low'
                risk_text = '低风险'

            # 生成建议
            if risk_level == 'High':
                suggestion = '建议大幅重写该段落，增加个人观点和具体事例'
            elif risk_level == 'Medium':
                suggestion = '建议调整句式结构，替换部分词汇'
            else:
                suggestion = '该段落较为自然，可保持原样'

            risk_analysis.append({
                'paragraph_index': i + 1,
                'preview': para[:100] + '...' if len(para) > 100 else para,
                'ai_probability': prob,
                'risk_level': risk_level,
                'risk_text': risk_text,
                'suggestion': suggestion
            })

        return risk_analysis


if __name__ == '__main__':
    # 测试代码
    generator = AdviceGenerator()

    test_text = """
    首先，人工智能是计算机科学的重要分支。
    其次，深度学习是人工智能的核心技术。
    总的来说，人工智能将会改变我们的生活方式。
    """

    print("=== 修改建议测试 ===")
    advice = generator.generate_advice(test_text, ai_probability=0.75)

    for item in advice:
        print(f"[{item['type']}] {item['advice']}")

    print("\n=== 具体改写建议 ===")
    suggestions = generator.get_specific_suggestions(test_text)
    for s in suggestions:
        print(f"{s['position']}: {s['suggestion']}")
        print(f"  原文: {s['original']}")
