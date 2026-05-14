# -*- coding: utf-8 -*-
"""
困惑度计算模块
功能：使用GPT2模型计算文本的困惑度（Perplexity）
困惑度越低，文本越像AI生成；困惑度越高，文本越像人类写作
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Union
import warnings
warnings.filterwarnings('ignore')


class PerplexityCalculator:
    """困惑度计算器"""

    def __init__(self, model_name: str = "IDEA-CCNL/Wenzhong-GPT2-110M", device: str = None):
        """
        初始化困惑度计算器

        Args:
            model_name: GPT2模型名称，默认使用中文GPT2模型
            device: 计算设备 (cuda/cpu)，None表示自动选择
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"正在加载GPT2模型: {model_name}")
        print(f"使用设备: {self.device}")

        try:
            # 加载模型和tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.to(self.device)
            self.model.eval()

            print("GPT2模型加载成功！")

        except Exception as e:
            print(f"加载模型失败: {e}")
            print("尝试使用备用模型...")
            # 使用备用模型
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.to(self.device)
            self.model.eval()

    def calculate_perplexity(self, text: str, max_length: int = 512, stride: int = 128) -> float:
        """
        计算单个文本的困惑度

        Args:
            text: 输入文本
            max_length: 最大序列长度
            stride: 滑动窗口步长

        Returns:
            困惑度值
        """
        if not text or len(text.strip()) == 0:
            return float('inf')

        try:
            # Tokenize
            encodings = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length
            )

            input_ids = encodings.input_ids.to(self.device)

            # 如果文本太短，直接计算
            if input_ids.size(1) <= stride:
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    neg_log_likelihood = outputs.loss.item()

                ppl = np.exp(neg_log_likelihood)
                return float(ppl)

            # 长文本使用滑动窗口计算
            nlls = []
            for i in range(0, input_ids.size(1), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, input_ids.size(1))

                target_len = end_loc - i

                input_batch = input_ids[:, begin_loc:end_loc]

                with torch.no_grad():
                    outputs = self.model(input_batch, labels=input_batch)
                    neg_log_likelihood = outputs.loss * target_len

                nlls.append(neg_log_likelihood)

            # 平均困惑度
            ppl = np.exp(torch.stack(nlls).sum() / end_loc)

            return float(ppl.item())

        except Exception as e:
            print(f"计算困惑度时出错: {e}")
            return float('inf')

    def calculate_batch_perplexity(self, texts: List[str]) -> List[float]:
        """
        批量计算困惑度

        Args:
            texts: 文本列表

        Returns:
            困惑度列表
        """
        ppls = []
        for i, text in enumerate(texts):
            ppl = self.calculate_perplexity(text)
            ppls.append(ppl)

            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/{len(texts)} 个样本")

        return ppls

    def get_ppl_score(self, ppl: float, min_ppl: float = 10.0, max_ppl: float = 500.0) -> float:
        """
        将困惑度转换为0-1分数
        低困惑度（可能是AI）-> 分数接近1
        高困惑度（可能是人类）-> 分数接近0

        Args:
            ppl: 困惑度值
            min_ppl: 最小困惑度阈值
            max_ppl: 最大困惑度阈值

        Returns:
            0-1之间的分数
        """
        if ppl == float('inf'):
            return 0.0

        # Sigmoid归一化
        if ppl <= min_ppl:
            return 1.0
        elif ppl >= max_ppl:
            return 0.0
        else:
            # 线性归一化
            return 1.0 - (ppl - min_ppl) / (max_ppl - min_ppl)

    def calculate_perplexity_with_details(self, text: str, window_size: int = 100) -> dict:
        """
        计算困惑度并返回详细信息

        Args:
            text: 输入文本
            window_size: 窗口大小（用于分段计算）

        Returns:
            包含详细信息的字典
        """
        # 整体困惑度
        overall_ppl = self.calculate_perplexity(text)

        # 分段计算困惑度
        paragraphs = text.split('\n\n')
        paragraph_ppls = []

        for para in paragraphs:
            if para.strip():
                para_ppl = self.calculate_perplexity(para.strip())
                paragraph_ppls.append({
                    'text': para.strip()[:100] + '...' if len(para) > 100 else para.strip(),
                    'ppl': para_ppl
                })

        return {
            'overall_perplexity': overall_ppl,
            'ai_probability': self.get_ppl_score(overall_ppl),
            'paragraph_perplexities': paragraph_ppls,
            'avg_paragraph_ppl': np.mean([p['ppl'] for p in paragraph_ppls]) if paragraph_ppls else overall_ppl,
            'std_paragraph_ppl': np.std([p['ppl'] for p in paragraph_ppls]) if paragraph_ppls else 0.0
        }


if __name__ == '__main__':
    # 测试代码
    print("初始化困惑度计算器...")

    calculator = PerplexityCalculator()

    # 测试文本
    ai_like_text = """
    人工智能是计算机科学的一个重要分支，它致力于研究如何使计算机能够模拟人类的智能行为。
    人工智能的发展可以分为几个阶段：符号主义、连接主义和行为主义。
    深度学习是连接主义的一种实现方式，它通过多层神经网络来学习数据的特征表示。
    """

    human_like_text = """
    哎，今天天气真不错啊！我想起了小时候，那时候夏天总是那么漫长。
    妈妈总是在院子里摆个小桌子，我们就坐在那儿乘凉、吃西瓜。
    那种感觉，现在想起来还是甜甜的。不过话说回来，现在的孩子可能很难体会到了吧。
    """

    print("\n=== AI风格文本 ===")
    result1 = calculator.calculate_perplexity_with_details(ai_like_text)
    print(f"困惑度: {result1['overall_perplexity']:.2f}")
    print(f"AI概率: {result1['ai_probability']:.4f}")

    print("\n=== 人类风格文本 ===")
    result2 = calculator.calculate_perplexity_with_details(human_like_text)
    print(f"困惑度: {result2['overall_perplexity']:.2f}")
    print(f"AI概率: {result2['ai_probability']:.4f}")
