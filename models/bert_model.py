# -*- coding: utf-8 -*-
"""
BERT分类模型
功能：基于bert-base-chinese的二分类模型
"""

import torch
import torch.nn as nn
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from typing import List, Dict, Optional, Tuple
import numpy as np
from datasets import Dataset


class BERTClassifier:
    """BERT文本分类器"""

    def __init__(self, model_name: str = "bert-base-chinese", num_labels: int = 2,
                 device: str = None, max_length: int = 512):
        """
        初始化BERT分类器

        Args:
            model_name: BERT模型名称
            num_labels: 分类数量（2表示二分类）
            device: 计算设备
            max_length: 最大序列长度
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"正在加载BERT模型: {model_name}")
        print(f"使用设备: {self.device}")

        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # 加载模型配置
        self.config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )

        # 加载模型
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            config=self.config
        )

        self.model.to(self.device)

        # 标签映射
        self.label_map = {0: 'Human', 1: 'AI'}
        self.reverse_label_map = {'Human': 0, 'AI': 1}

        print("BERT模型加载成功！")

    def tokenize_texts(self, texts: List[str]) -> Dict:
        """
        批量tokenize文本

        Args:
            texts: 文本列表

        Returns:
            Tokenized结果
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def predict(self, text: str) -> Dict:
        """
        预测单个文本

        Args:
            text: 输入文本

        Returns:
            预测结果字典
        """
        self.model.eval()

        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # 移到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 预测
            outputs = self.model(**inputs)
            logits = outputs.logits

            # 计算概率
            probs = torch.softmax(logits, dim=-1)
            ai_prob = probs[0, 1].item()  # AI类的概率
            human_prob = probs[0, 0].item()

            # 预测标签
            pred_label = torch.argmax(logits, dim=-1)[0].item()

        return {
            'label': self.label_map[pred_label],
            'ai_probability': ai_prob,
            'human_probability': human_prob,
            'logits': logits.cpu().numpy()
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        批量预测

        Args:
            texts: 文本列表

        Returns:
            预测结果列表
        """
        self.model.eval()

        results = []
        batch_size = 8  # 批处理大小

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

                for j in range(len(batch_texts)):
                    results.append({
                        'label': self.label_map[torch.argmax(logits[j]).item()],
                        'ai_probability': probs[j, 1].item(),
                        'human_probability': probs[j, 0].item()
                    })

        return results

    def get_embeddings(self, text: str) -> np.ndarray:
        """
        获取BERT的[CLS] token embedding作为文本表示

        Args:
            text: 输入文本

        Returns:
            Embedding向量
        """
        self.model.eval()

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 获取hidden states
            outputs = self.model.bert(**inputs)
            last_hidden_state = outputs.last_hidden_state

            # 取[CLS] token的embedding
            cls_embedding = last_hidden_state[:, 0, :].squeeze()

        return cls_embedding.cpu().numpy()

    def train_model(self, train_texts: List[str], train_labels: List[str],
                    val_texts: List[str] = None, val_labels: List[str] = None,
                    output_dir: str = "./checkpoints/bert",
                    num_epochs: int = 3,
                    batch_size: int = 16,
                    learning_rate: float = 2e-5,
                    warmup_steps: int = 500) -> Dict:
        """
        训练模型

        Args:
            train_texts: 训练文本列表
            train_labels: 训练标签列表 ('Human'/'AI')
            val_texts: 验证文本列表
            val_labels: 验证标签列表
            output_dir: 输出目录
            num_epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            warmup_steps: 预热步数

        Returns:
            训练历史
        """
        # 转换标签
        train_labels_num = [self.reverse_label_map[l] for l in train_labels]

        # 创建数据集
        train_dataset = self._create_dataset(train_texts, train_labels_num)

        val_dataset = None
        if val_texts and val_labels:
            val_labels_num = [self.reverse_label_map[l] for l in val_labels]
            val_dataset = self._create_dataset(val_texts, val_labels_num)

        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_f1" if val_dataset else None,
            greater_is_better=True,
            learning_rate=learning_rate,
            report_to="none",  # 不使用wandb等
            save_total_limit=2
        )

        # 自定义Trainer（添加指标计算）
        trainer = self._create_trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # 训练
        print("开始训练BERT模型...")
        trainer.train()

        # 保存模型
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"模型已保存到: {output_dir}")

        # 评估
        if val_dataset:
            metrics = trainer.evaluate()
            print(f"验证集指标: {metrics}")
            return metrics

        return {}

    def _create_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """创建HuggingFace Dataset"""
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)
        }

        return Dataset.from_dict(dataset_dict)

    def _create_trainer(self, model, args, train_dataset, eval_dataset=None):
        """创建自定义Trainer"""

        class CustomTrainer(Trainer):
            def compute_metrics(self, eval_pred):
                from sklearn.metrics import accuracy_score, precision_recall_fscore_support

                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)

                accuracy = accuracy_score(labels, predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='binary', zero_division=0
                )

                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

        return CustomTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        )

    def load_from_checkpoint(self, checkpoint_path: str):
        """
        从检查点加载模型

        Args:
            checkpoint_path: 检查点路径
        """
        print(f"从检查点加载模型: {checkpoint_path}")
        self.model = BertForSequenceClassification.from_pretrained(checkpoint_path)
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        print("模型加载完成！")

    def save_model(self, save_path: str):
        """
        保存模型

        Args:
            save_path: 保存路径
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")


if __name__ == '__main__':
    # 测试代码
    print("初始化BERT分类器...")

    classifier = BERTClassifier()

    # 测试预测
    test_texts = [
        "这是一段测试文本，用于测试BERT模型的功能。",
        "人工智能是计算机科学的重要分支，它致力于研究如何使计算机能够模拟人类的智能行为。深度学习是人工智能的核心技术之一。"
    ]

    print("\n=== 预测测试 ===")
    for text in test_texts:
        result = classifier.predict(text)
        print(f"\n文本: {text[:50]}...")
        print(f"预测标签: {result['label']}")
        print(f"AI概率: {result['ai_probability']:.4f}")

    # 测试embedding
    print("\n=== Embedding测试 ===")
    emb = classifier.get_embeddings(test_texts[0])
    print(f"Embedding形状: {emb.shape}")
