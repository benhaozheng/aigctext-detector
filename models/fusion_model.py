# -*- coding: utf-8 -*-
"""
融合模型模块
功能：融合BERT embedding + 统计特征 + 语言学特征的MLP分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
import os


class FusionDataset(Dataset):
    """融合模型数据集"""

    def __init__(self, embeddings: np.ndarray, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            embeddings: BERT embeddings (N, 768)
            features: 统计+语言学特征 (N, feature_dim)
            labels: 标签 (N,)
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'features': self.features[idx],
            'label': self.labels[idx]
        }


class FusionMLP(nn.Module):
    """融合网络的MLP部分"""

    def __init__(self, bert_dim: int = 768, feature_dim: int = 11,
                 hidden_dims: List[int] = [512, 256, 128], dropout: float = 0.3):
        """
        Args:
            bert_dim: BERT embedding维度
            feature_dim: 手工特征维度
            hidden_dims: 隐藏层维度列表
            dropout: Dropout概率
        """
        super(FusionMLP, self).__init__()

        # 输入维度 = BERT维度 + 手工特征维度
        input_dim = bert_dim + feature_dim

        # 构建MLP层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # 输出层（二分类）
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, bert_emb, features):
        """
        前向传播

        Args:
            bert_emb: BERT embedding (batch_size, bert_dim)
            features: 手工特征 (batch_size, feature_dim)

        Returns:
            logits (batch_size, 2)
        """
        # 拼接特征
        combined = torch.cat([bert_emb, features], dim=1)

        # 通过MLP
        logits = self.network(combined)

        return logits


class FusionModel:
    """融合模型包装类"""

    def __init__(self, bert_model=None, feature_dim: int = 11,
                 device: str = None, hidden_dims: List[int] = [512, 256, 128]):
        """
        Args:
            bert_model: BERT模型实例
            feature_dim: 手工特征维度
            device: 计算设备
            hidden_dims: MLP隐藏层维度
        """
        self.bert_model = bert_model
        self.feature_dim = feature_dim
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建MLP网络
        self.mlp = FusionMLP(
            bert_dim=768,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims
        ).to(self.device)

        # 优化器
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        # 标签映射
        self.label_map = {0: 'Human', 1: 'AI'}
        self.reverse_label_map = {'Human': 0, 'AI': 1}

        print(f"融合模型已初始化，设备: {self.device}")

    def prepare_data(self, texts: List[str], labels: List[str] = None,
                     features_list: List[Dict] = None) -> Tuple:
        """
        准备训练/预测数据

        Args:
            texts: 文本列表
            labels: 标签列表
            features_list: 预计算的特征列表

        Returns:
            (embeddings, features, labels)
        """
        embeddings = []
        features = []

        print("正在提取BERT embeddings...")
        for i, text in enumerate(texts):
            if self.bert_model:
                emb = self.bert_model.get_embeddings(text)
                embeddings.append(emb)
            else:
                # 如果没有BERT模型，使用零向量
                embeddings.append(np.zeros(768))

            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{len(texts)} 个样本")

        embeddings = np.array(embeddings)

        if features_list:
            # 使用预计算的特征
            from ..utils.features import FeatureExtractor
            extractor = FeatureExtractor()
            features = np.array([extractor.features_to_vector(f) for f in features_list])
        else:
            # 如果没有提供特征，使用零向量
            features = np.zeros((len(texts), self.feature_dim))

        labels_array = None
        if labels:
            labels_array = np.array([self.reverse_label_map[l] for l in labels])

        return embeddings, features, labels_array

    def train(self, train_texts: List[str], train_labels: List[str],
              train_features: List[Dict],
              val_texts: List[str] = None, val_labels: List[str] = None,
              val_features: List[Dict] = None,
              num_epochs: int = 10, batch_size: int = 32,
              learning_rate: float = 1e-4, patience: int = 3) -> Dict:
        """
        训练融合模型

        Args:
            train_texts: 训练文本
            train_labels: 训练标签
            train_features: 训练特征
            val_texts: 验证文本
            val_labels: 验证标签
            val_features: 验证特征
            num_epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            patience: 早停耐心值

        Returns:
            训练历史
        """
        # 准备数据
        train_emb, train_feat, train_y = self.prepare_data(
            train_texts, train_labels, train_features
        )

        val_emb, val_feat, val_y = None, None, None
        if val_texts:
            val_emb, val_feat, val_y = self.prepare_data(
                val_texts, val_labels, val_features
            )

        # 创建数据集
        train_dataset = FusionDataset(train_emb, train_feat, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_texts:
            val_dataset = FusionDataset(val_emb, val_feat, val_y)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 优化器
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learning_rate)

        # 训练循环
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # 训练
            self.mlp.train()
            train_loss = 0.0

            for batch in train_loader:
                embedding = batch['embedding'].to(self.device)
                feature = batch['features'].to(self.device)
                label = batch['label'].to(self.device)

                # 前向传播
                logits = self.mlp(embedding, feature)
                loss = self.criterion(logits, label)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            # 验证
            if val_texts:
                val_metrics = self._evaluate(val_loader)
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                history['val_f1'].append(val_metrics['f1'])

                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Val F1: {val_metrics['f1']:.4f}")

                # 早停
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    self.save_model("./checkpoints/fusion_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停触发，停止训练")
                        break
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

        return history

    def _evaluate(self, data_loader: DataLoader) -> Dict:
        """评估模型"""
        self.mlp.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                embedding = batch['embedding'].to(self.device)
                feature = batch['features'].to(self.device)
                label = batch['label'].to(self.device)

                logits = self.mlp(embedding, feature)
                loss = self.criterion(logits, label)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        # 计算指标
        from sklearn.metrics import accuracy_score, f1_score

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return {
            'loss': total_loss / len(data_loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0)
        }

    def predict(self, text: str, features: Dict, return_embedding: bool = False) -> Dict:
        """
        预测单个文本

        Args:
            text: 输入文本
            features: 特征字典
            return_embedding: 是否返回embedding

        Returns:
            预测结果
        """
        self.mlp.eval()

        # 获取BERT embedding
        if self.bert_model:
            embedding = self.bert_model.get_embeddings(text)
        else:
            embedding = np.zeros(768)

        # 转换特征
        from ..utils.features import FeatureExtractor
        extractor = FeatureExtractor()
        feature_vector = extractor.features_to_vector(features)

        # 转换为tensor
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
        feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            logits = self.mlp(embedding_tensor, feature_tensor)
            probs = F.softmax(logits, dim=-1)

            ai_prob = probs[0, 1].item()
            human_prob = probs[0, 0].item()
            pred_label = torch.argmax(logits, dim=-1)[0].item()

        result = {
            'label': self.label_map[pred_label],
            'ai_probability': ai_prob,
            'human_probability': human_prob
        }

        if return_embedding:
            result['embedding'] = embedding

        return result

    def predict_batch(self, texts: List[str], features_list: List[Dict]) -> List[Dict]:
        """批量预测"""
        results = []

        for text, features in zip(texts, features_list):
            result = self.predict(text, features)
            results.append(result)

        return results

    def save_model(self, save_path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save({
            'model_state_dict': self.mlp.state_dict(),
            'feature_dim': self.feature_dim,
        }, save_path)

        print(f"模型已保存到: {save_path}")

    def load_model(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)

        self.mlp.load_state_dict(checkpoint['model_state_dict'])
        self.feature_dim = checkpoint.get('feature_dim', 11)

        self.mlp.to(self.device)
        print(f"模型已从 {load_path} 加载")


if __name__ == '__main__':
    # 测试代码
    print("初始化融合模型...")

    fusion = FusionModel(feature_dim=11)

    # 测试数据
    test_features = {
        'ppl': 50.0,
        'ttr': 0.6,
        'sentence_length_mean': 30.0,
        'sentence_length_var': 100.0,
        'repetition_rate': 0.1,
        'syntactic_complexity': 0.5,
        'hwv': 0.3,
        'avg_word_length': 2.0,
        'punctuation_density': 0.1,
        'pos_diversity': 0.3,
        'function_word_ratio': 0.3
    }

    test_text = "这是一段测试文本，用于测试融合模型的功能。"

    print("\n=== 预测测试 ===")
    result = fusion.predict(test_text, test_features)
    print(f"预测标签: {result['label']}")
    print(f"AI概率: {result['ai_probability']:.4f}")
