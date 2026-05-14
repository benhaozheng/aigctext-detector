# -*- coding: utf-8 -*-
"""
模型模块初始化文件
EduGuard - AI作业原创性检测系统
"""

from .bert_model import BERTClassifier
from .fusion_model import FusionModel

__all__ = [
    'BERTClassifier',
    'FusionModel'
]
