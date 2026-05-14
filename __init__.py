# -*- coding: utf-8 -*-
"""
EduGuard - AI作业原创性检测系统
主模块初始化文件
"""

__version__ = '1.0.0'
__author__ = 'EduGuard Team'

from .models import BERTClassifier, FusionModel
from .utils import TextPreprocessor, PerplexityCalculator, FeatureExtractor, AdviceGenerator

__all__ = [
    'BERTClassifier',
    'FusionModel',
    'TextPreprocessor',
    'PerplexityCalculator',
    'FeatureExtractor',
    'AdviceGenerator'
]
