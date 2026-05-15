# -*- coding: utf-8 -*-
"""
Streamlit Web应用
功能：AI作业原创性检测系统的可视化界面
"""

import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 设置中文字体（自动检测）
from utils.font_config import setup_chinese_font
setup_chinese_font()

# 设置页面配置
st.set_page_config(
    page_title="EduGuard - AI作业原创性检测系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入模型和工具
from models.bert_model import BERTClassifier
from models.fusion_model import FusionModel
from utils.preprocess import TextPreprocessor
from utils.features import FeatureExtractor
from utils.advice import AdviceGenerator
from utils.language_detector import LanguageDetector
from utils.ppl import PerplexityCalculator


# 初始化session state
if 'detector_loaded' not in st.session_state:
    st.session_state.detector_loaded = False
    st.session_state.bert_model = None
    st.session_state.fusion_model = None
    st.session_state.preprocessor = None
    st.session_state.feature_extractor = None
    st.session_state.advice_generator = None
    st.session_state.ppl_calculator = None


@st.cache_resource
def load_models(language: str = 'zh'):
    """
    加载模型（缓存）

    Args:
        language: 语言代码 ('zh', 'en', 'auto')
    """
    with st.spinner("正在加载模型..."):
        device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

        # 根据语言选择模型
        if language == 'en':
            model_name = 'bert-base-uncased'
            checkpoint_base = './checkpoints/bert_en'
        else:
            model_name = 'bert-base-chinese'
            checkpoint_base = './checkpoints/bert_zh'

        # 加载BERT - 优先使用本地检查点
        if os.path.exists(checkpoint_base):
            try:
                print(f"从本地检查点加载BERT: {checkpoint_base}")
                bert = BERTClassifier(
                    model_name=checkpoint_base,  # 直接使用本地路径
                    num_labels=2,
                    device=device
                )
                print(f"✅ BERT检查点加载成功: {checkpoint_base}")
            except Exception as e:
                print(f"⚠️ BERT检查点加载失败: {e}，尝试使用在线模型")
                bert = BERTClassifier(
                    model_name=model_name,
                    num_labels=2,
                    device=device
                )
        else:
            print(f"⚠️ BERT检查点不存在: {checkpoint_base}，使用在线模型")
            bert = BERTClassifier(
                model_name=model_name,
                num_labels=2,
                device=device
            )

        # 加载融合模型（使用与训练时相同的hidden_dims）
        fusion = FusionModel(
            bert_model=bert,
            feature_dim=11,
            device=device,
            hidden_dims=[256, 128, 64]
        )

        fusion_checkpoint = f"./checkpoints/fusion_{language}_best.pt"
        # auto模式使用zh的检查点
        if language == 'auto' and not os.path.exists(fusion_checkpoint):
            fusion_checkpoint = "./checkpoints/fusion_zh_best.pt"
        if os.path.exists(fusion_checkpoint):
            try:
                fusion.load_model(fusion_checkpoint)
                print(f"✅ 融合模型检查点加载成功: {fusion_checkpoint}")
            except Exception as e:
                print(f"⚠️ 融合模型检查点加载失败: {e}")
        else:
            print(f"⚠️ 融合模型检查点不存在: {fusion_checkpoint}")

        # 初始化工具
        preprocessor = TextPreprocessor(language=language)
        feature_extractor = FeatureExtractor()
        advice_generator = AdviceGenerator()

        # 初始化PPL计算器（中文GPT2模型）
        try:
            ppl_calculator = PerplexityCalculator(model_name="IDEA-CCNL/Wenzhong-GPT2-110M", device=device)
            print("✅ PPL计算器加载成功")
        except Exception as e:
            print(f"⚠️ PPL计算器加载失败: {e}")
            ppl_calculator = None

        return bert, fusion, preprocessor, feature_extractor, advice_generator, ppl_calculator


def detect_text(text, model_type, bert, fusion, preprocessor, feature_extractor, advice_generator, ppl_calculator=None):
    """检测文本"""
    if not text or len(text.strip()) == 0:
        return None

    # 预处理
    cleaned = preprocessor.clean_text(text)
    paragraphs = preprocessor.split_paragraphs(cleaned)

    # 计算PPL（如果有PPL计算器）
    ppl = None
    if ppl_calculator is not None:
        try:
            ppl = ppl_calculator.calculate_perplexity(cleaned)
            # 检查PPL是否有效
            if not isinstance(ppl, (int, float)) or ppl != ppl or ppl <= 0:  # NaN检查：ppl != ppl
                print(f"无效的PPL值: {ppl}，使用默认值")
                ppl = 100.0
            print(f"PPL: {ppl:.2f}")
        except Exception as e:
            print(f"PPL计算失败: {e}，使用默认值")
            ppl = 100.0

    # 提取特征
    features = feature_extractor.extract_all_features(cleaned, ppl=ppl)

    # 预测
    if model_type == "融合模型 (推荐)":
        result = fusion.predict(cleaned, features)
    else:
        result = bert.predict(cleaned)

    ai_prob = result['ai_probability']

    # 检查AI概率是否有效
    if ai_prob != ai_prob or ai_prob < 0 or ai_prob > 1:  # NaN检查
        print(f"无效的AI概率: {ai_prob}，使用BERT预测")
        # 如果Fusion预测失败，回退到BERT
        result = bert.predict(cleaned)
        ai_prob = result['ai_probability']
        # 检查BERT预测是否也有效
        if ai_prob != ai_prob:
            ai_prob = 0.5  # 完全不确定时默认0.5

    originality_score = (1 - ai_prob) * 100

    # 段落级检测
    paragraph_results = []
    for para in paragraphs:
        if model_type == "融合模型 (推荐)":
            # 计算段落PPL
            para_ppl = None
            if ppl_calculator is not None:
                try:
                    para_ppl = ppl_calculator.calculate_perplexity(para)
                    # 检查PPL是否有效
                    if not isinstance(para_ppl, (int, float)) or para_ppl != para_ppl or para_ppl <= 0:
                        para_ppl = 100.0
                except:
                    para_ppl = 100.0
            para_features = feature_extractor.extract_all_features(para, ppl=para_ppl)
            para_result = fusion.predict(para, para_features)
        else:
            para_result = bert.predict(para)

        # 检查预测结果是否有效
        if 'ai_probability' in para_result and para_result['ai_probability'] == para_result['ai_probability']:
            paragraph_results.append({
                'text': para,
                'ai_probability': para_result['ai_probability'],
                'label': para_result['label']
            })

    # 生成建议
    advices = advice_generator.generate_advice(cleaned, ai_prob, features)

    # 风险分析
    risk_analysis = advice_generator.analyze_paragraph_risk(
        paragraphs, [p['ai_probability'] for p in paragraph_results]
    )

    return {
        'text': text,
        'cleaned_text': cleaned,
        'ai_probability': ai_prob,
        'originality_score': originality_score,
        'label': result['label'],
        'paragraphs': paragraph_results,
        'risk_analysis': risk_analysis,
        'features': features,
        'advices': advices
    }


def render_probability_gauge(ai_prob):
    """渲染AI概率仪表盘"""
    fig, ax = plt.subplots(figsize=(3, 2))

    if ai_prob < 0.3:
        color = '#2ecc71'
        level = 'Low Risk'
    elif ai_prob < 0.7:
        color = '#f39c12'
        level = 'Medium Risk'
    else:
        color = '#e74c3c'
        level = 'High Risk'

    ax.barh(['AI Probability'], [ai_prob], color=color, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability', fontsize=10)
    ax.set_title(f'{level} ({ai_prob:.1%})', fontsize=12, fontweight='bold')

    # 添加阈值线
    ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0.7, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    return fig


def render_originality_score(score):
    """渲染原创性评分"""
    fig, ax = plt.subplots(figsize=(3, 2))

    if score >= 70:
        color = '#2ecc71'
        level = 'Excellent'
    elif score >= 40:
        color = '#f39c12'
        level = 'Pass'
    else:
        color = '#e74c3c'
        level = 'Needs Work'

    ax.barh(['Originality Score'], [score], color=color, height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Score', fontsize=10)
    ax.set_title(f'{level} ({score:.1f}/100)', fontsize=12, fontweight='bold')

    ax.axvline(x=60, color='gray', linestyle='--', alpha=0.3, linewidth=1, label='Passing Line')

    plt.tight_layout()
    return fig


def render_feature_radar(features):
    """渲染特征雷达图"""
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='polar'))

    feature_names = ['PPL\n(Inv)', 'TTR\n(Vocab)', 'Sent Len\n(Norm)',
                     'HWV\n(Variance)', 'Syntax\nComplex']

    # 归一化特征值到0-1
    feature_values = [
        max(0, min(1, 1 - features.get('ppl', 50) / 200)),
        features.get('ttr', 0.5),
        min(1, features.get('sentence_length_mean', 30) / 50),
        features.get('hwv', 0.3),
        features.get('syntactic_complexity', 0.5)
    ]

    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    feature_values += feature_values[:1]
    angles += angles[:1]

    ax.plot(angles, feature_values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, feature_values, alpha=0.25, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Text Feature Analysis', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def render_paragraph_risk(paragraph_results):
    """渲染段落风险图"""
    num_paragraphs = len(paragraph_results)

    # 根据段落数调整图表大小
    if num_paragraphs > 20:
        fig_width = 15
    elif num_paragraphs > 10:
        fig_width = 12
    else:
        fig_width = 10

    fig, ax = plt.subplots(figsize=(fig_width, 4))

    ai_probs = [p['ai_probability'] for p in paragraph_results]
    x_pos = range(1, len(paragraph_results) + 1)

    colors = ['#2ecc71' if p < 0.3 else '#f39c12' if p < 0.7 else '#e74c3c' for p in ai_probs]

    bars = ax.bar(x_pos, ai_probs, color=colors)

    # 添加数值标签
    for bar, prob in zip(bars, ai_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Paragraph No.', fontsize=10)
    ax.set_ylabel('AI Probability', fontsize=10)
    ax.set_title(f'AI Probability by Paragraph ({num_paragraphs} paragraphs)', fontsize=12, fontweight='bold')
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_ylim(0, 1)

    # 设置x轴范围，确保所有段落都显示
    ax.set_xlim(0, num_paragraphs + 1)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Low Risk (<30%)'),
        Patch(facecolor='#f39c12', label='Medium Risk (30%-70%)'),
        Patch(facecolor='#e74c3c', label='High Risk (>70%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    return fig


def render_ppl_curve(paragraph_results):
    """渲染段落PPL变化曲线"""
    num_paragraphs = len(paragraph_results)

    # 根据段落数调整图表大小
    if num_paragraphs > 20:
        fig_width = 15
    elif num_paragraphs > 10:
        fig_width = 12
    else:
        fig_width = 10

    fig, ax = plt.subplots(figsize=(fig_width, 4))

    # 提取PPL值（如果有的话）
    ppl_values = []
    for p in paragraph_results:
        # 如果段落结果中有ppl信息
        if 'ppl' in p:
            ppl_values.append(p['ppl'])
        else:
            # 使用AI概率作为代理（概率越高，PPL越低）
            ppl_values.append(100 * (1 - p['ai_probability']) + 10)

    x_pos = range(1, len(paragraph_results) + 1)

    # 绘制曲线
    ax.plot(x_pos, ppl_values, marker='o', linewidth=2, markersize=6, color='#3498db')
    ax.fill_between(x_pos, ppl_values, alpha=0.3, color='#3498db')

    # 添加数值标签
    for i, ppl in enumerate(ppl_values):
        ax.text(i + 1, ppl, f'{ppl:.1f}', ha='center', va='bottom', fontsize=8)

    # 添加阈值线
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Medium PPL')

    ax.set_xlabel('Paragraph No.', fontsize=10)
    ax.set_ylabel('Perplexity (PPL)', fontsize=10)
    ax.set_title(f'PPL by Paragraph ({num_paragraphs} paragraphs)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_paragraphs + 1)

    plt.tight_layout()
    return fig


def render_ppl_distribution(all_ppl_values):
    """渲染PPL分布直方图"""
    if not all_ppl_values or len(all_ppl_values) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))

    # 绘制直方图
    n, bins, patches = ax.hist(all_ppl_values, bins=20, color='#3498db', alpha=0.7, edgecolor='white')

    # 添加均值线
    mean_ppl = np.mean(all_ppl_values)
    ax.axvline(x=mean_ppl, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ppl:.1f}')

    # 添加中位数线
    median_ppl = np.median(all_ppl_values)
    ax.axvline(x=median_ppl, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_ppl:.1f}')

    ax.set_xlabel('Perplexity (PPL)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('PPL Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def main():
    """主函数"""

    # 侧边栏
    with st.sidebar:
        st.title("🔍 EduGuard")
        st.markdown("---")

        # 模型选择
        model_type = st.selectbox(
            "选择检测模型",
            ["融合模型 (推荐)", "BERT模型"]
        )

        # 语言选择
        language = st.selectbox(
            "选择语言/模式",
            ["自动检测", "中文", "英文"],
            index=0
        )
        language_map = {"自动检测": "auto", "中文": "zh", "英文": "en"}
        selected_language = language_map[language]

        st.markdown("---")

        # 说明
        st.markdown("""
        ### 📖 使用说明

        1. **输入文本**：在文本框中输入或粘贴待检测的作业文本

        2. **上传文件**：支持上传 .txt 文件

        3. **查看结果**：
           - AI生成概率
           - 原创性评分
           - 段落级风险分析
           - 修改建议

        ### ⚠️ 注意事项

        - 建议输入文本长度 ≥ 50 字
        - 本系统仅供参考
        - 最终判断请结合实际情况
        """)

        st.markdown("---")

        # 加载状态
        if not st.session_state.detector_loaded:
            if st.button("加载模型", type="primary"):
                try:
                    bert, fusion, preprocessor, feature_extractor, advice_generator, ppl_calculator = load_models(selected_language)

                    st.session_state.bert_model = bert
                    st.session_state.fusion_model = fusion
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_extractor = feature_extractor
                    st.session_state.advice_generator = advice_generator
                    st.session_state.ppl_calculator = ppl_calculator
                    st.session_state.detector_loaded = True
                    st.session_state.current_language = selected_language

                    st.success("模型加载成功！")
                    st.rerun()
                except Exception as e:
                    st.error(f"模型加载失败: {str(e)}")
        else:
            # 如果语言改变了，重新加载模型
            if st.session_state.get('current_language', 'zh') != selected_language:
                if st.button("切换语言模型", type="secondary"):
                    st.session_state.detector_loaded = False
                    st.rerun()
            else:
                st.success("✅ 模型已就绪")
                if st.button("重新加载模型"):
                    st.session_state.detector_loaded = False
                    st.rerun()

    # 主内容区
    st.title("AI作业原创性检测系统")
    st.markdown("---")

    # 检查模型是否加载
    if not st.session_state.detector_loaded:
        st.info("👈 请先在侧边栏加载模型")
        st.stop()

    # 输入选项
    tab1, tab2 = st.tabs(["📝 文本输入", "📁 文件上传"])

    with tab1:
        input_text = st.text_area(
            "请输入待检测的文本：",
            height=200,
            placeholder="在此输入或粘贴您的作业文本..."
        )

        col1, col2 = st.columns(2)
        with col1:
            detect_btn = st.button("开始检测", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("清空", use_container_width=True)

        if clear_btn:
            input_text = ""
            st.rerun()

    with tab2:
        uploaded_file = st.file_uploader("上传文本文件", type=['txt'])

        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            input_text = stringio.read()

            st.text_area("文件内容：", input_text, height=200, disabled=True)

            if st.button("检测文件内容", type="primary"):
                detect_btn = True

    # 检测逻辑
    if detect_btn and input_text:
        if len(input_text.strip()) < 10:
            st.warning("⚠️ 输入文本太短，建议输入至少50个字符")
        else:
            with st.spinner("正在检测中..."):
                result = detect_text(
                    input_text,
                    model_type,
                    st.session_state.bert_model,
                    st.session_state.fusion_model,
                    st.session_state.preprocessor,
                    st.session_state.feature_extractor,
                    st.session_state.advice_generator,
                    st.session_state.ppl_calculator
                )

            if result:
                # 显示结果
                st.success("✅ 检测完成！")

                # 主要指标
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("检测标签", result['label'])

                with col2:
                    st.metric("AI生成概率", f"{result['ai_probability']:.2%}")

                with col3:
                    score = result['originality_score']
                    delta_color = "normal" if score >= 60 else "inverse"
                    st.metric("原创性评分", f"{score:.1f}/100", delta=f"{score:.1f}")

                st.markdown("---")

                # 可视化
                col1, col2 = st.columns(2)

                with col1:
                    st.pyplot(render_probability_gauge(result['ai_probability']))

                with col2:
                    st.pyplot(render_originality_score(result['originality_score']))

                st.markdown("---")

                # 段落分析
                if result['paragraphs']:
                    st.subheader("📊 段落风险分析")

                    # 段落风险图
                    st.pyplot(render_paragraph_risk(result['paragraphs']))

                    # PPL曲线图
                    st.pyplot(render_ppl_curve(result['paragraphs']))

                    # 段落详情（可展开）
                    with st.expander("查看段落详情"):
                        for i, para in enumerate(result['paragraphs'], 1):
                            risk_label = "🟢 低风险" if para['ai_probability'] < 0.3 else \
                                        "🟡 中风险" if para['ai_probability'] < 0.7 else \
                                        "🔴 高风险"

                            st.markdown(f"""
                            **段落 {i}** {risk_label} (AI概率: {para['ai_probability']:.2%})

                            {para['text'][:200]}{'...' if len(para['text']) > 200 else ''}
                            """)
                            st.markdown("---")

                # 特征分析
                with st.expander("🔍 特征详细分析"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.pyplot(render_feature_radar(result['features']))

                    with col2:
                        st.subheader("特征数值")
                        feature_df = pd.DataFrame([
                            ['困惑度', f"{result['features'].get('ppl', 0):.2f}"],
                            ['词汇多样性 (TTR)', f"{result['features'].get('ttr', 0):.4f}"],
                            ['句长均值', f"{result['features'].get('sentence_length_mean', 0):.2f}"],
                            ['句长方差', f"{result['features'].get('sentence_length_var', 0):.2f}"],
                            ['写作方差 (HWV)', f"{result['features'].get('hwv', 0):.4f}"],
                            ['句法复杂度', f"{result['features'].get('syntactic_complexity', 0):.4f}"],
                        ], columns=['特征', '数值'])
                        st.dataframe(feature_df, use_container_width=True, hide_index=True)

                # 修改建议
                if result['advices']:
                    st.subheader("💡 修改建议")

                    for i, advice in enumerate(result['advices'], 1):
                        st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <strong>{i}. [{advice['type']}]</strong> {advice['advice']}
                        </div>
                        """, unsafe_allow_html=True)

            else:
                st.error("检测失败，请重试")

    # 示例文本
    st.markdown("---")
    st.subheader("📋 示例文本")

    example_tab1, example_tab2 = st.tabs(["AI生成示例", "人类写作示例"])

    with example_tab1:
        st.code("""人工智能是计算机科学的重要分支，它致力于研究如何使计算机能够模拟人类的智能行为。深度学习是人工智能的核心技术之一，它通过多层神经网络来学习数据的特征表示。

首先，神经网络的基本单元是神经元，它接收输入信号并产生输出。其次，多层神经网络可以学习更复杂的特征。此外，深度学习在图像识别、自然语言处理等领域取得了显著成果。

总的来说，人工智能技术的发展为各个行业带来了新的机遇，同时也带来了一些挑战。""", language="text")

        if st.button("使用AI示例进行检测"):
            input_text = """人工智能是计算机科学的重要分支，它致力于研究如何使计算机能够模拟人类的智能行为。深度学习是人工智能的核心技术之一，它通过多层神经网络来学习数据的特征表示。

首先，神经网络的基本单元是神经元，它接收输入信号并产生输出。其次，多层神经网络可以学习更复杂的特征。此外，深度学习在图像识别、自然语言处理等领域取得了显著成果。

总的来说，人工智能技术的发展为各个行业带来了新的机遇，同时也带来了一些挑战。"""
            st.rerun()

    with example_tab2:
        st.code("""哎，今天天气真不错啊！我想起了小时候，那时候夏天总是那么漫长。妈妈总是在院子里摆个小桌子，我们就坐在那儿乘凉、吃西瓜。那种感觉，现在想起来还是甜甜的。

不过话说回来，现在的孩子可能很难体会到了吧。前几天我去菜市场买菜，看到卖西瓜的大爷，突然就想起奶奶了。她以前总说挑西瓜要看纹路，还要听听声音，咚咚的才好。

其实我也不懂，但每次买回来的西瓜确实挺甜的，可能是运气好吧。说真的，现在的生活节奏太快了，有时候真想慢下来，好好感受一下生活。""", language="text")

        if st.button("使用人类示例进行检测"):
            input_text = """哎，今天天气真不错啊！我想起了小时候，那时候夏天总是那么漫长。妈妈总是在院子里摆个小桌子，我们就坐在那儿乘凉、吃西瓜。那种感觉，现在想起来还是甜甜的。

不过话说回来，现在的孩子可能很难体会到了吧。前几天我去菜市场买菜，看到卖西瓜的大爷，突然就想起奶奶了。她以前总说挑西瓜要看纹路，还要听听声音，咚咚的才好。

其实我也不懂，但每次买回来的西瓜确实挺甜的，可能是运气好吧。说真的，现在的生活节奏太快了，有时候真想慢下来，好好感受一下生活。"""
            st.rerun()


if __name__ == '__main__':
    main()
