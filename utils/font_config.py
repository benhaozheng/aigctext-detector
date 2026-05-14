# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_chinese_font():
    system = platform.system()

    if system == 'Linux':
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    else:
        font_list = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']

    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font in font_list:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"Using font: {font}")
            return font

    print("Warning: No Chinese font found")
    return None
