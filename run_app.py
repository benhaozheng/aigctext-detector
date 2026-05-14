# -*- coding: utf-8 -*-
"""
快速启动脚本
功能：一键启动Streamlit应用
"""

import os
import sys
import subprocess

def main():
    """启动Streamlit应用"""
    print("=" * 60)
    print("EduGuard - AI作业原创性检测系统".center(60))
    print("=" * 60)
    print()

    # 检查依赖
    try:
        import streamlit
        import torch
        import transformers
        print("✅ 所有依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return

    # 检查应用文件
    app_path = os.path.join(os.path.dirname(__file__), 'app', 'streamlit_app.py')
    if not os.path.exists(app_path):
        print(f"❌ 应用文件不存在: {app_path}")
        return

    print()
    print("🚀 正在启动Web应用...")
    print("📱 浏览器访问地址: http://localhost:8501")
    print()
    print("按 Ctrl+C 停止应用")
    print("=" * 60)
    print()

    # 启动Streamlit
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path])


if __name__ == '__main__':
    main()
