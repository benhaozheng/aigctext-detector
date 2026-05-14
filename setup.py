# -*- coding: utf-8 -*-
"""
快速设置脚本
功能：自动检查和安装依赖
"""

import subprocess
import sys
import os


def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("需要Python 3.8或更高版本")
        return False
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """安装依赖"""
    print("\n正在安装依赖包...")
    print("这可能需要几分钟，请耐心等待...")

    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install',
            '-r', 'requirements.txt',
            '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'  # 使用清华镜像
        ])
        print("\n✅ 依赖安装成功！")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ 依赖安装失败")
        print("请尝试手动安装: pip install -r requirements.txt")
        return False


def check_model_download():
    """检查模型下载"""
    print("\n检查BERT模型...")

    try:
        from transformers import BertTokenizer
        print("正在下载 bert-base-chinese 模型（首次运行约400MB）...")
        BertTokenizer.from_pretrained("bert-base-chinese")
        print("✅ 模型准备就绪")
        return True
    except Exception as e:
        print(f"⚠️ 模型检查失败: {e}")
        print("首次运行时会自动下载")
        return True


def create_directories():
    """创建必要的目录"""
    directories = ['./checkpoints', './data', './logs']

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✅ 目录结构已创建")


def print_usage_guide():
    """打印使用说明"""
    print("\n" + "=" * 60)
    print("EduGuard 设置完成！".center(60))
    print("=" * 60)
    print("""
📖 使用方法:

1️⃣  启动Web应用:
    python run_app.py
    或
    streamlit run app/streamlit_app.py

2️⃣  命令行检测:
    python predict.py --text "待检测的文本"

3️⃣  训练模型:
    python train.py

📚 更多信息请参考 README.md
    """)
    print("=" * 60)


def main():
    """主函数"""
    print("=" * 60)
    print("EduGuard - AI作业原创性检测系统".center(60))
    print("快速设置向导".center(60))
    print("=" * 60)
    print()

    # 检查Python版本
    if not check_python_version():
        return

    # 创建目录
    create_directories()

    # 安装依赖
    if input("\n是否安装依赖? (y/n): ").lower() == 'y':
        if not install_dependencies():
            return

    # 检查模型
    check_model_download()

    # 打印使用说明
    print_usage_guide()


if __name__ == '__main__':
    main()
