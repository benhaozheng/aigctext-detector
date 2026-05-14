# -*- coding: utf-8 -*-
"""
修复hc3_all.csv中的格式问题
处理文本中的换行符，使pandas能正确解析
"""

import pandas as pd
import re

print("正在读取hc3_all.csv...")
print("注意：由于格式问题，可能需要多次尝试")

# 方法1: 使用quoting=8处理引号内的换行符
try:
    print("\n=== 方法1: 使用quoting参数 ===")
    df = pd.read_csv('./data/hc3_all.csv', encoding='utf-8-sig',
                     quotechar='"', quoting=1, on_bad_lines='warn')
    print(f"成功加载 {len(df)} 行")
    print(f"列名: {list(df.columns)}")

    # 检查标签
    if 'label' in df.columns:
        print(f"\n标签分布:")
        print(df['label'].value_counts())

    # 保存修复后的文件
    output_path = './data/hc3_all_fixed.csv'
    df.to_csv(output_path, index=False, quoting=1)
    print(f"\n✅ 已保存到 {output_path}")

except Exception as e:
    print(f"方法1失败: {e}")

    # 方法2: 手动修复换行符
    print("\n=== 方法2: 手动修复CSV文件 ===")
    fixed_lines = []
    with open('./data/hc3_all.csv', 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # 简单统计
    lines = content.split('\n')
    print(f"原始文件有 {len(lines)} 行")

    # 保存原始备份，使用pandas尽可能多的读取
    df = pd.read_csv('./data/hc3_all.csv', encoding='utf-8-sig', on_bad_lines='skip')
    print(f"pandas能解析 {len(df)} 行")

    if 'label' in df.columns:
        print(f"\n当前标签分布:")
        print(df['label'].value_counts())

        # 清理文本中的换行符
        df['text'] = df['text'].str.replace('\n', ' ', regex=True).str.replace('\r', ' ', regex=True)

        # 保存
        output_path = './data/hc3_all_fixed.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✅ 已保存清理后的数据到 {output_path}")
        print(f"共 {len(df)} 条数据")
