# -*- coding: utf-8 -*-
import os
import pandas as pd

os.chdir('/home/eric_zhao/projects/aigctextdetector')

csv_paths = ['./data/hc3_all_fixed.csv', './data/ai_vs_human_text_2026.csv', './data/sample_data.csv']
dfs = []

for path in csv_paths:
    if os.path.exists(path):
        print(f"\n从CSV文件加载数据: {path}")
        try:
            if 'hc3' in path:
                df = pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='skip')
            else:
                df = pd.read_csv(path, on_bad_lines='skip')

            if 'text' not in df.columns or 'label' not in df.columns:
                print(f"跳过 {path}：格式错误")
                continue

            # 清理数据
            df = df.dropna(subset=['text', 'label'])
            df['text'] = df['text'].astype(str)

            df['label'] = df['label'].str.strip().str.lower()
            df['label'] = df['label'].replace({'ai': 'AI', 'human': 'Human'})

            df = df[df['text'].str.len() > 10]
            df = df[df['label'].isin(['AI', 'Human'])]

            if len(df) > 0:
                dfs.append(df)
                print(f"✅ {path}: 加载 {len(df)} 条数据")
                print(f"   AI: {sum(df['label']=='AI')}, Human: {sum(df['label']=='Human')}")
            else:
                print(f"⚠️ {path}: 没有有效数据")
        except Exception as e:
            print(f"⚠️ 加载 {path} 失败: {e}")

if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ 总计加载 {len(combined_df)} 条数据")

    ai_count = sum(combined_df['label'] == 'AI')
    human_count = sum(combined_df['label'] == 'Human')
    print(f"   AI: {ai_count}, Human: {human_count}")

    ratio = max(ai_count, human_count) / min(ai_count, human_count)
    print(f"   比例: {ratio:.2f}:1")

    if ratio > 2.0:
        min_count = min(ai_count, human_count)
        ai_df = combined_df[combined_df['label'] == 'AI'].sample(n=min(ai_count, min_count), random_state=42)
        human_df = combined_df[combined_df['label'] == 'Human'].sample(n=min(human_count, min_count), random_state=42)
        combined_df = pd.concat([ai_df, human_df], ignore_index=True)
        print(f"\n✅ 平衡后: {len(combined_df)} 条数据")
        print(f"   AI: {sum(combined_df['label']=='AI')}, Human: {sum(combined_df['label']=='Human')}")
