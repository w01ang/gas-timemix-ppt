#!/usr/bin/env python3
"""
数据异常值检查脚本
检查数据中是否存在NaN、Inf或异常值，这些可能导致训练时出现NaN损失
"""

import pandas as pd
import numpy as np
import sys

def check_data(data_path):
    """检查数据异常值"""
    print("=" * 80)
    print("🔍 数据异常值检查")
    print("=" * 80)
    print(f"\n正在读取数据: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"❌ 无法读取数据: {e}")
        return False
    
    print(f"✅ 数据加载成功")
    print(f"   总行数: {len(df)}")
    print(f"   列名: {df.columns.tolist()}")
    
    # 检查目标列
    if 'OT' not in df.columns:
        print("❌ 未找到目标列'OT'")
        return False
    
    target_col = df['OT']
    
    print("\n" + "=" * 80)
    print("【异常值检测】")
    print("=" * 80)
    
    # 1. 检查NaN
    nan_count = target_col.isna().sum()
    nan_pct = (nan_count / len(df)) * 100
    print(f"\n1. NaN值:")
    print(f"   数量: {nan_count}")
    print(f"   占比: {nan_pct:.2f}%")
    if nan_count > 0:
        print(f"   ❌ 发现NaN值！")
    else:
        print(f"   ✅ 无NaN值")
    
    # 2. 检查Inf
    inf_count = np.isinf(target_col).sum()
    print(f"\n2. Inf值:")
    print(f"   数量: {inf_count}")
    if inf_count > 0:
        print(f"   ❌ 发现Inf值！")
    else:
        print(f"   ✅ 无Inf值")
    
    # 3. 基本统计
    print(f"\n3. 数据统计:")
    valid_data = target_col.dropna()
    valid_data = valid_data[np.isfinite(valid_data)]
    
    if len(valid_data) == 0:
        print("   ❌ 没有有效数据！")
        return False
    
    max_val = valid_data.max()
    min_val = valid_data.min()
    mean_val = valid_data.mean()
    median_val = valid_data.median()
    std_val = valid_data.std()
    
    print(f"   最大值: {max_val:.2f}")
    print(f"   最小值: {min_val:.2f}")
    print(f"   均值: {mean_val:.2f}")
    print(f"   中位数: {median_val:.2f}")
    print(f"   标准差: {std_val:.2f}")
    
    # 4. 检查负值
    negative_count = (valid_data < 0).sum()
    print(f"\n4. 负值:")
    print(f"   数量: {negative_count}")
    if negative_count > 0:
        print(f"   ⚠️  产量不应为负值")
    else:
        print(f"   ✅ 无负值")
    
    # 5. 检查异常大的值
    threshold_high = mean_val + 5 * std_val
    threshold_low = mean_val - 5 * std_val
    outliers_high = (valid_data > threshold_high).sum()
    outliers_low = (valid_data < threshold_low).sum()
    outliers_total = outliers_high + outliers_low
    outliers_pct = (outliers_total / len(valid_data)) * 100
    
    print(f"\n5. 异常值 (超过均值±5倍标准差):")
    print(f"   上界: {threshold_high:.2f}")
    print(f"   下界: {threshold_low:.2f}")
    print(f"   异常高值: {outliers_high}")
    print(f"   异常低值: {outliers_low}")
    print(f"   总异常值: {outliers_total} ({outliers_pct:.2f}%)")
    
    if outliers_pct > 5:
        print(f"   ⚠️  异常值比例较高")
    else:
        print(f"   ✅ 异常值比例正常")
    
    # 6. 检查零值
    zero_count = (valid_data == 0).sum()
    zero_pct = (zero_count / len(valid_data)) * 100
    print(f"\n6. 零值:")
    print(f"   数量: {zero_count} ({zero_pct:.2f}%)")
    if zero_pct > 10:
        print(f"   ⚠️  零值比例较高")
    
    # 7. 按井检查
    if '井号' in df.columns:
        print(f"\n7. 按井统计:")
        well_count = df['井号'].nunique()
        print(f"   总井数: {well_count}")
        
        # 检查每口井的长度
        well_lengths = df.groupby('井号').size()
        print(f"   最长井: {well_lengths.max()} 步")
        print(f"   最短井: {well_lengths.min()} 步")
        print(f"   平均长度: {well_lengths.mean():.1f} 步")
        
        # 检查是否有井全是NaN
        wells_with_nan = df.groupby('井号')['OT'].apply(lambda x: x.isna().all()).sum()
        if wells_with_nan > 0:
            print(f"   ❌ 有 {wells_with_nan} 口井全是NaN值")
    
    # 总结
    print("\n" + "=" * 80)
    print("【总结】")
    print("=" * 80)
    
    issues = []
    if nan_count > 0:
        issues.append(f"NaN值 ({nan_count})")
    if inf_count > 0:
        issues.append(f"Inf值 ({inf_count})")
    if negative_count > 0:
        issues.append(f"负值 ({negative_count})")
    if outliers_pct > 5:
        issues.append(f"异常值比例高 ({outliers_pct:.1f}%)")
    
    if len(issues) == 0:
        print("\n✅ 数据正常，无明显异常")
        print("\n如果训练仍出现NaN，请尝试:")
        print("  1. 降低学习率到 1e-5")
        print("  2. 使用梯度裁剪 (已自动添加)")
        print("  3. 增加batch size到32")
        return True
    else:
        print(f"\n❌ 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"   • {issue}")
        
        print("\n" + "=" * 80)
        print("【建议清理方案】")
        print("=" * 80)
        
        # 生成清理建议
        print("\n是否需要清理数据? (y/n): ", end='')
        try:
            response = input().strip().lower()
            if response == 'y':
                clean_data(df, data_path)
        except:
            print("\n⚠️  请手动运行清理")
        
        return False

def clean_data(df, original_path):
    """清理异常数据"""
    print("\n开始清理数据...")
    
    df_clean = df.copy()
    original_count = len(df_clean)
    
    # 1. 删除NaN和Inf
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna(subset=['OT'])
    print(f"  删除NaN/Inf: {original_count} → {len(df_clean)} 行")
    
    # 2. 删除负值
    df_clean = df_clean[df_clean['OT'] >= 0]
    print(f"  删除负值: → {len(df_clean)} 行")
    
    # 3. 裁剪异常值
    mean_val = df_clean['OT'].mean()
    std_val = df_clean['OT'].std()
    upper_bound = mean_val + 5 * std_val
    lower_bound = max(0, mean_val - 5 * std_val)
    
    before_clip = len(df_clean)
    df_clean['OT'] = df_clean['OT'].clip(lower_bound, upper_bound)
    print(f"  裁剪异常值到 [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # 4. 删除全零的井
    if '井号' in df_clean.columns:
        wells_before = df_clean['井号'].nunique()
        well_means = df_clean.groupby('井号')['OT'].mean()
        valid_wells = well_means[well_means > 0].index
        df_clean = df_clean[df_clean['井号'].isin(valid_wells)]
        wells_after = df_clean['井号'].nunique()
        print(f"  删除零值井: {wells_before} → {wells_after} 口井")
    
    # 保存清理后的数据
    output_path = original_path.replace('.csv', '_cleaned.csv')
    df_clean.to_csv(output_path, index=False)
    
    print(f"\n✅ 清理完成！")
    print(f"   原始数据: {original_count} 行")
    print(f"   清理后: {len(df_clean)} 行 (保留 {len(df_clean)/original_count*100:.1f}%)")
    print(f"   保存路径: {output_path}")
    
    print("\n使用清理后的数据训练:")
    print(f"  --data_path {output_path.split('/')[-1]}")

def main():
    # 默认数据路径
    default_path = '/Users/wangjr/Documents/yk/timemixer/data/preprocessed_daily_gas_by_well.csv'
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = default_path
    
    check_data(data_path)

if __name__ == '__main__':
    main()

