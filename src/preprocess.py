import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(train_path, test_path):
    # 加载数据
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    print(f"训练集: {train_df.shape}, 测试集: {test_df.shape}")
    return train_df, test_df


def detect_leakage(train_df, target='bad_good'):
    # 检测并移除可能泄露目标信息的特征
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]

    corr = train_df[numeric_cols + [target]].corr()[target].abs().sort_values(ascending=False)
    leakage = corr[corr > 0.50].index.tolist()
    leakage = [c for c in leakage if c != target]

    # 关键词匹配
    keywords = ['FLAG', 'OVERDUE', 'DELINQ', 'DEFAULT', 'BAD', 'STATUS', 'RISK']
    for col in train_df.columns:
        if any(k in col.upper() for k in keywords) and col != target:
            if col not in leakage:
                leakage.append(col)

    if leakage:
        print(f"移除 {len(leakage)} 个泄露特征")
    return leakage


def clean_data(df, leakage_cols):
    drop_cols = []

    # 缺失率过高
    missing = df.isnull().sum() / len(df)
    drop_cols.extend(missing[missing > 0.95].index.tolist())

    # 常数列
    for col in df.columns:
        if df[col].nunique() <= 1:
            drop_cols.append(col)

    # 泄露特征
    drop_cols.extend(leakage_cols)

    drop_cols = list(set(drop_cols))
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df, drop_cols


def encode_features(train_df, test_df):
    # 对类别特征进行Label Encoding
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['CUST_ID', 'bad_good']]

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = combined[col].fillna('MISSING')
        combined[col] = le.fit_transform(combined[col].astype(str))

    train_enc = combined.iloc[:len(train_df)].copy()
    test_enc = combined.iloc[len(train_df):].copy()
    return train_enc, test_enc


def fill_missing(train_df, test_df):
    # 用中位数填充缺失值
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ['CUST_ID', 'bad_good']]

    for col in num_cols:
        median = train_df[col].median()
        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)
    return train_df, test_df


def prepare_features(train_df, test_df):
    # 检测泄露
    leakage = detect_leakage(train_df)

    # 清洗
    train_clean, _ = clean_data(train_df, leakage)
    test_clean = test_df.drop(columns=[c for c in leakage if c in test_df.columns])

    # 编码
    train_enc, test_enc = encode_features(train_clean, test_clean)

    # 填充缺失
    train_fill, test_fill = fill_missing(train_enc, test_enc)

    # 分离特征和标签
    feature_cols = [c for c in train_fill.columns if c not in ['CUST_ID', 'bad_good']]
    X_train = train_fill[feature_cols]
    y_train = train_fill['bad_good']
    X_test = test_fill[feature_cols]
    test_ids = test_fill['CUST_ID']

    print(f"特征数量: {len(feature_cols)}")
    return X_train, y_train, X_test, test_ids, feature_cols
