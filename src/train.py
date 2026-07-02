import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import os
import pickle

from preprocess import load_data, prepare_features


def train_model(X_train, y_train, feature_cols, n_splits=5):
    """使用LightGBM 5折交叉验证训练"""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,          # 增大叶子数，模型更复杂
        'learning_rate': 0.05,     # 学习率不变
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 1000,      # 从500增加到1000，让模型充分训练
        'early_stopping_rounds': 50,
        'min_child_samples': 20,   # 从50降到20，允许更细的分裂
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    models = []
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"第{fold+1}折训练...")

        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(params, train_data, valid_sets=[val_data],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        val_binary = (val_pred > 0.5).astype(int)

        f1 = f1_score(y_val, val_binary, average='macro')
        scores.append(f1)
        oof_preds[val_idx] = val_pred
        models.append(model)
        print(f"  F1={f1:.4f}, 最佳迭代轮数={model.best_iteration}")

    oof_binary = (oof_preds > 0.5).astype(int)
    oof_f1 = f1_score(y_train, oof_binary, average='macro')

    print(f"平均F1: {np.mean(scores):.4f}, OOF F1: {oof_f1:.4f}")

    # 保存模型
    os.makedirs("../model", exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(f"../model/lgb_model_fold{i}.txt")

    with open("../model/feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    return models, oof_preds


if __name__ == "__main__":
    train_df, test_df = load_data("../data/train.csv", "../data/test.csv")
    X_train, y_train, X_test, test_ids, feature_cols = prepare_features(train_df, test_df)
    train_model(X_train, y_train, feature_cols)
