import pandas as pd
import numpy as np
import lightgbm as lgb
import os

from preprocess import load_data, prepare_features


def predict_and_submit(models, X_test, test_ids, output_path="../output/submission.csv"):
    # 预测并生成提交文件
    preds = np.zeros(len(X_test))
    for model in models:
        preds += model.predict(X_test, num_iteration=model.best_iteration) / len(models)

    preds_binary = (preds > 0.3).astype(int)

    submission = pd.DataFrame({
        'CUST_ID': test_ids.astype(int),
        'bad_good': preds_binary
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"提交文件已保存: {output_path}")
    print(f"预测结果:{submission['bad_good'].value_counts()}")
    return submission


if __name__ == "__main__":
    train_df, test_df = load_data("../data/train.csv", "../data/test.csv")
    X_train, y_train, X_test, test_ids, feature_cols = prepare_features(train_df, test_df)

    models = []
    for i in range(5):
        models.append(lgb.Booster(model_file=f"../model/lgb_model_fold{i}.txt"))

    predict_and_submit(models, X_test, test_ids)
