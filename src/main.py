import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_data, prepare_features
from train import train_model
from predict import predict_and_submit


def main():
    print("="*50)
    print("用户逾期行为预测")
    print("="*50)

    # 清理旧模型
    if os.path.exists("../model"):
        shutil.rmtree("../model")
    os.makedirs("../model", exist_ok=True)

    # 加载数据
    print("[1] 加载数据...")
    train_df, test_df = load_data("../data/train.csv", "../data/test.csv")

    # 预处理
    print("[2] 数据预处理...")
    X_train, y_train, X_test, test_ids, feature_cols = prepare_features(train_df, test_df)

    # 训练
    print("[3] 训练模型...")
    models, _ = train_model(X_train, y_train, feature_cols)

    # 预测
    print("[4] 生成预测结果...")
    predict_and_submit(models, X_test, test_ids)

    print("" + "="*50)
    print("完成！提交文件: output/submission.csv")
    print("="*50)


if __name__ == "__main__":
    main()
