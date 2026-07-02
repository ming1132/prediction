# 用户逾期行为预测

DataFountain 竞赛项目，使用 LightGBM 预测银行信贷用户是否逾期。

## 项目结构

```
user-overdue-prediction/
├── data/              # 数据文件（需自行下载）
│   ├── train.csv
│   └── test.csv
├── src/               # 代码
│   ├── main.py        # 一键运行
│   ├── preprocess.py  # 数据预处理
│   ├── train.py       # 模型训练
│   └── predict.py     # 预测生成
├── model/             # 保存的模型
├── output/            # 预测结果
├── requirements.txt   # 依赖包
└── README.md
```

## 环境配置

Python 3.8+

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据下载

从 [DataFountain 竞赛页面](https://www.datafountain.cn/competitions/449/datasets) 下载：
- train.csv
- test.csv

放入 `data/` 文件夹。

## 运行方法

```bash
cd src
python main.py
```

运行后会在 `output/submission.csv` 生成提交文件。

## 模型说明

- **算法**：LightGBM
- **验证**：5折交叉验证
- **评价指标**：Macro-F1
- **预处理**：删除缺失率>95%的特征、删除疑似泄露特征、Label Encoding、中位数填充缺失值

## 实验结果

本地5折交叉验证平均 Macro-F1 约为 0.618。

## 作者

[你的名字]
