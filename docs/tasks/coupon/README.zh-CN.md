# 优惠券接受预测：任务说明（中文对照）

> 非官方中文译文，翻译日期：2026-11-15。对应[英文原文](Readme.md)；有歧义时以原文及老师澄清为准。文件名和字段标识保留原样。

使用你们的 SuperAgent 开发一个分类器，预测驾驶员是否会接受所提供的优惠券：`1` 表示接受，`0` 表示拒绝。

对于每个数据划分（`train`、`val`、`test`），`C_*.npy` 包含描述优惠券、驾驶员和驾驶情境的特征；对应 `y_*.npy` 包含按相同行顺序排列的标签。

使用 `np.load(path, allow_pickle=True)` 加载特征数组（`C_*.npy`）。测试集只用于最终评估，并报告**测试集准确率（test accuracy）**。

每个 `C_*.npy` 的列从左至右依次为：

```text
temperature, time, coupon, expiration, gender, age, maritalStatus,
has_children, education, occupation, income, Bar, CoffeeHouse,
CarryAway, RestaurantLessThan20, Restaurant20To50, toCoupon_GEQ5min,
toCoupon_GEQ15min, toCoupon_GEQ25min, direction_same, direction_opp
```

原始特征含义见 [UCI 数据集页面](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)。其中，`GEQ` 意为“至少”，`opp` 意为“相反方向”。

## 附：课程元信息中文对照

本节不是上述 README 的译文，而是对同目录 [info.json](info.json) 的字段对照；不修改机器读取的 JSON。

| 字段 | 含义 | 课程文件中的值 |
|---|---|---|
| `task_type` | 任务类型 | `binclass`，二分类 |
| `num_classes` | 类别数 | 2 |
| `n_num_features` | 数值特征数 | 0 |
| `n_cat_features` | 类别特征数 | 21 |
| `train_size` | 训练集样本数 | 8117 |
| `val_size` | 验证集样本数 | 2030 |
| `test_size` | 测试集样本数 | 2537 |
| `source` | 原始数据来源 | 上方 UCI 链接 |

安全提示（工作区补充，非原文）：仅对可信课程文件启用 `allow_pickle=True`，不要用其读取不可信文件。
