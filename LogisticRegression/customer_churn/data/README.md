# 电信客户流失预测数据集

## 案例需求

- 已知：用户个人、通话、上网等信息数据。
- 需求：通过分析特征属性，确定用户流失的原因，以及哪些因素可能导致用户流失；建立预测模型判断用户是否流失，并提出用户流失预警策略。

## 数据集概览

| 属性     | 说明              |
| -------- | ----------------- |
| 样本数量 | 7042              |
| 字段数量 | 16 列             |
| 特征数量 | 15 列（不含标签） |
| 标签字段 | Churn             |
| 缺失值   | 0                 |
| 任务类型 | 二分类预测        |

## 标签说明

- Churn：客户是否流失的标记字段。
- No：未流失，5174 条，占比约 73.46%。
- Yes：已流失，1869 条，占比约 26.54%。

## 数据集介绍

这份数据用于电信客户流失预测。样本记录了客户的基础信息、家庭关系、通信与网络服务、合约类型、支付方式以及费用情况，目标是根据这些特征预测客户是否会流失。

当前 CSV 文件中的字段已经过编码和筛选，不包含截图中提到的 CustomerID 字段；其余字段含义与截图描述基本一致。

## 字段说明

| 字段名            | 含义                      |
| ----------------- | ------------------------- |
| Churn             | 客户是否流失的标记字段    |
| gender            | 性别                      |
| Partner_att       | 配偶是否也是 ATT 用户     |
| Dependents_att    | 家人是否也是 ATT 用户     |
| landline          | 是否使用 ATT 固话服务     |
| internet_att      | 是否使用 ATT 的互联网服务 |
| internet_other    | 是否使用其他互联网服务    |
| StreamingTV       | 是否使用在线视频服务      |
| StreamingMovies   | 是否使用在线电影服务      |
| Contract_Month    | 是否为按月合约            |
| Contract_1YR      | 是否为 1 年期合约         |
| PaymentBank       | 是否使用银行转账付款      |
| PaymentCreditcard | 是否使用信用卡付款        |
| PaymentElectronic | 是否使用电子支付          |
| MonthlyCharges    | 每月话费                  |
| TotalCharges      | 累计话费                  |

## 字段分组理解

### 1. 用户基础与家庭关系

- gender
- Partner_att
- Dependents_att

### 2. 通信与网络服务

- landline
- internet_att
- internet_other
- StreamingTV
- StreamingMovies

### 3. 合约与支付方式

- Contract_Month
- Contract_1YR
- PaymentBank
- PaymentCreditcard
- PaymentElectronic

### 4. 费用特征

- MonthlyCharges
- TotalCharges

### 5. 预测目标

- Churn

## 建模说明

- 该数据集适合用于逻辑回归、决策树、随机森林、XGBoost 等分类模型。
- 除 gender 外，多数字段已经是 0 或 1 的数值编码形式。
- MonthlyCharges 和 TotalCharges 为连续数值字段，建模前可根据需要进行标准化。
- 如果要分析客户流失原因，可重点关注合约类型、支付方式、网络服务和费用水平与 Churn 之间的关系。
