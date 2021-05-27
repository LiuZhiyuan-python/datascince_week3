"""
使用说明：
测试：fit(xtrain,ytrain) 迭代1000次
输出：fit(X,y) 输出csv
"""
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import re

# 文件读取
train_df_org = pd.read_csv('train.csv')
test_df_org = pd.read_csv("test.csv")
test_df_org["Survived"] = 0

# 特征工程
# 1.设置Title属性
combine_train_test = train_df_org.append(test_df_org)
PassengerId = test_df_org['PassengerId']

combine_train_test['Title'] = combine_train_test["Name"].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])

title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))

combine_train_test['Title'] = combine_train_test['Title'].map(title_Dict)

# 为了后面的特征分析，这里我们也将 Title 特征进行facrorizing
combine_train_test['Title'] = pd.factorize(combine_train_test['Title'])[0]
title_dummies_df = pd.get_dummies(combine_train_test['Title'], prefix=combine_train_test[['Title']].columns[0])
combine_train_test = pd.concat([combine_train_test, title_dummies_df], axis=1)

# Sex属性
combine_train_test['Sex'] = pd.factorize(combine_train_test['Sex'])[0]

sex_dummies_df = pd.get_dummies(combine_train_test['Sex'], prefix=combine_train_test[['Sex']].columns[0])
combine_train_test = pd.concat([combine_train_test, sex_dummies_df], axis=1)

# Embark属性
combine_train_test['Embarked'].fillna(combine_train_test['Embarked'].mode()).iloc[0]
combine_train_test['Embarked'] = pd.factorize(combine_train_test['Embarked'])[0]

emb_dummies_df = pd.get_dummies(combine_train_test['Embarked'], prefix=combine_train_test[["Embarked"]].columns[0])
combine_train_test = pd.concat([combine_train_test, emb_dummies_df], axis=1)

# Age属性：补充缺省值
# combine_train_test.Embarked[combine_train_test.Embarked.isnull()] = combine_train_test.Embarked.dropna().mode().values
combine_train_test["Age"] = combine_train_test["Age"].fillna(combine_train_test["Age"].median())

# Pclass属性：根据Fare的再分类
from sklearn.preprocessing import LabelEncoder

# 建立PClass Fare Category
def pclass_fare_category(df, pclass1_mean_fare, pclass2_mean_fare, pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_Low'
        else:
            return 'Pclass1_High'
    elif df['Pclass'] == 2:
        if df['Fare'] <= pclass2_mean_fare:
            return 'Pclass2_Low'
        else:
            return 'Pclass2_High'
    elif df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_Low'
        else:
            return 'Pclass3_High'

Pclass1_mean_fare = combine_train_test['Fare'].groupby(by=combine_train_test['Pclass']).mean().get([1]).values[0]
Pclass2_mean_fare = combine_train_test['Fare'].groupby(by=combine_train_test['Pclass']).mean().get([2]).values[0]
Pclass3_mean_fare = combine_train_test['Fare'].groupby(by=combine_train_test['Pclass']).mean().get([3]).values[0]

# 建立Pclass_Fare Category
combine_train_test['Pclass_Fare_Category'] = combine_train_test.apply(pclass_fare_category, args=(
    Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
pclass_level = LabelEncoder()

# 给每一项添加标签
pclass_level.fit(np.array(
    ['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))

# 转换成数值
combine_train_test['Pclass_Fare_Category'] = pclass_level.transform(combine_train_test['Pclass_Fare_Category'])

# dummy 转换
pclass_dummies_df = pd.get_dummies(combine_train_test['Pclass_Fare_Category']).rename(
    columns=lambda x: 'Pclass_' + str(x))
combine_train_test = pd.concat([combine_train_test, pclass_dummies_df], axis=1)

combine_train_test['Pclass'] = pd.factorize(combine_train_test['Pclass'])[0]

# 模型训练
train_data = combine_train_test[:891]
test_data = combine_train_test[891:]

from sklearn.ensemble import RandomForestClassifier

y = train_data["Survived"]
features = ["Age", "Sex", "Embarked", "Pclass", "Title"]
X = pd.get_dummies(train_data[features])

# 模型验证
rate = list()
for a in range(1):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, shuffle=True)

    model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=1)
    model.fit(X, y)
    n_test = xtest.shape[0]
    y_pred = model.predict(xtest)
    ytest = list(ytest)
    n_right = 0
    for i in range(n_test):
        if ytest[i] == y_pred[i]:
            n_right += 1
    #     else:
    #         print(f"这个样本点所得到的预测类别是{y_pred[i]}，但真实类别是{ytest[i]}")
    rate.append(n_right)

print(f"模型预测的准确率是{np.sum(rate) / n_test / 10}%")

# 测实际试集
X_test = pd.get_dummies(test_data[features])
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
