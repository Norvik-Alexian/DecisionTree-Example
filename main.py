import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

warnings.simplefilter(action="ignore", category=FutureWarning)

col = [f'Class Name', 'Left weight', 'Left distance', 'Right weight', 'Right distance']

df = pd.read_csv('./dataset/balance-scale.data', sep=',', names=col)

print(df.head())
print(df.dtypes)
print(df.isnull().sum())
print(df.columns)
print(df.describe())
print(df.info())

sns.countplot(df['Class Name'])
sns.countplot(df['Left weight'], hue=df['Class Name'])
sns.countplot(df['Right weight'], hue=df['Class Name'])
# plt.show()

X = df.drop('Class Name', axis=1)
y = df[['Class Name']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
