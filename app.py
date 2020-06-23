
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pd.read_csv('diabetes.csv')

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
data_tf = minmax.fit_transform(data)
data_tf = pd.DataFrame(data=data_tf)
data_tf.columns = data.columns

X_train,X_test,y_train,y_test = train_test_split(data_tf.drop('Outcome',axis=1),data_tf.Outcome,test_size=0.2,random_state = 1)

lr = LogisticRegression(C=8)
lr.fit(y=y_train,X=X_train)

y_pred = lr.predict(X_test)


print(metrics.classification_report(y_test, y_pred, labels=[0, 1]))