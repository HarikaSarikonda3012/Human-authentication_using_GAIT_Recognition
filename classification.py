# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#/kaggle/input/pima-indians-diabetes-database/diabetes.csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix,plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('results/mukeshj.csv')
print(df.head())
print(df.shape)
print(df.info())
y = df['label']
x = df.drop('label',axis=1)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30,random_state=42)
dt = DecisionTreeClassifier(criterion='entropy')

dt_model = dt.fit(xtrain,ytrain)

ypred_train = dt_model.predict(xtrain)

ypred_test = dt_model.predict(xtest)
plot_tree(dt_model,max_depth=6)
plt.show()
print("The accuracy of train data is: ",accuracy_score(ytrain,ypred_train))
print(plot_confusion_matrix(estimator=dt,X=xtrain,y_true=ytrain))
print(plt.show())
print("The accuracy of test data is: ",accuracy_score(ytest,ypred_test))
print(plot_confusion_matrix(estimator=dt,X=xtest,y_true=ytest))
print(plt.show())
'''
dt_model.feature_importances_

importance=pd.DataFrame(data=xtrain.columns,columns=['Features'])
print(importance)
importance['Feature importances']=dt_model.feature_importances_
print(importance)
imp = importance.sort_values(by='Feature importances',ascending=False)
print(imp)
sns.barplot(x='Feature importances',y='Features',data=imp)
print(plt.show())'''
