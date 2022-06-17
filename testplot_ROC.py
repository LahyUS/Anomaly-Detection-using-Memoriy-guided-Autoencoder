import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# import dataset from CSV file on Github
url = "https://raw.githubusercontent.com/Statology/Python-Guides/main/default.csv"
data = pd.read_csv(url)

# define the predictor variables and the response variable
X = data[['student', 'balance', 'income']]
y = data['default']

# split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# instantiate the model
log_regression = LogisticRegression()

# fit the model using the training data
log_regression.fit(X_train, y_train)

# define metrics
y_pred_proba = log_regression.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# create ROC curve

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='ROC curve (AUC = %0.4f)' % auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--', label='random predict')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


plt.plot([0, 1], [1, 0], color='black', linewidth=1.5, linestyle='dashed')
plt.legend(loc='lower right')
#plt.plot(fpr, tpr, label="AUC="+str(auc))
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')
# plt.legend(loc=4)

plt.savefig('./exp/ped2/pred/ROC.png')
plt.show()
