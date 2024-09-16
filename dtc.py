import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics")

hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    train_size=0.7,
                                                    random_state=2022)

dtc = DecisionTreeClassifier(random_state=2022,
                             max_depth=3)
dtc.fit(X_train, y_train)

plt.figure(figsize=(40,20))
plot_tree(dtc, feature_names=X.columns,
          class_names=['0','1'], fontsize=14)
plt.show()

y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = dtc.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))
################ Grid Search ###############
dtc = DecisionTreeClassifier(random_state=2022)
params = {'max_depth':[2,3,4,5,None],
          'min_samples_split':[2, 5 ,10],
          'min_samples_leaf':[1, 5, 10]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)
gcv = GridSearchCV(dtc, param_grid=params, verbose=3, 
                   cv=kfold, scoring='roc_auc')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(50,30))
plot_tree(best_model, feature_names=X.columns,
          class_names=['0','1'], fontsize=13)
plt.show()
# Feature Importance Plot
print(best_model.feature_importances_)
imps = best_model.feature_importances_
plt.barh(X.columns, imps)
plt.title("Feature Importances")
plt.show()
################## Bankruptcy #########################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy")
brupt = pd.read_csv("Bankruptcy.csv", index_col=0)
X = brupt.drop(['D', 'YR'], axis=1)
y = brupt['D']

dtc = DecisionTreeClassifier(random_state=2022)
params = {'max_depth':[2,3,4,5,None],
          'min_samples_split':[2, 5 ,10],
          'min_samples_leaf':[1, 5, 10]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)
gcv = GridSearchCV(dtc, param_grid=params, verbose=3, 
                   cv=kfold, scoring='roc_auc')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(30,25))
plot_tree(best_model, feature_names=X.columns,
          class_names=['0','1'], fontsize=17)
plt.show()

##################### Vehicle ##############################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Vehicle Silhouettes")
from sklearn.preprocessing import LabelEncoder
vehicle = pd.read_csv("Vehicle.csv")
X = vehicle.drop('Class', axis=1)
y = vehicle['Class']

le = LabelEncoder()
le_y = le.fit_transform(y)

dtc = DecisionTreeClassifier(random_state=2022)
params = {'max_depth':[2,3,4,5,None],
          'min_samples_split':[2, 5 ,10],
          'min_samples_leaf':[1, 5, 10]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)
gcv = GridSearchCV(dtc, param_grid=params, verbose=3, 
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, le_y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(30,25))
plot_tree(best_model, feature_names=X.columns,
          class_names=le.classes_ , fontsize=17)
plt.show()

print(best_model.feature_importances_)
imps = best_model.feature_importances_
plt.barh(X.columns, imps)
plt.show()

############### Heart Attack #####################
os.chdir(r"C:\Training\Kaggle\Datasets\Heart Attack")
heart = pd.read_csv("heart.csv")
X = heart.drop('output', axis=1)
y = heart['output']

dtc = DecisionTreeClassifier(random_state=2022)
params = {'max_depth':[2,3,4,5,None],
          'min_samples_split':[2, 5 ,10],
          'min_samples_leaf':[1, 5, 10]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=2022)
gcv = GridSearchCV(dtc, param_grid=params, verbose=3, 
                   cv=kfold, scoring='roc_auc')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(50,30))
plot_tree(best_model, feature_names=X.columns,
          class_names=['0','1'], fontsize=13)
plt.show()
# Feature Importance Plot
print(best_model.feature_importances_)
imps = best_model.feature_importances_
plt.barh(X.columns, imps)
plt.title("Feature Importances")
plt.show()