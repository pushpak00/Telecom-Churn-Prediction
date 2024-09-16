import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import os
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

housing = pd.read_csv("Housing.csv")
dum_house = pd.get_dummies(housing, drop_first=True)
X = dum_house.drop('price', axis=1)
y = dum_house['price']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.7,
                                                    random_state=2022)

dtr = DecisionTreeRegressor(random_state=2022, max_depth=2)
dtr.fit(X_train, y_train)

plt.figure(figsize=(30,20))
plot_tree(dtr, feature_names=X.columns,fontsize=17, 
          filled=True)
plt.show()

y_pred = dtr.predict(X_test)
print(r2_score(y_test, y_pred))

################ Grid Search ###############
dtr = DecisionTreeRegressor(random_state=2022)
params = {'max_depth':[2,3,4,5,None],
          'min_samples_split':[2, 5 ,10],
          'min_samples_leaf':[1, 5, 10]}
kfold = KFold(n_splits=5, shuffle=True, 
                        random_state=2022)
gcv = GridSearchCV(dtr, param_grid=params, verbose=3, 
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(50,30))
plot_tree(best_model, feature_names=X.columns, fontsize=13)
plt.show()
# Feature Importance Plot
print(best_model.feature_importances_)
imps = best_model.feature_importances_
plt.barh(X.columns, imps)
plt.title("Feature Importances")
plt.show()

#################### Concrete ###########################
os.chdir(r"C:\Training\Academy\Statistics (Python)\Cases\Concrete Strength")
concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1)
y = concrete['Strength']

dtr = DecisionTreeRegressor(random_state=2022)
params = {'max_depth':[2,3,4,5,None],
          'min_samples_split':[2, 5 ,10],
          'min_samples_leaf':[1, 5, 10]}
kfold = KFold(n_splits=5, shuffle=True, 
                        random_state=2022)
gcv = GridSearchCV(dtr, param_grid=params, verbose=3, 
                   cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
plt.figure(figsize=(50,30))
plot_tree(best_model, feature_names=X.columns, fontsize=13)
plt.show()
# Feature Importance Plot
print(best_model.feature_importances_)
imps = best_model.feature_importances_
plt.barh(X.columns, imps)
plt.title("Feature Importances")
plt.show()

### Sorted Plot
i_sorted = np.argsort(-imps)
n_sorted = X.columns[i_sorted]
imp_sort = imps[i_sorted]
plt.barh(n_sorted, imp_sort)
plt.title("Sorted Feature Importances")
plt.show()



