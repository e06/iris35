import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn import model_selection

from sklearn import ensemble
from sklearn import datasets

from sklearn.utils import shuffle

X = list('aabcccdddd')
print(X)
k_fold = model_selection.KFold(n_splits=10)
for train_indices, test_indices in k_fold.split(X):
  print('Train: %s | Test: %s' % (
    train_indices, test_indices))

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
y = y.astype(np.float32)

params = {
  'n_estimators': 200,
  'max_depth':4,
  'min_samples_split': 2, 
  'learning_rate':0.01,
  'loss': 'ls'
}

models = []
for i in [100, 200, 300, 400, 500, 600, 700]:
  params.update(n_estimators=i)
  models.append([str(i), ensemble.GradientBoostingRegressor(**params)])
for i in [100, 200, 300, 400, 500, 600, 700]:
  models.append(['g' + str(i), lgb.LGBMRegressor(
    objective='regression', num_leaves=31,learning_rate=0.05,n_estimators=i)])

seed = 13
scoring = 'neg_mean_squared_log_error'
results = []
names = []
for name, model in models:
  kfold = model_selection.KFold(n_splits=10, random_state=seed)
  cv_results = model_selection.cross_val_score(
    model, X, y, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f), %f %f" % (name,
    cv_results.mean(), cv_results.std(),
    cv_results.min(), cv_results.max())
  print(msg)
  
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
