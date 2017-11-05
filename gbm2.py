from sklearn import datasets
from sklearn import preprocessing
import pandas as pd
import lightgbm as lgb



#print(iris)
#print(type(iris))

#url = "iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#dataset = pandas.read_csv(url, names=names)

# https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset
#df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
#df['target'] = iris['target']
#print(df.head())

# https://github.com/Microsoft/LightGBM/issues/932
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

'''
param_algo = {
  'task': 'train',
  'boosting_type': 'gbdt',
  'objective': 'multiclass',
  'metric': {'multi_logloss'},
  'is_unbalance' : True,
  'num_class': 3,
  'num_leaves': 5,
  'min_data_in_leaf': 1,
  'zero_as_missing': False,
  'learning_rate': 0.1,
}
'''

param_algo = {
  'task': 'train',
  'boosting_type': 'gbdt',
  'objective': 'regression',
  'metric': {'multi_logloss'},
  'is_unbalance' : True,
  # 'num_class': 3,
  'num_leaves': 5,
  'min_data_in_leaf': 1,
  'zero_as_missing': False,
  'learning_rate': 0.1,
}

feature_name = ['a', 'b', 'c', 'd']
lgb_train = lgb.Dataset(
  iris_X, iris_y, feature_name=feature_name)
print(iris_y)
gbm = lgb.train(
  param_algo, lgb_train,
  num_boost_round=16,
  feature_name=feature_name)


print('---STEP-1---')
print(gbm)
print(dir(gbm))
print(gbm.predict([
  [5.1,3.5,1.4,0.2],
  [4.9,3.0,1.4,0.2],
  [7.0,3.2,4.7,1.4],
  [5.9,3.0,5.1,1.8]]))
gbm.save_model('g0')


# http://blog.csdn.net/niaolianjiulin/article/details/76584785

bst = lgb.cv(param_algo, lgb_train,
  num_boost_round=280, 
  nfold=3, early_stopping_rounds=20)
estimators = lgb.train(param_algo, lgb_train,
  feature_name=feature_name,
  num_boost_round=len(bst['multi_logloss-mean']))


ypred = estimators.predict(iris_X)
#print(ypred)
print(len(bst['multi_logloss-mean']))
