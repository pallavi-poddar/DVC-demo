from sklearn import datasets
boston=datasets.load_boston()
xdata=boston.data
ydata=boston.target


from sklearn.model_selection import train_test_split

xtr,xts,ytr,yts=train_test_split(xdata,ydata,test_size=0.2)

#lieanr regression on training data
from sklearn.linear_model import LinearRegression
alg=LinearRegression()

#train the model
alg.fit(xtr,ytr)

#accuracy
accuracy=alg.score(xts,yts)
print(accuracy)

#acccuracies - cross validation
from sklearn import model_selection

accuracies=model_selection.cross_val_score(alg,xdata,ydata,cv=5)
print(accuracies)
print(accuracies.mean())

import numpy
alg.predict(numpy.array([0.17004,12.50,7.870,0,0.5240,6.0040,85.90,6.5921,5,311.0,15.20,386.71,17.10]).reshape(1,13))

from sklearn.externals import joblib
joblib.dump(alg,'model.pkl')
