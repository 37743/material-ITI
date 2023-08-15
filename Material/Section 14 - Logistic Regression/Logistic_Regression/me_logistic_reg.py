# -*- coding: utf-8 -*-
#"""Created on Thu Oct  4 15:47:52 2018  @author: Nour"""

#logistic regression
#it is linear classifier (separator is straight line)

#description
#company wants to know the people who concerns to buy to appear ads in their account
#in the social network using age and salary to perdict who will buy the product 

'''                     '' Importing the libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''                     ''Importing the dataset'''
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

'''      '' Splitting the dataset into the Training set and Test set'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)
#test_size=0.25 means 100 values for test

'''                       ''features scaling'''
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
'''           '' fitting logistic regression to the training set'''
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)    
classifier.fit(x_train,y_train)  #train data

'''              ''predicting the test set result'''
y_pred=classifier.predict(x_test) 
#y_test is the actual values & y_predict is the prediction values
print(y_pred)


'''                 ''making the confusion matrix'''
#to evaluate the performance of model
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
#No.of true predictions =65+24=89
#No.of false predictions=8+3=11

'''accuracy'''
acc=accuracy_score(y_test, y_pred)

"""## Computing the accuracy with k-Fold Cross Validation"""

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

'''             ''visualising the train set result'''
from matplotlib.colors import ListedColormap #to color  graph
x_set, y_set = x_train, y_train #x_set :to change only x_train by any vaiable when changing code
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
#min , max :to not squeezing points 
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

'''             ''visualising the test set result'''
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()