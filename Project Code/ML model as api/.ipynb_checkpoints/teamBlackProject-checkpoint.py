#!/usr/bin/env python
# coding: utf-8

# # <p style="text-align: center;"> <b> <img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Flogodix.com%2Flogo%2F1926702.png&f=1&nofb=1&ipt=eaea19530bb66c5d3a8d88d7ad0d15d5a530f60bcf6c988614bb81afadfd0d0c&ipo=images" alt="ITI Logo" width="200"/> <br><br> ITI Summer Training Camp - Introduction to Machine Learning <br><br> Team Black - Final Project <br><br> <hr width=400> <b> </p>
# ## <p style="text-align: center;"> <b> Insurance Company Claims - Analysis, Cleaning and Prediction <br><br> <hr width=800> </b> </p>

# # Predicting whether a person will default on their premium
# 
# Importing necessary libraries

# In[1]:


# Data Wrangling 
import numpy as np
import pandas as pd 

# Data Visualisation 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron 
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


# In[2]:


train_data = pd.read_csv('finalProject/train.csv')
test_data = pd.read_csv('finalProject/test.csv')
combine = [train_data, test_data]


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[5]:


train_data.describe()


# In[6]:


train_data.describe(percentiles = [.08, .07, .06])


# **Inference**
# 
# * 93% of the people have paid their premiums. 
# * The age of people is very varied between 21 and 103

# In[7]:


plt.figure(figsize = (15, 6))
sns.heatmap(train_data.corr(numeric_only=True), annot = True)


# ## Data Wrangling 

# In[8]:


train_data.isnull().sum()


# In[9]:


test_data.isnull().sum()


# In[10]:


for dataset in combine: 
    dataset['age'] = dataset['age_in_days']//365
    dataset.drop(['age_in_days'], axis = 1, inplace = True)
train_data.head()


# In[11]:


train_data[['sourcing_channel', 'target']].groupby('sourcing_channel', as_index = False).mean()


# ### Application Under-writing Score

# We might need to make income groups to understand the relations better 

# In[12]:


train_data['IncomeBands'] = pd.cut(train_data['Income'], 5)
train_data[['IncomeBands', 'target']].groupby('IncomeBands', as_index = False).count()


# Let's standardize our data by using a standard scaler

# In[13]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(train_data[['Income']])
x_scaled = scaler.transform(train_data[['Income']])
x_scaled


# In[14]:


print(scaler.scale_)


# In[15]:


train_data['scaled_income'] = x_scaled
train_data.head()


# In[16]:


train_data['IncomeBands'] = pd.cut(train_data['scaled_income'], 5)
train_data[['IncomeBands', 'target']].groupby('IncomeBands', as_index = False).count()


# ### Let's try and deal with outlier values

# In[17]:


print(train_data['Income'].mean())
print(train_data['Income'].median())


# In[18]:


plt.hist(train_data['Income'])
plt.show()


# In[19]:


upper_bound = 0.95
lower_bound = 0.1
res = train_data['Income'].quantile([lower_bound, upper_bound])
print(res)


# So, we can collect all the values in this range and let go of the other ones. 

# In[20]:


true_index = (train_data['Income'] < res.loc[upper_bound])
true_index


# In[21]:


false_index = ~true_index


# In[22]:


no_outlier_data = train_data[true_index].copy()
no_outlier_data.head()


# In[23]:


no_outlier_data['IncomeBands'] = pd.cut(no_outlier_data['Income'], 5)
no_outlier_data[['IncomeBands', 'target']].groupby('IncomeBands', as_index = False).count()


# In[24]:


combine = [train_data, test_data]
for dataset in combine: 
    dataset.loc[ dataset['Income'] <= 23603.99, 'Income'] = 0
    dataset.loc[(dataset['Income'] > 23603.99) & (dataset['Income'] <= 109232.0), 'Income'] = 1
    dataset.loc[(dataset['Income'] > 109232.0) & (dataset['Income'] <= 194434.0), 'Income'] = 2
    dataset.loc[(dataset['Income'] > 194434.0) & (dataset['Income'] <= 279636.0), 'Income'] = 3
    dataset.loc[(dataset['Income'] > 279636.0) & (dataset['Income'] <= 364838.0), 'Income'] = 4
    dataset.loc[(dataset['Income'] > 364838.0) & (dataset['Income'] <= 450040.0), 'Income'] = 5
    dataset.loc[ dataset['Income'] > 450040.0, 'Income'] = 6
    
train_data.head()


# In[25]:


train_data.loc[false_index, 'Income'] = 5
train_data.head()


# In[26]:


train_data.drop(['IncomeBands', 'scaled_income'], axis = 1, inplace = True)
train_data.head()


# **Let's also make groups for Age**

# In[27]:


train_data['AgeBands'] = pd.cut(train_data['age'], 5)
train_data[['AgeBands', 'target']].groupby('AgeBands', as_index = False).count()


# In[28]:


for dataset in combine:    
    dataset.loc[ dataset['age'] <= 37.4, 'age'] = 0
    dataset.loc[(dataset['age'] > 37.4) & (dataset['age'] <= 53.8), 'age'] = 1
    dataset.loc[(dataset['age'] > 53.8) & (dataset['age'] <= 70.2), 'age'] = 2
    dataset.loc[(dataset['age'] > 70.2) & (dataset['age'] <= 86.6), 'age'] = 3
    dataset.loc[ dataset['age'] > 86.6, 'age'] = 4
train_data.drop('AgeBands', axis = 1, inplace = True)
combine = [train_data, test_data]
train_data.head()


# In[29]:


train_data[['age', 'application_underwriting_score']].groupby('age').mean()


# In[30]:


train_data['PremBand'] = pd.cut(train_data['no_of_premiums_paid'], 5)
train_data[['PremBand', 'application_underwriting_score']].groupby('PremBand').count()


# In[31]:


print(train_data['application_underwriting_score'].mean())
print(train_data['application_underwriting_score'].std())


# In[32]:


print(train_data[train_data['sourcing_channel'] == 'A']['application_underwriting_score'].std())
train_data[['sourcing_channel', 'target']].groupby('sourcing_channel', as_index = False).mean()


# In[33]:


# print(train_data[train_data['sourcing_channel'] == 'C']['application_underwriting_score'].std())
train_data[['sourcing_channel', 'application_underwriting_score']].groupby('sourcing_channel', as_index = False).mean()


# In[34]:


train_data[['residence_area_type', 'application_underwriting_score']].groupby('residence_area_type', as_index = False).mean()


# We can set the values of underwriting score on the basis of the sourcing channel

# In[35]:


train_data.dtypes


# In[36]:


combine = [train_data, test_data]
for dataset in combine: 
    mask1 = dataset['application_underwriting_score'].isnull()
    for source in ['A', 'B', 'C', 'D', 'E']:
        mask2 = (dataset['sourcing_channel'] == source)
        source_mean = dataset[dataset['sourcing_channel'] == source]['application_underwriting_score'].mean()
        dataset.loc[mask1 & mask2, 'application_underwriting_score'] = source_mean
train_data.head()


# In[37]:


dataset['application_underwriting_score'].isnull()


# In[38]:


test_data[test_data['Count_3-6_months_late'].isnull()]


#  Add  a new variable 'late premium' for late premiums

# In[39]:


sns.countplot(x = 'Count_3-6_months_late', data = train_data, hue = 'target')


# In[40]:


sns.countplot(x = 'Count_6-12_months_late', data = train_data, hue = 'target')


# In[41]:


combine = [train_data, test_data]
for dataset in combine: 
    dataset['late_premium'] = 0.0
train_data.head()


# In[42]:


combine = [train_data, test_data]
for dataset in combine:
        dataset.loc[(dataset['Count_3-6_months_late'].isnull()),  'late_premium'] = np.NaN
        dataset.loc[(dataset['Count_3-6_months_late'].notnull()), 'late_premium'] = dataset['Count_3-6_months_late'] + dataset['Count_6-12_months_late'] + dataset['Count_more_than_12_months_late']
train_data.head() 


# In[43]:


train_data['target'].corr(train_data['late_premium'])


# In[44]:


plt.figure(figsize = (15, 6))
sns.heatmap(test_data.corr(numeric_only=True), annot = True)


# In[45]:


sns.countplot(x = 'late_premium', data = train_data, hue = 'target')


# In[46]:


train_data[['late_premium', 'target']].groupby('late_premium').mean()


# In[47]:


# for dataset in [train_data]:
train_data.loc[(train_data['target'] == 0) & (train_data['late_premium'].isnull()),'late_premium'] = 7
train_data.loc[(train_data['target'] == 1) & (train_data['late_premium'].isnull()),'late_premium'] = 2
train_data.head()


# In[48]:


print(train_data.isnull().sum())
print('\n')
print(test_data.isnull().sum())


# ### Replacing the "late_premium" Value in the Test Data

# In[49]:


guess_prem = np.zeros(5)
for dataset in [test_data]:
    for i in range(1, 6):
        guess_df = dataset[(dataset['Income'] == i)]['late_premium'].dropna()

        # age_mean = guess_df.mean()
        # age_std = guess_df.std()
        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

        premium_guess = guess_df.median()
        guess_prem[i - 1] = int(premium_guess) 

    for j in range(1, 6):
        dataset.loc[(dataset.late_premium.isnull()) & (dataset.Income == j), 'late_premium'] = guess_prem[j - 1] + 1

    dataset['late_premium'] = dataset['late_premium'].astype(int)

test_data.head(10)


# In[50]:


train_data.drop(['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late'], axis = 1, inplace = True)
test_data.drop(['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late'], axis = 1, inplace = True)


# ## Conversion to Numerical Data

# In[51]:


# Converting Area Type and sourcing channel to Ordinal Variables
combine = [train_data, test_data]
for dataset in combine: 
    dataset['residence_area_type'] = dataset['residence_area_type'].map( {'Urban' : 1, 'Rural' : 0} )
    dataset['sourcing_channel'] = dataset['sourcing_channel'].map( {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4} )
train_data.head()


# In[52]:


train_data['application_underwriting_score'] = train_data['application_underwriting_score']/100
train_data.head()


# ### No. of Premiums Paid

# In[53]:


upper_bound = 0.95
res = train_data['no_of_premiums_paid'].quantile([upper_bound])
print(res)


# In[54]:


true_index = train_data['no_of_premiums_paid'] < res.loc[upper_bound]
false_index = ~true_index
true_index


# In[55]:


train_data['PremBand'] = pd.cut(train_data[true_index]['no_of_premiums_paid'], 4)
train_data[['PremBand', 'application_underwriting_score']].groupby('PremBand').count()


# ### Premium Column Conversion

# In[56]:


upper_bound = 0.90
res = train_data['premium'].quantile([upper_bound])
print(res)
true_index = train_data['premium'] < res.loc[upper_bound]
false_index = ~true_index
true_index


# In[57]:


train_data['PremBand'] = pd.cut(train_data[true_index]['premium'], 4)
train_data[['PremBand', 'target']].groupby('PremBand').count()


# In[58]:


test_data.head()


# In[59]:


combine = [train_data]
for dataset in combine: 
    dataset.loc[ dataset['premium'] <= 5925.0, 'premium'] = 0
    dataset.loc[(dataset['premium'] > 5925.00) & (dataset['premium'] <= 10650.0), 'premium'] = 1
    dataset.loc[(dataset['premium'] > 10650.0) & (dataset['premium'] <= 15375.0), 'premium'] = 2
    dataset.loc[(dataset['premium'] > 15375.0) & (dataset['premium'] <= 201200.0), 'premium'] = 3
    dataset.loc[ dataset['premium'] > 201200.0, 'premium'] = 4
train_data.drop('PremBand', axis = 1, inplace = True)
train_data.head()
combine = [train_data, test_data]


# Finally convert percentage premium paid

# In[60]:


train_data['PremBand'] = pd.cut(train_data['perc_premium_paid_by_cash_credit'], 4)
train_data[['PremBand', 'target']].groupby('PremBand').mean()


# In[61]:


combine = [train_data, test_data]
for dataset in combine: 
    dataset.loc[ dataset['perc_premium_paid_by_cash_credit'] <= 0.25, 'perc_premium_paid_by_cash_credit'] = 0
    dataset.loc[(dataset['perc_premium_paid_by_cash_credit'] > 0.25) & (dataset['perc_premium_paid_by_cash_credit'] <= 0.5), 'perc_premium_paid_by_cash_credit'] = 1
    dataset.loc[(dataset['perc_premium_paid_by_cash_credit'] > 0.5) & (dataset['perc_premium_paid_by_cash_credit'] <= 0.75), 'perc_premium_paid_by_cash_credit'] = 2
    dataset.loc[ dataset['perc_premium_paid_by_cash_credit'] > 0.75, 'perc_premium_paid_by_cash_credit'] = 3
train_data.drop('PremBand', axis = 1, inplace = True)
train_data.head()


# In[62]:


test_data.head()


# In[63]:


train_data[['perc_premium_paid_by_cash_credit', 'late_premium']] = train_data[['perc_premium_paid_by_cash_credit', 'late_premium']].astype(int)
test_data[['perc_premium_paid_by_cash_credit']] = test_data[['perc_premium_paid_by_cash_credit']].astype(int)
test_data.head()


# ## Building our models

# Let's make the data splits

# In[64]:


X_train = train_data.drop(['id', 'target', 'premium', 'perc_premium_paid_by_cash_credit'], axis = 1).copy()
y_train = train_data['target']
X_test = test_data.drop(['id', 'perc_premium_paid_by_cash_credit'], axis = 1).copy()
print(X_train.shape, y_train.shape, X_test.shape)


# In[65]:


X_train.head()


# In[66]:


X_test.head()


# ### Machine Learning Models (Supervised)

# In[67]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_log


# In[68]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian


# In[69]:


# KNN - K Nearest Neighbours

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn


# In[70]:


# Perceptron Algorithm

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
acc_perceptron


# In[71]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
acc_sgd


# In[72]:


# Decision Tree

decision_tree = DecisionTreeClassifier(max_depth = 7)
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree


# In[73]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators = 10)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest


# In[74]:


pred_values = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
pred_values.sort_values(by='Score', ascending=False)


# In[75]:


colors = sns.color_palette('pastel')[0:7]
plt.pie(pred_values['Score'], labels = pred_values['Model'], colors = colors, autopct='%.0f%%')
plt.title('Supervised Machine Learning Model Accuracy Pie Plot')
plt.legend(title="Machine Learning Model", loc="center left", bbox_to_anchor=(1.25, 0, 0.5, 1))
plt.show()


# In[79]:


model_names = ["KNN","LR","RF","NV","P","SGD","DT"]
plt.bar(x=model_names, height = pred_values['Score']-90, bottom = 90)
plt.title('Supervised Machine Learning Model Accuracy Bar Plot')
plt.show()


# In[80]:


prediction = pd.DataFrame({
        "id": test_data["id"],
        "target": y_pred
    })
prediction.to_csv('prediction.csv', index=False)


# In[ ]:


prediction.describe()

import pickle
filename = 'trained_model.sav'
pickle.dump(random_forest, open(filename, 'wb'))