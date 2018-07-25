
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
np.set_printoptions(suppress=True)


# ### Function (merge_fill to merge data and and fill NaN values with mean, symmetric_mean_absolute_percentage_error to calculate SMAPE Score

# In[51]:


def merge_fill_test(df1, df2, column):
    x = pd.merge(df1, df2, on=['Year','Month','Product_ID','Country'])
    del x[column]
    diff = pd.concat([x, df1]).drop_duplicates(keep=False)
    resX = pd.concat([df2,diff],sort = False).drop_duplicates(keep=False)
    resX = resX.fillna(resX.mean())
    resX = pd.merge(df1, resX, on=['Year','Month','Product_ID','Country'])
    return resX

def merge_fill_train(df1, df2, column):
    x = pd.merge(df1, df2, on=['Year','Month','Product_ID','Country'])
    xi = pd.merge(df1, df2, on=['Year','Month','Product_ID','Country'])
    del xi[column]
    diff = pd.concat([xi, df1]).drop_duplicates(keep=False)
    resX = pd.concat([x,diff],sort = False).drop_duplicates(keep=False)
    resX = resX.fillna(resX.mean())
    return resX

def symmetric_mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_pred + y_true))) * 100


# In[3]:


train = pd.read_csv('yds_train2018.csv')
test = pd.read_csv('yds_test2018.csv')
promotional_expense = pd.read_csv('promotional_expense.csv')
holidays = pd.read_excel('holidays.xlsx')

train = train[(train['Sales'] > 0)]     #keeping only positive values


# In[4]:


promotional_expense.columns = ['Year', 'Month', 'Country', 'Product_ID', 'Expense_Price']
promotional_expense.head()


# In[5]:


salary_sum = train.join(train.groupby(['Year','Month','Product_ID','Country']).Sales.sum(),
                        on = ['Year','Month','Product_ID','Country'], rsuffix='_r')


# In[6]:


del salary_sum['S_No']
del salary_sum['Week']
del salary_sum['Merchant_ID']
del salary_sum['Sales']
salary_sum = salary_sum.drop_duplicates()


# In[7]:


res = merge_fill_train(salary_sum, promotional_expense, 'Expense_Price')
res.shape


# In[8]:


hd_train = res.iloc[:,0:6]
print(hd_train.head())


# In[9]:



for i in range(hd_train.shape[0]) :
    hd_train['holidays'] = 0
    
for i in range(holidays.shape[0]):
    a = holidays.iloc[i,0]
    datee = datetime.datetime.strptime(a, "%Y, %m, %d")
    for j in range(hd_train.shape[0]) :
        #print(str(hd.iloc[j,0])+'|'+str(datee.year)+'|'+str(hd.iloc[j,1])+'|'+ str(datee.month)+'|'+str(hd.iloc[j,2])+'|'+ str(holidays.iloc[i,1]))
        #print(datee.month)
        if ((str(hd_train.iloc[j,0]) == str(datee.year)) & (str(hd_train.iloc[j,1]) == str(datee.month)) & (str(hd_train.iloc[j,3]) == str(holidays.iloc[i,1]))):
            
            hd_train.iloc[j,6] = hd_train.iloc[j,6] +1
            #print('+1')     


# In[10]:


train_final = hd_train
train_final.head()


# In[11]:


X_test = test.iloc[:,1:5]


# In[12]:


test_exp = merge_fill_test(X_test,promotional_expense, 'Expense_Price')
test_exp.head()


# In[13]:


hd_test = test_exp
hd_test.head()


# In[14]:



for i in range(hd_test.shape[0]) :
    hd_test['holidays'] = 0
    
for i in range(holidays.shape[0]):
    a = holidays.iloc[i,0]
    datee = datetime.datetime.strptime(a, "%Y, %m, %d")
    for j in range(hd_test.shape[0]) :
        #print(str(hd.iloc[j,0])+'|'+str(datee.year)+'|'+str(hd.iloc[j,1])+'|'+ str(datee.month)+'|'+str(hd.iloc[j,2])+'|'+ str(holidays.iloc[i,1]))
        #print(datee.month)
        if ((str(hd_test.iloc[j,0]) == str(datee.year)) & (str(hd_test.iloc[j,1]) == str(datee.month)) & (str(hd_test.iloc[j,3]) == str(holidays.iloc[i,1]))):
            
            hd_test.iloc[j,5] = hd_test.iloc[j,5] +1
            #print('+1') 


# In[15]:


test_final = hd_test
test_final.head()


# Features combined in single dataset

# Rest of the Preprocessing :

# In[16]:


train_final.head()


# In[17]:


X = train_final.iloc[:,[0,1,2,3,5,6]].values
y = train_final.iloc[:, 4].values


# In[18]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 3] = le.fit_transform(X[:, 3])


# In[19]:


ohe_p = OneHotEncoder(categorical_features = [2,3])
X = ohe_p.fit_transform(X).toarray()
X = X[:,[1,2,3,4,6,7,8,9,10,11,12,13,14]] #to escape dummy variable trap


# In[20]:


X[0,:]


# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =0)


# ## Data is prepared

# ## Lets Evaluate Models

# In[22]:


from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# In[55]:


rfr =  RandomForestRegressor (n_estimators=300, max_depth=14,max_features= "log2")
rfr.fit(X_train,y_train)

ext = ExtraTreesRegressor(n_estimators=300, max_depth=14, max_features= "log2")
ext.fit(X_train,y_train)

dtr = DecisionTreeRegressor(max_depth=14, max_features= "log2")
dtr.fit(X_train,y_train)

print("Random Forest Model")
print("Train Score {}".format(rfr.score(X_train,y_train)))
print("Test Score {}".format(rfr.score(X_test,y_test)))
print("\n")
print("ExtraTreesRegressor Model")
print("Train Score {}".format(ext.score(X_train,y_train)))
print("Test Score {}".format(ext.score(X_test,y_test)))
print("\n")
print("DecisonTree Model")
print("Train Score {}".format(dtr.score(X_train,y_train)))
print("Test Score {}".format(dtr.score(X_test,y_test)))

print("\n")
print("\n")

print("Random Forest Model")
print("SMAPE Score {}".format(symmetric_mean_absolute_percentage_error(y_test, rfr.predict(X_test))))
print("ExtraTreesRegressor Model")
print("SMAPE Score {}".format(symmetric_mean_absolute_percentage_error(y_test, ext.predict(X_test))))
print("DecisonTree Model")
print("SMAPE Score {}".format(symmetric_mean_absolute_percentage_error(y_test, dtr.predict(X_test))))


# ### We can clearly see that the ExtraTreeRegressor is more accurate than the others, hence we will use this to forecast our sales.

# ### Now, Lets try with the original test set

# In[24]:


test_final.head()


# In[25]:


test_arr = test_final.values


# In[26]:


le_test = LabelEncoder()
test_arr[:, 3] = le_test.fit_transform(test_arr[:, 3])


# In[27]:


ohe_p_test = OneHotEncoder(categorical_features = [2,3])
test_arr = ohe_p_test.fit_transform(test_arr).toarray()
test_arr = test_arr[:,[1,2,3,4,6,7,8,9,10,11,12,13,14]] #to escape dummy variable trap


# In[28]:


test_arr[0,:]


# In[30]:


sales_predic_ext = ext.predict(test_arr)


# In[32]:


test.head()


# In[33]:


test_submission = test


# In[34]:


test_submission['Sales'] = sales_predic_ext


# In[35]:


test_submission.head()

#del test_submission['S_No']


# In[ ]:


test_submission.to_csv('yds_submission2018.csv', index = False)

