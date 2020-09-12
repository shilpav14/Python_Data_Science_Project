#!/usr/bin/env python
# coding: utf-8

# ## Importing necssary libraries

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading exising .csv file

# In[6]:


data = pd.read_csv(r"C:/Users/shilpa v gowda/Desktop/DataScienceProject/911.csv")
data


# In[ ]:


##Reading last 5 rows 
data.head()


# In[ ]:


##Displays first 3 rows
data.head(3)


# In[21]:


##Reading first 5 rows
data.tail()


# In[46]:


##Displays last 3 rows
data.tail(3)


# In[47]:


# Get the list of columns
data.columns


# In[22]:


#Displaying the no. of rows and columns
data.shape


# In[25]:


#Displays the information about the datatypes
#used in the dataset.
data.info()


# In[10]:


##Describe the DataFrame

data.describe()


# In[31]:


#Describe the DataFrame including categorical features

data.describe(include = "all")


# In[18]:


#Sorting using values

data.sort_values('zip')


# In[34]:


#Get the number of unique / distinct values in a column
data.nunique()


# ### Applying aggregate functions to columns

# In[39]:


data['zip'].mean()


# In[40]:


# Checking for number of null values
data.isnull().sum()


# ### Apply and Map functions
# 
# Apply and Map funcations enables this funcationality:
# Apply function can be used for both Series and DataFrame objects while Map function can be used only for Series.

# In[54]:


data['lat'] = data['lng'].apply(lambda x: x * 100)
data[['lat', 'lng']].head(3)


# In[21]:


data['lat'] = df['lng'].map(lambda x: x * 100)
d[['lat', 'lng']].head(3)

df.apply(lambda x: x * 100)


# ### Indexing, Sorting and Slicing
# 
# Used for picking up specific records by indexing or location.

# In[19]:


## Indexing is used for picking up a record based on index.
data.loc[2]


# In[12]:


#plot distribution plot for longitude and latitude column

data.hist()


# In[15]:


data.fillna(0)


# ### Data Visualization
# 
# """Along with data analysis, Pandas also supports visualization of data. Though packages like Matplotlib and Seaborn are most commonly used for plotting, Pandas provides inbuilt functions for visualization."""
# 

# In[22]:


data.plot()


# In[26]:


## Plotting the bar graph by taking 
## 10 rows and columns.
data = pd.DataFrame(np.random.rand(10,4),columns=['lat','lng','zip','e'])
data.plot.bar()


# In[30]:


data.plot.bar(stacked=True)


# In[31]:


##Plotting horizontally

data.plot.barh(stacked=True)


# ## hist()

# In[ ]:


data.plot.hist(bins=)


# ## box() 

# In[ ]:


data.plot.box()


# In[ ]:


data = pd.DataFrame(np.random.rand(10, 4), columns=['lat', 'lng','zip', 'e'])
data.plot.area()

data.plot.scatter(x='a', y='b')


# ## Seaborn Styles

# In[10]:


#set the current default style of seaborn
sns.set()
plt.show()


# ## Bar plot

# In[13]:


import random

x = np.arange(0,10)
y = np.array(random.sample(range(1,50),10))
x

y

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x,y)
ax.pie(x,y)

plt.bar(x,y)


# ## Creating new feature
# 
# ### countplot()

# In[ ]:


sns.countplot(data['lat'])


# In[ ]:


## Display unique titles

data[data['Reason']] == 'EMS'


# ## TimeStamp

# In[7]:


data['timeStamp'] = pd.to_datetime(data['timeStamp'])
type(data['timeStamp'][0])


# ## Graph
# 
# 

# In[ ]:


### Creating 3 columns

data['Hour'] = data['timeStamp'].appply(lambda t:t.hour) 
data['Month'] = data['timeStamp'].appply(lambda t:t.month) 
data['DayOfWeek'] = data['timeStamp'].appply(lambda t:t.dayofwek) 
data


# In[ ]:


d = {'0':'Mon',1:'Tues',2:'Wed',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}
data['DayOfWeek'] = df['DayOfWeek'].map(d)
data.head(1)


# In[ ]:


### No. of calls based on 'DaysOfWeek'

sns.countplot(data['DayOfWeek'] hue=data['lat'])


# ### groupy()

# In[ ]:


byMonth = data.groupby('Month')
byMonth = data.groupby('Month').agg('count')
byMonth


# In[ ]:


byMonth['lat'].plot()


# ### LM plot (OR) Regression plot

# In[ ]:


byMonth.reset_index(inplace = "True")

sns.lmplot(data=byMonth,x='Month',y='lat')


# In[ ]:


## Making an extra date column

data['Date'] = data['timeStamp'].apply(lambda x:x.date())

data['Date'].head(2)


# In[ ]:


### Checking in which daet we have this data

data[data['Reason'] == 'Fire'].groupby('Date').agg('count')['lat'].plot(label='Fire')
data[data['Reason'] == 'EMS'].groupby('Date').agg('count')['lat'].plot(label='EMS')
data[data['Reason'] == 'Traffic'].groupby('Date').agg('count')['lat'].plot(label='Traffic')
plt.legend()


# ### Creating new dataset

# In[11]:


#dataset
df2 = pd.DataFrame({'name':['bob','sam','ram','rita','suma','bob','sammy','monu','shilpav','sindhuv'],
                    'score':[5000,4000,3000,2000,3000,1000,5000,7000,9000,8000]})
df2


# In[12]:


#bob attended twice kahoot kwiz
df2['name'].value_counts()


# In[13]:


#in descending order
df2.groupby('name').agg('count').sort_values(by='score',ascending='False')

df2.groupby('name').agg('sum')


# In[14]:


df2.shape


# In[15]:


#about columns
df2.info()


# In[ ]:


df2.isnull().sum()
df2.isnull().sum().sum()


# In[16]:


df2.sum()


# In[17]:


#gives the top 5 names in the decreasing order(kind of optimizing the whole systems)
df2['name'].value_counts().head()


# In[ ]:


df2['score'].value_counts().head()


# In[21]:


df2['name'].nunique()
df2.hist()


# In[22]:


df2[(df2['score']>3000) & (df2['score']<5000)]['score'].hist(bins=20)

