#!/usr/bin/env python
# coding: utf-8

# # Project: Reucurrent Neural Network
# - A project on weather predictin on time series data

# ### Step 1: Import libraries

# In[1]:


import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 2: Download dataset
# - Excute the cell below

# In[2]:


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


# ### Step 3: Read the data
# - Use Pandas [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) method to read **csv_path** (from step 2).
#     - Also **parse_dates=True** and **index_col=0**

# In[3]:


data = pd.read_csv(csv_path, parse_dates=True, index_col=0)
data.head()


# In[4]:


len(data)


# ### Step 4: Limit dataset
# - The dataset has metrics for every 10 minutes - we will limit it to only once per hour
#     - HINT: **data[5::6]** will start at 5 and step 6.
#         - **a[start:stop:step]** start through not past stop, by step

# In[5]:


data = data[5::6]


# In[6]:


len(data)


# In[7]:


data.head()


# ### Step 5: Investigate data
# - Call **corr()** on the data to see correlations
# - Inspect what columns are correlated and not

# In[8]:


data.corr()


# In[ ]:





# ### Step 6: Remove data
# - Potential some data could be transformed **'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'**
#     - We will ignorre it

# In[9]:


df = data.drop(['wv (m/s)', 'max. wv (m/s)', 'wd (deg)'], axis=1)


# In[ ]:





# ### Step 7: Add periodic time intervals
# - Temperature is correlated to the time of day - e.g. it is warmer at mid day than at mid night
# - Temperature is correlated to seasons (most places in the world) - e.g. it is warmer in summer than in winter
# - The datetime index is not easy for the model to interpret, hence we can transform it into sinus and cosinus curves based on day and year.
# - Do it like this
#     - Assign the dataframe index to a variable, say, **timestamp_s**
#     - Transform that by using **map(pd.Timestamp.timestamp)**
#     - Use the period **day =** $24\times 60 \times 60$ and **year =** $(365.2425)\times$**day**
#     - Make the following columns **'Day sin', 'Day cos', 'Year sin'**, and **'Year cos'** as follows:
#         - e.g. **df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))**

# In[10]:


timestamp_s = df.index
timestamp_s = timestamp_s.map(pd.Timestamp.timestamp)


# In[11]:


timestamp_s


# In[12]:


df.index


# In[13]:


day = 24 * 60 * 60
year = (365.2425) * day


# In[14]:


df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))


# In[15]:


df.corr()


# ### Step 8: Splitting data
# 
# #### About splitting
# If you want to build a solid model you have to follow that specific protocol of splitting your data into three sets: One for training, one for validation and one for final evaluation, which is the test set.
# 
# The idea is that you train on your training data and tune your model with the results of metrics (accuracy, loss etc) that you get from your validation set.
# 
# Your model doesn't "see" your validation set and isn't in any way trained on it, but you as the architect and master of the hyperparameters tune the model according to this data. Therefore it indirectly influences your model because it directly influences your design decisions. You nudge your model to work well with the validation data and that can possibly bring in a tilt.
# 
# #### What to do?
# - Use the length of data and split it into
#     - 70% for training
#     - 20% for validation
#     - 10% for testing set

# In[16]:


nb_elts = len(df)
train_df = df[:int(nb_elts * .7)]
val_df = df[int(nb_elts * .7):int(nb_elts *.9)]
test_df = df[int(nb_elts * .9):]


# In[ ]:





# ### Step 9: Normalize data
# - Only normalize data based on training data
#     - Notice you should only normalize the training data - because validation and test data could affect the normalization
# - Get the mean and standard deviation of the data
#     - HINT: Use **.mean()** and **.std()** on the dataframe.
# - Noramlize the data as follows
#     - **train_df = (train_df - train_mean) / train_std** (assuming naming fits)
#     - HINT: The transformation of validation and test data is done similarly with **train_mean** and **train_std**.

# In[17]:


train_mean = train_df.mean()
train_std = train_df.std()


# In[18]:


train_df = (train_df - train_mean) / train_std 
val_df = (val_df - train_mean) / train_std 
test_df = (test_df - train_mean) / train_std 


# ### Step 10: Create datasets
# <img src='img/data_windowing.png' width=600 align='left'>

# - Make a function with **input_width** and **offset** - assume we always use **label_width=1**.
# - Call the function **create_dataset**, which takes arguments **df, input_width=24, offset=0, predict_column='T (degC)'**
#     - Let it create two empty lists **x** and **y**
#     - Convert the dataframe **df** to numpy and assign it to **data_x**
#     - Do the same for the **predict_column** but assign it to **data_y**
#     - Iterate over the range of starting from **input_width** to **len(data_x) - offset**
#         - Append to **x** with **data_x[i-input_width:i,:]**
#         - Append to **y** with **data_y[i + offset]**
#     - Convert **x** and **y** to numpy arrays
#         - HINT: Use **np.array(...)**
#     - Return the **x** and **y** (but reshape y with **reshape(-1, 1)**)
# - Apply the function on training, validation, and test data

# In[19]:


def create_dataset(df, input_width:int=24, offset:int=0, predict_column:str='T (degC)'):
  x = []
  y = []
  data_x = df.to_numpy()
  data_y = df[predict_column].to_numpy()

  for i in range(input_width, len(data_x) - offset ):
    x.append(data_x[i - input_width:i, :])
    y.append(data_y[i + offset])
  
  x = np.array(x)
  y = np.array(y)

  return x, y.reshape(-1, 1)


# In[20]:


train_ds = create_dataset(train_df)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)


# In[21]:


train_ds[0].shape


# ### Step 11: Create model
# - Create the following model
#     - **model = models.Sequential()**
#     - **model.add(layers.LSTM(32, return_sequences=True, input_shape=train_ds[0].shape[1:]))**
#     - **model.add(layers.Dense(units=1))**

# In[22]:


model = models.Sequential()
model.add(layers.LSTM(32, return_sequences=True, input_shape=train_ds[0].shape[1:]))
model.add(layers.Dense(units=1))


# In[ ]:





# ### Step 12: Train model
# - Compile and fit the model
# - Complie the model as follows
#     - **model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])**
# - Fit the model as follows
#     - **model.fit(x=train_ds[0], y=train_ds[1], validation_data=(val_ds[0], val_ds[1]), epochs=5)**

# In[23]:


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(x=train_ds[0], y=train_ds[1], validation_data=(val_ds[0], val_ds[1]), epochs=10)


# In[ ]:





# In[ ]:





# ### Step 13: Predict data
# - Apply the model on the test data
#     - HINT: Use **model.predict(x)**, where **x** is assigned to the test data.

# In[24]:


x, y = test_ds


# In[25]:


y_pred = model.predict(x)


# In[26]:


y_pred.shape


# ### Step 14: Plot the result
# - Plot a window of the data predicted together with the actual data.
# - One way:
#     - **fig, ax = plt.subplots()**
#     - **ax.plot(y[i:i+96*2,0], c='g')**
#     - **ax.plot(pred[i:i+96*2,-1,0], c='r')**
# - It will plot a window of 96 hours, where you can index with **i** (**i=150** as an example) and **y** is the real values and **pred** are the predicted values

# In[27]:


fig, ax = plt.subplots()
i = 200
ax.plot(y[i:i+96*2,0], c='g')
ax.plot(y_pred[i:i+96*2,-1,0], c='r')


# In[ ]:





# ### Step 15 (Optional): Calculate the correlation
# - Create a dataframe with real and predicted values.
# - Apply the **.corr()** method on the dataframe.

# In[28]:


df_c = pd.DataFrame({'real': y[:,0], 'pred': y_pred[:, -1,0]})

df_c.corr()


# In[ ]:





# In[ ]:




