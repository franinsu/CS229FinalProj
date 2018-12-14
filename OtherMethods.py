#!/usr/bin/env python
# coding: utf-8

# # Other Models

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pylab import *
import csv,os, glob
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random
import pickle


# In[2]:


home_dir = "/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/users/franinsu/testRun3/"
processed_dir = home_dir+"processed/"


# In[3]:


All_Traces = np.load(home_dir+"All_Traces.npz")
traces_sim_inner=All_Traces['traces_sim_inner']
traces_sim_outer=All_Traces['traces_sim_outer']
n_done_traces=All_Traces['n_done_traces']
traces_sim_inner=traces_sim_inner[:n_done_traces]
traces_sim_outer=traces_sim_outer[:n_done_traces]
print(traces_sim_inner.shape)
print(traces_sim_outer.shape)

All_OtherFeatures = np.load(home_dir+"All_OtherFeatures.npz")
otherFeatures=All_OtherFeatures['otherFeatures']
n_done_otherFeatures=All_OtherFeatures['n_done_otherFeatures']
otherFeatures=otherFeatures[:n_done_otherFeatures]
print(otherFeatures.shape)


# In[4]:


noiseData = np.load(home_dir+"SomeNoise.npz")
traces_noise_inner=noiseData['traces_noise_inner']
traces_noise_outer=noiseData['traces_noise_outer']


# In[5]:


# Combine traces
N = min([len(traces_sim_inner),len(traces_sim_outer),len(traces_noise_inner),len(traces_noise_outer)])

traces_comb_inner = np.empty((N,2048))
traces_comb_outer = np.empty((N,2048))
otherFeatures = otherFeatures[:N]
start_bins = np.empty((N,1))
random.seed(300)
for i in range(N):
    traces_sim_inner_padded = np.pad(traces_sim_inner[i],(2048,1024), 'constant')
    traces_sim_outer_padded = np.pad(traces_sim_outer[i],(2048,1024), 'constant')
    shift = random.randint(0, 2048+1024)
    traces_sim_inner_shifted = traces_sim_inner_padded[shift:shift+2048]
    traces_sim_outer_shifted = traces_sim_outer_padded[shift:shift+2048]
    traces_comb_inner[i] = traces_sim_inner_shifted+traces_noise_inner[i]
    traces_comb_outer[i] = traces_sim_outer_shifted+traces_noise_outer[i]
    start_bins[i] = 256-shift+2048

print(traces_comb_inner.shape)
print(traces_comb_outer.shape)


# In[6]:


N0 = otherFeatures[:,0].shape[0]
exc = np.zeros(N0,dtype=int)
f, m = 0.75,0.2
exc[:round(f*N0)] = otherFeatures[:round(f*N0),0]<=m
#idx = np.arange(N0)*inc
idx = np.where(1-exc)

otherFeatures = otherFeatures[idx]
traces_comb_inner = traces_comb_inner[idx]
traces_comb_outer = traces_comb_outer[idx]
start_bins = start_bins[idx]

print(traces_comb_inner.shape)
print(traces_comb_outer.shape)

plt.hist(otherFeatures[:,0],bins=np.arange(0,28,0.1))
plt.title("Distribution of Energies")
plt.show()
plt.figure()
plt.hist(start_bins,bins=np.arange(min(start_bins[:,0]),max(start_bins[:,0]),100))
plt.title("Distribution of StartTimes")
plt.show()


# In[7]:


X = np.hstack((traces_comb_inner,traces_comb_outer))
Y = np.hstack((start_bins, otherFeatures[:,0:1]))
print(std(X))
print(std(Y))
# Y = start_bins
np.random.seed(229)
d=0
downsample_factor = 1
closer = (start_bins >= 0-d) & (start_bins < 2048+d) & (otherFeatures[:,0:1]<=20)
Y = Y[np.squeeze(closer)]
X = X[np.squeeze(closer)]
#Y = np.squeeze(start_bins[closer])
print(X.shape)
print(Y.shape)

X = np.array([np.mean(x.reshape(-1, downsample_factor), 1) for x in X])
j=122
plt.title(Y[j,1])
plt.plot(X[j])
plt.axvline(Y[j,0]/downsample_factor,color='red')
#Ymin= np.min(Y)
#Ymax= np.max(Y)+1
Yrange = np.array([2048,20])# Ymax-Ymin
Y = Y/Yrange #(Y-Ymin)/Yrange
N=len(X)
m_train = round(.95*N)
m_dev =  round(.1*m_train)
idx = np.random.permutation(N)
X_train, X_test = X[idx[:m_train]], X[idx[m_train:]]
Y_train, Y_test = Y[idx[:m_train]], Y[idx[m_train:]]
#dev is subset of train but isn't actually used in training
X_dev, Y_dev  = X_train[-m_dev:], Y_train[-m_dev:]
print("Size train:",X_train.shape[0])
print("Size dev:",X_dev.shape[0])
print("Size test:", X_test.shape[0])


# In[3]:


XY=np.load(home_dir+"XY60k.npz")
X=XY["X"]
Y=XY["Y"]
Yrange = np.array([2048,20])# Ymax-Ymin
N=len(X)
m_train = round(.95*N)
m_dev =  round(10/95*m_train)
idx = np.random.permutation(N)
X_train, X_test = X[idx[:m_train]], X[idx[m_train:]]
Y_train, Y_test = Y[idx[:m_train]], Y[idx[m_train:]]
#dev is subset of train but isn't actually used in training
X_dev, Y_dev  = X_train[-m_dev:], Y_train[-m_dev:]
print("Size train:",X_train.shape[0]-X_dev.shape[0])
print("Size dev:",X_dev.shape[0])
print("Size test:", X_test.shape[0])


# ## Generalized Linear Models 

# ### Least Squares

# In[5]:


# Create feature column and estimator
X_training = X_train[:-m_dev]
Y_training = Y_train[:-m_dev]
m = X_training.shape[1]


# In[6]:


regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_training, Y_training[:,0])


# In[8]:


#Training
plt.figure()
Y_training_pred = regr.predict(X_training)
plt.title("MAE: %.2f"
      % np.mean(np.abs(Y_training[:,0]-Y_training_pred)*2048))
plt.scatter(Y_training[:,0]*2048,Y_training_pred*2048,s=3,c='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot(np.arange(2048),np.arange(2048),color='black')
plt.xlim([0,2048])
#Dev
plt.figure()
Y_dev_pred = regr.predict(X_dev)
plt.title("MAE: %.2f"
      % np.mean(np.abs(Y_dev[:,0]-Y_dev_pred)*2048))
plt.scatter(Y_dev[:,0]*2048,Y_dev_pred*2048,s=3,c='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot(np.arange(2048),np.arange(2048),color='black')
plt.xlim([0,2048])
plt.figure()
Y_test_pred = regr.predict(X_test)
plt.title("MAE: %.2f"
      % np.mean(np.abs(Y_test[:,0]-Y_test_pred)*2048))
plt.scatter(Y_test[:,0]*2048,Y_test_pred*2048,s=3,c='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot(np.arange(2048),np.arange(2048),color='black')
plt.xlim([0,2048])


# ### Ridge Regression

# In[32]:


regr = linear_model.Ridge(alpha=.5)
# Train the model using the training sets
regr.fit(X_training, Y_training[:,0])


# In[ ]:


#Training
plt.figure()
Y_training_pred = regr.predict(X_training)
plt.title("MAE: %.2f"
      % np.mean(np.abs(Y_training[:,0]-Y_training_pred)*2048))
plt.scatter(Y_training[:,0]*2048,Y_training_pred*2048,s=3,c='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot(np.arange(2048),np.arange(2048),color='black')
plt.xlim([0,2048])
#Dev
plt.figure()
Y_dev_pred = regr.predict(X_dev)
plt.title("MAE: %.2f"
      % np.mean(np.abs(Y_dev[:,0]-Y_dev_pred)*2048))
plt.scatter(Y_dev[:,0]*2048,Y_dev_pred*2048,s=3,c='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot(np.arange(2048),np.arange(2048),color='black')
plt.xlim([0,2048])


# ### Bayesian Ridge

# In[8]:


regr = linear_model.BayesianRidge()
regr.fit(X_training, Y_training[:,0])


# In[13]:


#Training
plt.figure()
Y_training_pred = regr.predict(X_training)
plt.title("MAE: %.2f"
      % np.mean(np.abs(Y_training[:,0]-Y_training_pred)*2048))
plt.scatter(Y_training[:,0]*2048,Y_training_pred*2048,s=3,c='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot(np.arange(2048),np.arange(2048),color='black')
plt.xlim([0,2048])
#Dev
plt.figure()
Y_dev_pred = regr.predict(X_dev)
plt.title("MAE: %.2f"
      % np.mean(np.abs(Y_dev[:,0]-Y_dev_pred)*2048))
plt.scatter(Y_dev[:,0]*2048,Y_dev_pred*2048,s=3,c='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot(np.arange(2048),np.arange(2048),color='black')
plt.xlim([0,2048])


# ## Random Forest Regressor

# In[14]:


from sklearn.ensemble import RandomForestRegressor


# In[15]:


regr = RandomForestRegressor()
regr.fit(X_training, Y_training[:,0])
#Training
plt.figure()
Y_training_pred = regr.predict(X_training)
plt.title("MAE: %.2f"
      % np.mean(np.abs(Y_training[:,0]-Y_training_pred)*2048))
plt.scatter(Y_training[:,0]*2048,Y_training_pred*2048,s=3,c='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot(np.arange(2048),np.arange(2048),color='black')
plt.xlim([0,2048])
#Dev
plt.figure()
Y_dev_pred = regr.predict(X_dev)
plt.title("MAE: %.2f"
      % np.mean(np.abs(Y_dev[:,0]-Y_dev_pred)*2048))
plt.scatter(Y_dev[:,0]*2048,Y_dev_pred*2048,s=3,c='g')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.plot(np.arange(2048),np.arange(2048),color='black')
plt.xlim([0,2048])


# ## Shallow FCNN

# In[6]:


def plot_logs(all_logs):
    plt.figure()
    y1= [log['time_metric'] for log in all_logs]
    y2=[log['val_time_metric'] for log in all_logs]
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title("Final MAE: Train {:7.2f}, Valid: {:7.2f}".format(y1[-1]*Yrange[0],y2[-1]*Yrange[0]))
    plt.plot(range(len(y1)), y1,
        label='Train Loss')
    plt.plot(range(len(y2)), y2,
        label = 'Val loss')
    plt.legend()
    plt.ylim([0, 
              max(
                  max(y1),
                  max(y2)
              )])


# In[4]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
    ])
    optimizer = tf.train.AdamOptimizer(0.0008) #instead of 0.001
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[time_metric])
    return model
model = build_model()
all_logs = []
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 2 == 0:
        print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
        all_logs.append(logs)
EPOCHS = 50
# Store training stats


# In[5]:


history = model.fit(X_train, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0, #0.1 should be 10/95
                    callbacks=[PrintDot()])


# In[7]:


plot_logs(all_logs)


# ## ResNet

# In[53]:


first_input = keras.layers.Input(shape=(2048,))
first_dense = keras.layers.Dense(1, )(first_input)
first2_dense = keras.layers.Dense(4, )(first_dense)

second_input = keras.layers.Input(shape=(2048,))
second_dense = keras.layers.Dense(1, )(second_input)
second2_dense = keras.layers.Dense(4, )(second_dense)

merge_one = keras.layers.concatenate([first_dense, second_dense])
dense_merge = keras.layers.Dense(2, )(merge_one)
out = keras.layers.Dense(1, )(dense_merge)

model = keras.models.Model(inputs=[first_input, second_input], outputs=out)
model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse',
               metrics=['mae'])


# In[17]:


# model.summary()
model.fit(x=[X_train[:,:,0],X_train[:,:,1]], y=Y_train[:,0], epochs=50, verbose=1)


# ## CNN Best

# In[26]:


XY=np.load(home_dir+"XY60k.npz")
X=XY["X"]
Y=XY["Y"]
Yrange = np.array([2048,20])# Ymax-Ymin
N=len(X)
X=X.reshape(-1,2048,2)
m_train = round(.95*N)
m_dev =  round(10/95*m_train)
idx = np.random.permutation(N)
X_train, X_test = X[idx[:m_train]], X[idx[m_train:]]
Y_train, Y_test = Y[idx[:m_train]], Y[idx[m_train:]]
#dev is subset of train but isn't actually used in training
X_dev, Y_dev  = X_train[-m_dev:], Y_train[-m_dev:]
print("Size train:",X_train.shape[0]-X_dev.shape[0])
print("Size dev:",X_dev.shape[0])
print("Size test:", X_test.shape[0])


# In[27]:


X_train.shape


# In[29]:


#reg = tf.contrib.layers.l2_regularizer(0.001)
reg = None
act = 'relu'
init = 'random_uniform'
#init = None
dropout_rate = 0.1
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Conv1D (kernel_size = (4), filters = 64, strides=(1),
                             input_shape=(X_train.shape[1:]), activation=act,
                             kernel_initializer=init,kernel_regularizer=reg), 
        keras.layers.Dropout(dropout_rate),
        keras.layers.MaxPooling1D(pool_size = (2), strides=(2)),
        keras.layers.Conv1D (kernel_size = (4), filters = 64, strides=(1), activation=act,
                             kernel_initializer=init,kernel_regularizer=reg),
        keras.layers.Dropout(dropout_rate),
        keras.layers.MaxPooling1D(pool_size = (2), strides=(2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation=act,
                           kernel_initializer=init,kernel_regularizer=reg),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(256, activation=act,
                           kernel_initializer=init,kernel_regularizer=reg),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(128, activation=act,
                           kernel_initializer=init,kernel_regularizer=reg),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(64, activation=act,
                           kernel_initializer=init,kernel_regularizer=reg),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation=act,
                           kernel_initializer=init,kernel_regularizer=reg)
    ])
    optimizer = tf.train.AdamOptimizer(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[time_metric,energy_metric])
    return model
model = build_model()
all_logs = []
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 2 == 0:
        print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
        all_logs.append(logs)
EPOCHS = 50
# Store training stats


# In[31]:


history = model.fit(X_train, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0, #0.1 should be 10/95
                    callbacks=[PrintDot()])


# In[ ]:




