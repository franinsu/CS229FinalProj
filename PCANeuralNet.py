
# coding: utf-8

# In[6]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pylab import *
import csv,os, glob
from sklearn.decomposition import PCA,IncrementalPCA,KernelPCA
import random
import pickle


# In[7]:


home_dir = "/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/users/franinsu/testRun3/"
processed_dir = home_dir+"processed/"


# In[16]:


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


# In[5]:


# Get Noise Traces
# M = traces_sim_inner.shape[0]
M = 60000
noise_names = glob.glob(home_dir+'noise/*.npy')
noise_names
traces_noise_inner = np.empty((0,2048))
traces_noise_outer = np.empty((0,2048))
i = 0
noise_factor = 1
for noise_name in noise_names:
    traces = np.load(noise_name)
    traces = traces * noise_factor *1e6 # Convert from A to uA
    traces_noise_inner = np.vstack([traces_noise_inner,traces[:,0]])
    traces_noise_outer = np.vstack([traces_noise_outer,traces[:,1]]) 
    i+=1
    if i%10==0: print("loaded %d noise traces" %traces_noise_inner.shape[0])
    if traces_noise_inner.shape[0]>= M: break 
print("Noise Traces shape",traces_noise_inner.shape)
np.savez(home_dir+"SomeNoise60k",traces_noise_inner=traces_noise_inner,traces_noise_outer=traces_noise_outer)


# In[3]:


noiseData = np.load(home_dir+"SomeNoise60k.npz")
traces_noise_inner=noiseData['traces_noise_inner']
traces_noise_outer=noiseData['traces_noise_outer']


# In[16]:


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
np.savez(home_dir+"AllData60k",otherFeatures=otherFeatures,
         traces_comb_inner=traces_comb_inner,traces_comb_outer=traces_comb_outer,start_bins=start_bins)


# In[3]:


alldata = np.load(home_dir+"AllData60k.npz")
traces_comb_inner=alldata['traces_comb_inner']
traces_comb_outer=alldata['traces_comb_outer']
start_bins=alldata['start_bins']
otherFeatures=alldata['otherFeatures']


# In[4]:


N0 = otherFeatures[:,0].shape[0]
exc = np.zeros(N0,dtype=int)
#f, m = 0.75,0.2
f, m = 0.75,0.3
exc[:round(f*N0)] = otherFeatures[:round(f*N0),0]<=m
#idx = np.arange(N0)*inc
idx = np.where(1-exc)

otherFeatures = otherFeatures[idx]
traces_comb_inner = traces_comb_inner[idx]
traces_comb_outer = traces_comb_outer[idx]
start_bins = start_bins[idx]

print(traces_comb_inner.shape)
print(traces_comb_outer.shape)


# In[5]:


plt.figure()
plt.style.use('seaborn-talk')
plt.hist(otherFeatures[:,0],bins=np.arange(0,28,0.2))
plt.title("Distribution of Energies")
plt.xlabel("Energy (eV)")
plt.ylabel("Frequency")
plt.savefig("Edistr")
plt.show()
plt.figure()
plt.style.use('seaborn-talk')
plt.hist(start_bins,bins=np.arange(min(start_bins[:,0]),max(start_bins[:,0]),50))
plt.title("Distribution of StartTimes")
plt.xlabel("Start Time Bin")
plt.ylabel("Frequency")
plt.savefig("Tdistr")
plt.show()


# In[3]:


def plot_history_time(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title("Final MAE Valid: {:7.2f} [time]".format(np.array(history.history['val_time_metric'])[-1]*Yrange[0]))
    plt.plot(history.epoch, np.array(history.history['time_metric']),
    label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_time_metric']),
    label = 'Val loss')
    plt.legend()
    plt.ylim([0, 
              max(
                  max(np.array(history.history['time_metric'])),
                  max(np.array(history.history['val_time_metric']))
              )])
def plotPred(X,Y,E,Yrange):
    Y_pred_s = model.predict(X)
    Y_pred_s = Y_pred_s*Yrange
    Y_test_s = Y*Yrange
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(30,6))

    ax1.plot(Y_test_s,Y_pred_s,"g.",markersize=4)
    ax1.plot(np.arange(2048),np.arange(2048),color='black')
    ax1.ylim([0,max(Y_pred_s)])
    
    mae = np.abs(Y_test_s-Y_pred_s)
    
    #ax2.title("MAE vs StartTime")
    ax2.plot(Y_test_s,mae,"r.",markersize=2)
    ax2.axhline(np.mean(mae),color='black')
    ax2.ylim([0,max(Y_pred_s)])
    
    #ax3.title("MAE vs Energy")
    ax3.plot(E,mae,"b.",markersize=2)
    ax3.axhline(np.mean(mae),color='black')


# In[7]:


1-10/95


# In[8]:


X = np.hstack((traces_comb_inner,traces_comb_outer))
Y = np.hstack((start_bins, otherFeatures[:,0:1]))
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
plt.figure()
plt.title("%3.2f eV"%Y[j,1])
plt.plot(X[j])
plt.ylabel("Current Amplitude (μA)")
plt.xlabel("Start Time Bin")
plt.axvline(Y[j,0]/downsample_factor,color='red')
plt.savefig("samplePulse")
#Ymin= np.min(Y)
#Ymax= np.max(Y)+1
Yrange = np.array([2048,20])# Ymax-Ymin
Y = Y/Yrange #(Y-Ymin)/Yrange
N=len(X)
m_train = round(.95*N)
m_dev =  round(10/95*m_train)
idx = np.random.permutation(N)
X_train, X_test = X[idx[:m_train]], X[idx[m_train:]]
Y_train, Y_test = Y[idx[:m_train]], Y[idx[m_train:]]
#dev is subset of train but isn't actually used in training
X_dev, Y_dev  = X_train[-m_dev:], Y_train[-m_dev:]
print("Size train:",X_train.shape[0])
print("Size dev:",X_dev.shape[0])
print("Size test:", X_test.shape[0])


# In[9]:


np.savez(home_dir+"XY60k",X=X,Y=Y)


# In[8]:


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


# In[11]:


corr = np.corrcoef(X_train.T)


# In[12]:


plt.matshow(np.triu(corr).T,cmap='bwr', aspect='equal',vmin=-1, vmax=1)
plt.colorbar()
plt.gca().axes.get_xaxis().set_visible(False)
plt.savefig("corrMat.png")


# In[5]:


#np.random.seed(230)
# f = 0.33
#sample = X_train[np.random.choice(m_train-m_dev,round(f*(m_train-m_dev)),replace=False)]
# pca = PCA(n_components=1024)
pca = IncrementalPCA(n_components=1024, batch_size=5000)
pca.fit(X_train[:round(0.2*(m_train-m_dev))])


# In[6]:


kpca = KernelPCA(kernel="rbf", n_components=1024,batch_size=5000)
# kpca.fit(X_train[:5000])
kpca.fit(X_train[:round(0.2*(m_train-m_dev))])


# In[9]:


with open(home_dir+'pca_20p_rbt.pkl', 'wb') as f:
    pickle.dump(kpca, f, pickle.HIGHEST_PROTOCOL)


# In[9]:


with open(home_dir+'pca5k.pkl', 'rb') as f:
    pca = pickle.load(f)


# In[10]:


with open(home_dir+'pca_20p_rbt.pkl', 'rb') as f:
    pca = pickle.load(f)


# In[7]:


with open(home_dir+'kpca_5k_rbt.pkl', 'rb') as f:
    kpca = pickle.load(f)


# In[13]:


X_train_T = pca.transform(X_train)


# In[11]:


X_train_T = kpca.transform(X_train)


# In[14]:


'de'


# In[18]:


corr_T = np.corrcoef(X_train_T.T)


# In[13]:


plt.figure(figsize=(15, 12))
plt.style.use('seaborn-talk')
plt.matshow(np.triu(corr_T).T,cmap='bwr', aspect='equal',vmin=-1, vmax=1)
plt.colorbar()
plt.gca().axes.get_xaxis().set_visible(False)
plt.savefig("corrMatT.png")


# In[2]:


explained_variance_ratio = pca.explained_variance_ratio_
np.sum(explained_variance_ratio)


# In[12]:


explained_variance_ratio = pca.explained_variance_ratio_
plt.style.use('seaborn-talk')
#plt.title("Cumulative Explained Variance Ratio, Tot=%1.4f"%np.sum(explained_variance_ratio))
plt.plot(np.cumsum(explained_variance_ratio), label="Lin PCA")
plt.xlabel("Index of Principle Component")
plt.ylim([0,1])
explained_variance = np.var(X_train_T, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
plt.plot(np.cumsum(explained_variance_ratio),label="KPCA")
plt.legend()


# In[10]:


explained_variance = np.var(X_train_T, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
plt.title("Cumulative Explained Variance Ratio, Tot=%1.4f"%np.sum(explained_variance_ratio))
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel("Index of Principle Component")
plt.ylim([0,1])


# ## Final Runs

# In[ ]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.13),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.07),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
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
EPOCHS = 200
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0, #0.1 should be 10/95
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[45]:


EPOCHS = 30
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])


# In[49]:


plot_logs(all_logs)


# In[50]:


X_test_T = kpca.transform(X_test)
Y_pred_s = model.predict(X_test_T)
print(np.mean(np.abs(Y_pred_s-Y_test))*2048)


# In[36]:


np.mean(np.abs(Y_pred_s-Y_test))*2048


# In[42]:


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


# ## Experiments

# In[10]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.1),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.05),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.01),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.005),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.001),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
    ])
    optimizer = tf.train.AdamOptimizer(0.001) #instead of 0.001
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[time_metric])
    return model
model = build_model()
all_logs = []
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 100
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[9]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.12),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.06),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
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
    if epoch % 10 == 0: print(epoch,end='')
        print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
        all_logs.append(logs)
EPOCHS = 100
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[15]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.12),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.06),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
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
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 150
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[17]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.12),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.06),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
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
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 150
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[13]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.122),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.06),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
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
    if epoch % 5 == 0:
        print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
        all_logs.append(logs)
EPOCHS = 200
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[ ]:


## RUN THIS ##

activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.122),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.06),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
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
    if epoch % 10 == 0:
        print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
        all_logs.append(logs)
EPOCHS = 300
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[ ]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.124),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        #keras.layers.Dropout(0.06),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        #keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
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
    if epoch % 5 == 0:
        print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
        all_logs.append(logs)
EPOCHS = 250
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[ ]:


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 50
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[25]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.12),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.06),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
    ])
    optimizer = tf.train.AdamOptimizer(0.00085) #instead of 0.001
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[time_metric])
    return model
model = build_model()
all_logs = []
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 150
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[20]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.12),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.06),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
    ])
    optimizer = tf.train.AdamOptimizer(0.001) #instead of 0.001
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[time_metric])
    return model
model = build_model()
all_logs = []
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 150
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[24]:


# activation_func = tf.nn.relu
# #reg = tf.contrib.layers.l2_regularizer(0.001)
# init = None
# def time_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[0]
# def energy_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[1]
# def build_model():
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
#         keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
#         keras.layers.Dropout(0.12),
#         keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
#         keras.layers.Dropout(0.06),
#         keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
#         keras.layers.Dropout(0.02),
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
#         keras.layers.Dropout(0.01),
#         keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
#     ])
#     optimizer = tf.train.AdamOptimizer(0.0006) #instead of 0.001
#     #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#     model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=[time_metric])
#     return model
# model = build_model()
# all_logs = []
# class PrintDot(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs):
#     #if epoch % 10 == 0: print(epoch,end='')
#     print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
#     all_logs.append(logs)
# EPOCHS = 150
# # Store training stats
# history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
#                     validation_split=0.1, verbose=0,
#                     callbacks=[PrintDot()])
# plot_history_time(history)


# In[21]:


# activation_func = tf.nn.relu
# #reg = tf.contrib.layers.l2_regularizer(0.001)
# init = None
# def time_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[0]
# def energy_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[1]
# def build_model():
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
#         keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
#         keras.layers.Dropout(0.12),
#         keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
#         keras.layers.Dropout(0.06),
#         keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
#         keras.layers.Dropout(0.02),
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
#         keras.layers.Dropout(0.01),
#         keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
#         keras.layers.Dense(8, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
#     ])
#     optimizer = tf.train.AdamOptimizer(0.001) #instead of 0.001
#     #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#     model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=[time_metric])
#     return model
# model = build_model()
# all_logs = []
# class PrintDot(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs):
#     #if epoch % 10 == 0: print(epoch,end='')
#     print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
#     all_logs.append(logs)
# EPOCHS = 150
# # Store training stats
# history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
#                     validation_split=0.1, verbose=0,
#                     callbacks=[PrintDot()])
# plot_history_time(history)


# In[18]:


# activation_func = tf.nn.relu
# #reg = tf.contrib.layers.l2_regularizer(0.001)
# init = tf.contrib.layers.xavier_initializer()
# def time_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[0]
# def energy_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[1]
# def build_model():
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
#         keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
#         keras.layers.Dropout(0.12),
#         keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
#         keras.layers.Dropout(0.06),
#         keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
#         keras.layers.Dropout(0.02),
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
#         keras.layers.Dropout(0.01),
#         keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
#     ])
#     optimizer = tf.train.AdamOptimizer(0.0008) #instead of 0.001
#     #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#     model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=[time_metric])
#     return model
# model = build_model()
# all_logs = []
# class PrintDot(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs):
#     #if epoch % 10 == 0: print(epoch,end='')
#     print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
#     all_logs.append(logs)
# EPOCHS = 150
# # Store training stats
# history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
#                     validation_split=0.1, verbose=0,
#                     callbacks=[PrintDot()])
# plot_history_time(history)


# In[19]:


# activation_func = tf.nn.relu
# reg = tf.contrib.layers.l2_regularizer(0.00005)
# init = tf.contrib.layers.xavier_initializer()
# def time_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[0]
# def energy_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[1]
# def build_model():
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
#         keras.layers.Dense(512, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dropout(0.12),
#         keras.layers.Dense(256, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dropout(0.06),
#         keras.layers.Dense(128, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dropout(0.02),
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         keras.layers.Dropout(0.01),
#         keras.layers.Dense(32, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(16, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
#     ])
#     optimizer = tf.train.AdamOptimizer(0.0008) #instead of 0.001
#     #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#     model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=[time_metric])
#     return model
# model = build_model()
# all_logs = []
# class PrintDot(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs):
#     #if epoch % 10 == 0: print(epoch,end='')
#     print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
#     all_logs.append(logs)
# EPOCHS = 150
# # Store training stats
# history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
#                     validation_split=0.1, verbose=0,
#                     callbacks=[PrintDot()])
# plot_history_time(history)


# In[2]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = tf.contrib.layers.xavier_initializer()
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.12),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.06),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
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
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 150
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[38]:


# activation_func = tf.nn.relu
# #reg = tf.contrib.layers.l2_regularizer(0.001)
# init = None
# def time_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[0]
# def energy_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[1]
# def build_model():
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
#         keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
#         keras.layers.Dropout(0.125),
#         keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
#         keras.layers.Dropout(0.065),
#         keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
#         keras.layers.Dropout(0.02),
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
#         keras.layers.Dropout(0.01),
#         keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
#     ])
#     optimizer = tf.train.AdamOptimizer(0.0007) #instead of 0.001
#     #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#     model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=[time_metric])
#     return model
# model = build_model()
# all_logs = []
# class PrintDot(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs):
#     #if epoch % 10 == 0: print(epoch,end='')
#     print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
#     all_logs.append(logs)
# EPOCHS = 100
# # Store training stats
# history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
#                     validation_split=0.1, verbose=0,
#                     callbacks=[PrintDot()])
# plot_history_time(history)


# In[ ]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.14),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.07),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
    ])
    optimizer = tf.train.AdamOptimizer(0.0007) #instead of 0.001
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[time_metric])
    return model
model = build_model()
all_logs = []
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 100
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[ ]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.14),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.07),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
    ])
    optimizer = tf.train.AdamOptimizer(0.001) #instead of 0.001
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[time_metric])
    return model
model = build_model()
all_logs = []
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 100
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[ ]:


activation_func = tf.nn.relu
reg = tf.contrib.layers.l2_regularizer(0.001)
init = None
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
        keras.layers.Dropout(0.14),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.07),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
    ])
    optimizer = tf.train.AdamOptimizer(0.001) #instead of 0.001
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[time_metric])
    return model
model = build_model()
all_logs = []
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 100
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[ ]:


activation_func = tf.nn.relu
#reg = tf.contrib.layers.l2_regularizer(0.001)
init = tf.contrib.layers.xavier_initializer()
def time_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[0]
def energy_metric(y_true, y_pred):
    return np.abs(y_true-y_pred)[1]
def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
        keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.125),
        keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
        keras.layers.Dropout(0.065),
        keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
        keras.layers.Dropout(0.02),
        keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
        #keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
    ])
    optimizer = tf.train.AdamOptimizer(0.0007) #instead of 0.001
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[time_metric])
    return model
model = build_model()
all_logs = []
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    #if epoch % 10 == 0: print(epoch,end='')
    print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
    all_logs.append(logs)
EPOCHS = 100
# Store training stats
history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])
plot_history_time(history)


# In[39]:


# activation_func = tf.nn.relu
# #reg = tf.contrib.layers.l2_regularizer(0.001)
# init = None
# def time_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[0]
# def energy_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[1]
# def build_model():
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
#         keras.layers.Dense(512, activation=activation_func,kernel_initializer=init), 
#         keras.layers.Dropout(0.13),
#         keras.layers.Dense(256, activation=activation_func,kernel_initializer=init), 
#         keras.layers.Dropout(0.07),
#         keras.layers.Dense(128, activation=activation_func, kernel_initializer=init), 
#         keras.layers.Dropout(0.02),
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(0.01),
#         keras.layers.Dense(32, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(16, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(8, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(4, activation=activation_func, kernel_initializer=init),
#         #keras.layers.Dropout(dropout_rate),
#         keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
#     ])
#     optimizer = tf.train.AdamOptimizer(0.0007) #instead of 0.001
#     #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#     model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=[time_metric])
#     return model
# model = build_model()
# all_logs = []
# class PrintDot(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs):
#     if epoch % 5 == 0:
#         print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
#         all_logs.append(logs)
# EPOCHS = 100
# # Store training stats
# history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
#                     validation_split=0.1, verbose=0,
#                     callbacks=[PrintDot()])
# plot_history_time(history)


# In[ ]:


# activation_func = tf.nn.relu
# reg = keras.regularizers.l2(0.00007)
# init = tf.contrib.layers.xavier_initializer()
# dropout_rate = 0.5
# def time_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[0]
# def energy_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[1]
# def build_model():
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
#         keras.layers.Dense(512, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dense(512, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dense(256, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dense(256, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dense(128, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dense(128, activation=activation_func, kernel_initializer=init,kernel_regularizer=reg), 
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         keras.layers.Dense(32, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         keras.layers.Dense(16, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
#     ])
#     optimizer = tf.train.AdamOptimizer(0.001) #instead of 0.001
#     #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#     model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=[time_metric])
#     return model
# model = build_model()
# all_logs = []
# class PrintDot(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs):
#     #if epoch % 10 == 0: print(epoch,end='')
#     print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
#     all_logs.append(logs)
# EPOCHS = 100
# # Store training stats
# history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
#                     validation_split=0.1, verbose=0,
#                     callbacks=[PrintDot()])
# plot_history_time(history)


# In[29]:


# activation_func = tf.nn.relu
# reg = keras.regularizers.l2(0.00008)
# init = tf.contrib.layers.xavier_initializer()
# dropout_rate = 0.5
# def time_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[0]
# def energy_metric(y_true, y_pred):
#     return np.abs(y_true-y_pred)[1]
# def build_model():
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(X_train_T.shape[1],)),
#         keras.layers.Dense(512, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dense(512, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dense(256, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dense(256, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dense(128, activation=activation_func,kernel_initializer=init, kernel_regularizer=reg), 
#         keras.layers.Dense(128, activation=activation_func, kernel_initializer=init,kernel_regularizer=reg), 
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         keras.layers.Dense(64, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         keras.layers.Dense(32, activation=activation_func, kernel_initializer=init, kernel_regularizer=reg),
#         keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer=init)
#     ])
#     optimizer = tf.train.AdamOptimizer(0.001) #instead of 0.001
#     #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

#     model.compile(loss='mse',
#                 optimizer=optimizer,
#                 metrics=[time_metric])
#     return model
# model = build_model()
# all_logs = []
# class PrintDot(keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs):
#     #if epoch % 10 == 0: print(epoch,end='')
#     print("%d: %f, %f" %(epoch,logs['time_metric'],logs['val_time_metric']))
#     all_logs.append(logs)
# EPOCHS = 100
# # Store training stats
# history = model.fit(X_train_T, Y_train[:,0], epochs=EPOCHS,
#                     validation_split=0.1, verbose=0,
#                     callbacks=[PrintDot()])
# plot_history_time(history)

