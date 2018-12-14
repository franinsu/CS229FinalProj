
# coding: utf-8

# In[1]:


import glob
import numpy as np
from pylab import *
from random import randint


# In[2]:


# Get simulates traces
fnames_sim = glob.glob('/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/users/cstan4d/testRun1/sims/*_Readout.txt')

traces_sim_inner = np.empty((0,2048))
traces_sim_outer = np.empty((0,2048))
for fname in fnames_sim:
    traces_file = np.empty((0,2048))
    with open(fname) as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if i<3: continue # skip header lines
            line_list = line.split()
            #print(line_list[0:10])
            trace = line_list[10:]
            # Supersim traces are 4096 bins spaced 800ns apart. Noise traces are 2048 bins spaced 1.6us apart. So by taking every other bin from the supersim trace we get a match.                                  
            trace = np.array(trace[::2]).astype(np.float)
            trace = trace-trace[0]
            traces_file = np.vstack([traces_file,trace])
        traces_sim_inner = np.vstack([traces_sim_inner,traces_file[::2]])
        traces_sim_outer = np.vstack([traces_sim_outer,traces_file[1::2]])

print(traces_sim_inner.shape)
print(traces_sim_outer.shape)


# In[3]:


# Get moise traces
fnames_noise = glob.glob('/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/users/cstan4d/testRun1/noise/*.npy')

traces_noise_inner = np.empty((0,2048))
traces_noise_outer = np.empty((0,2048))
for fname in fnames_noise:
    traces = np.load(fname)
    traces = traces * 1e6 # Convert from A to uA
    traces_noise_inner = np.vstack([traces_noise_inner,traces[:,0]])
    traces_noise_outer = np.vstack([traces_noise_outer,traces[:,1]])
    
print(traces_noise_inner.shape)
print(traces_noise_outer.shape)


# In[38]:


# Combine traces
N = min([len(traces_sim_inner),len(traces_sim_outer),len(traces_noise_inner),len(traces_noise_outer)])

traces_comb_inner = np.empty((N,2048))
traces_comb_outer = np.empty((N,2048))
start_bins = np.empty((N,1))

for i in range(N):
    traces_sim_inner_padded = np.pad(traces_sim_inner[i],(2048,1024), 'constant')
    traces_sim_outer_padded = np.pad(traces_sim_outer[i],(2048,1024), 'constant')
    shift = randint(0, 2048+1024)
    traces_sim_inner_shifted = traces_sim_inner_padded[shift:shift+2048]
    traces_sim_outer_shifted = traces_sim_outer_padded[shift:shift+2048]
    traces_comb_inner[i] = traces_sim_inner_shifted+traces_noise_inner[i]
    traces_comb_outer[i] = traces_sim_outer_shifted+traces_noise_outer[i]
    start_bins[i] = 256-shift+2048

print(traces_comb_inner.shape)
print(traces_comb_outer.shape)


# In[42]:



np.savez('traces_and_times',traces_comb_inner=traces_comb_inner,traces_comb_outer=traces_comb_outer,start_bins=start_bins)


# In[41]:


# Plot some examples
fig,ax = subplots(4,4,figsize=(18,18))
for i in range(4):
    for j in range(4):
        ax[i,j].plot(traces_comb_inner[(i+1)*(j+1)])
        ax[i,j].plot(traces_comb_outer[(i+1)*(j+1)])
        ax[i,j].axvline(start_bins[(i+1)*(j+1)],color='r')

        #ax[i,j].set_xlim(250,260)


# In[120]:


# Plot smallest pulse (turns out they are still kind of big, so we may want to reduce the minimum energy in the simulation)
std_inner = np.std(traces_sim_inner,1)
std_outer = np.std(traces_sim_outer,1)
figure()
plot(traces_sim_inner[np.argmin(std_inner)]+traces_noise_inner[0])
figure()
plot(traces_sim_inner[np.argmin(std_outer)]+traces_noise_inner[0])

