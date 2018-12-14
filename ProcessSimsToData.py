
# coding: utf-8

# In[3]:


import numpy as np
from pylab import *
from random import randint,shuffle
import csv,os, glob


# In[4]:


# Get simulates traces
fnames_sim = glob.glob('/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/users/franinsu/testRun3/sims/*_Readout.txt')
shuffle(fnames_sim)
traces_sim_inner = np.empty((0,2048))
traces_sim_outer = np.empty((0,2048))
for fname in fnames_sim[:1]:
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


# In[ ]:


# Get moise traces
fnames_noise = glob.glob('/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/users/franinsu/testRun3/noise/*.npy')

traces_noise_inner = np.empty((0,2048))
traces_noise_outer = np.empty((0,2048))
for fname in fnames_noise:
    traces = np.load(fname)
    traces = traces * 1e6 # Convert from A to uA
    traces_noise_inner = np.vstack([traces_noise_inner,traces[:,0]])
    traces_noise_outer = np.vstack([traces_noise_outer,traces[:,1]])
    
print(traces_noise_inner.shape)
print(traces_noise_outer.shape)


# In[97]:


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


# In[112]:


DMCFiles = [fname[:-11]+"DMC.txt" for fname in fnames_sim]
# 6 energy, 8,9,10 position
DMCfltCols = [6,8,9,10]
#DMCfltCols = [2,6,8,9,10,11,12,13,15,16,17,18]
otherFeatures = np.empty((0,4))
i=0
while(len(otherFeatures)<N):
    file=DMCFiles[i]
    with open(file, 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')
        headers = next(reader, None)
        next(reader, None)
        for row in reader:
            if row[5]=="G4CMPDriftElectron" or row[5]=="G4CMPDriftHole":
                fltrow = [float(row[i]) for i in DMCfltCols]
                otherFeatures = np.vstack([otherFeatures,fltrow])
    i+=1
#         filHeaders = [headers[i] for i in DMCstrCols]
#         filHeaders += [headers[i] for i in DMCfltCols] 
#         print(filHeaders)
otherFeatures = otherFeatures[:N]


# In[113]:


len(otherFeatures)


# In[142]:


plt.hist(otherFeatures[:,0],bins=np.arange(0,28,0.4))
plt.title("Distribution of Energies")
plt.show()
plt.figure()
plt.hist(start_bins,bins=np.arange(min(start_bins[:,0]),max(start_bins[:,0]),100))
plt.title("Distribution of StartTimes")
plt.show()


# In[140]:


np.savez("/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/users/franinsu/testRun3/Data",
         traces_comb_inner=traces_comb_inner, 
         traces_comb_outer=traces_comb_outer,
         start_bins=start_bins,
         otherFeatures=otherFeatures
    )


# In[141]:


np.savez("/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/users/franinsu/testRun3/Traces",
         traces_sim_inner=traces_sim_inner, 
         traces_sim_outer=traces_sim_outer
    )

