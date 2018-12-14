
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from pylab import *
import csv,os, glob


# In[ ]:


home_dir = "/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/users/franinsu/testRun3/"
sims_dir = home_dir+"sims/"
processed_dir = home_dir+"processed/"

trace_names_sim = glob.glob(sims_dir+'*_Readout.txt')
shuffle(trace_names_sim)
dmc_names_sim = [trace_name[:-11]+"DMC.txt" for trace_name in trace_names_sim]
def processReadout(fname):
    #fname is the full filename of the trace
    traces_sim_inner = np.empty((0,2048))
    traces_sim_outer = np.empty((0,2048))
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
    
    return traces_sim_inner,traces_sim_outer
    
def processDMC(fname):
    #fname is the full filename of the DMC file
    # 6 energy, 8,9,10 position
    DMCfltCols = [6,8,9,10]
    #DMCfltCols = [2,6,8,9,10,11,12,13,15,16,17,18]
    otherFeatures = np.empty((0,4))
    with open(fname, 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')
        headers = next(reader, None)
        for row in reader:
            if row[5]=="G4CMPDriftElectron" or row[5]=="G4CMPDriftHole":
                fltrow = [float(row[i]) for i in DMCfltCols]
                otherFeatures = np.vstack([otherFeatures,fltrow])
    return otherFeatures
    
processed_traces_names=glob.glob(processed_dir+'*_Trace.npz')
proc_trace_names_cut = [trace_name[83:-9] for trace_name in processed_traces_names]
trace_names_cut = [trace_name[78:-11] for trace_name in trace_names_sim]
todo_traceFname = list(set(trace_names_cut)-set(proc_trace_names_cut))

i=0
for name in todo_traceFname:
    trace_name=sims_dir+name+"Readout.txt"
    traces_sim_inner,traces_sim_outer = processReadout(trace_name)
    np.savez(processed_dir+name+"Trace", traces_sim_inner=traces_sim_inner,traces_sim_outer=traces_sim_outer)
    i+=1
    if i%10 == 0:
        print("Processed %d Trace files" %i)
        
processed_dmc_names=glob.glob(processed_dir+'*_OtherFeatures.npz')
proc_dmc_names_cut = [dmc_name[83:-17] for dmc_name in processed_dmc_names]
dmc_names_cut = [dmc_name[78:-7] for dmc_name in dmc_names_sim]
todo_dmcFname = list(set(dmc_names_cut)-set(proc_dmc_names_cut))

i=0
for name in todo_dmcFname:
    dmc_name=sims_dir+name+"DMC.txt"
    otherFeatures = processDMC(dmc_name)
    np.savez(processed_dir+name+"OtherFeatures", otherFeatures=otherFeatures)
    i+=1
    if i%10 == 0:
        print("Processed %d DMCs" %i)
#     DMC_name = trace_name[:-11]+"DMC.txt"
#     if(~(name+"Trace.npz" in processed_traces_names)):
#         traces_sim_inner,traces_sim_outer = processReadout(trace_name)
#         np.savez(processed_dir+name+"Trace", traces_sim_inner=traces_sim_inner,traces_sim_outer=traces_sim_outer)
#         i += 1
#     if(~(name+"OtherFeatures.npz" in os.listdir(processed_dir))):
#         otherFeatures = processDMC(DMC_name)
#         np.savez(processed_dir+name+"OtherFeatures", otherFeatures=otherFeatures)
#         continue


# In[88]:


home_dir = "/gpfs/slac/staas/fs1/g/supercdms/tf/northwestern/users/franinsu/testRun3/"
processed_dir = home_dir+"processed/"


# In[89]:


size_temp = 1000
traces_sim_inner_temp = np.empty((size_temp,2048))
traces_sim_outer_temp = np.empty((size_temp,2048))
otherFeatures_temp = np.empty((size_temp,4))
done_trace_names_temp = ['' for i in range(size_temp)]
done_otherFeatures_names_temp = ['' for i in range(size_temp)]


# In[90]:


def saveTraces(traces_sim_inner_temp, traces_sim_outer_temp, done_trace_names_temp):
    print('start saving traces...',end=" ")
       
    All_Traces = np.load(home_dir+"All_Traces.npz")

    done_trace_names=All_Traces['done_trace_names']
    n_done_trace_names=All_Traces['n_done_trace_names']
    traces_sim_inner=All_Traces['traces_sim_inner']
    traces_sim_outer=All_Traces['traces_sim_outer']
    n_done_traces=All_Traces['n_done_traces']
    
    size_done_trace_names_temp=len(done_trace_names_temp)
    if n_done_traces+size_temp <= alloc_size:
        traces_sim_inner[n_done_traces:n_done_traces+size_temp]=traces_sim_inner_temp
        traces_sim_outer[n_done_traces:n_done_traces+size_temp]=traces_sim_outer_temp
        done_trace_names[n_done_trace_names:n_done_trace_names+size_done_trace_names_temp]=done_trace_names_temp
        n_done_trace_names += size_done_trace_names_temp
        n_done_traces += size_temp
    else:
        #temporary 
        return -1
#     traces_sim_inner=np.vstack([traces_sim_inner,traces_sim_inner_temp])
#     traces_sim_outer=np.vstack([traces_sim_outer,traces_sim_outer_temp])
#     done_trace_names=np.hstack([done_trace_names,done_trace_names_temp])
    np.savez(home_dir+"All_Traces",
             done_trace_names=done_trace_names,
             n_done_trace_names=n_done_trace_names,
             traces_sim_inner=traces_sim_inner,
             traces_sim_outer=traces_sim_outer,
             n_done_traces=n_done_traces)
    print('done saving traces')
def saveOtherFeatures(otherFeatures_temp, done_otherFeatures_names_temp):
    print('start saving otherFeatures...',end=" ")
    All_OtherFeatures = np.load(home_dir+"All_OtherFeatures.npz")
    otherFeatures=All_OtherFeatures['otherFeatures']
    done_otherFeatures_names=All_OtherFeatures['done_otherFeatures_names']
    n_done_otherFeatures = All_OtherFeatures['n_done_otherFeatures']
    n_done_otherFeatures_names = All_OtherFeatures['n_done_otherFeatures_names']
    
    size_done_otherFeatures_names_temp=len(done_otherFeatures_names_temp)
    if n_done_otherFeatures+size_temp <= alloc_size:
        otherFeatures[n_done_otherFeatures:n_done_otherFeatures+size_temp]=otherFeatures_temp
        done_otherFeatures_names[n_done_otherFeatures_names:n_done_otherFeatures_names+size_done_otherFeatures_names_temp]=done_otherFeatures_names_temp
        n_done_otherFeatures_names += size_done_otherFeatures_names_temp
        n_done_otherFeatures += size_temp
    else:
        #temporary 
        return -1
    
#     otherFeatures = np.vstack([otherFeatures,otherFeatures_temp])
#     done_otherFeatures_names = np.hstack([done_otherFeatures_names,done_otherFeatures_names_temp])
    
    np.savez(home_dir+"All_OtherFeatures",
         done_otherFeatures_names=done_otherFeatures_names,
         n_done_otherFeatures_names=n_done_otherFeatures_names,
         otherFeatures=otherFeatures,
         n_done_otherFeatures=n_done_otherFeatures)
    print('done saving traces')
def freeTempTraces(traces_sim_inner_temp,traces_sim_outer_temp,size_temp):
    traces_sim_inner_temp = np.empty((size_temp,2048))
    traces_sim_outer_temp = np.empty((size_temp,2048))
def freeTempOtherFeatures(otherFeatures_temp,size_temp):
    otherFeatures_temp = np.empty((size_temp,4))
# def save(traces_sim_inner,traces_sim_inner_temp,
#          traces_sim_outer,traces_sim_outer_temp,
#          otherFeatures,otherFeatures_temp,
#          done_trace_names,done_otherFeatures_names):
#     print('start saving...',end=" ")
#     traces_sim_inner = np.vstack([traces_sim_inner,traces_sim_inner_temp])
#     traces_sim_outer = np.vstack([traces_sim_outer,traces_sim_outer_temp])
#     otherFeatures = np.vstack([otherFeatures,otherFeatures_temp])
#     np.savez(home_dir+"AllData",
#          done_trace_names=done_trace_names,
#          done_otherFeatures_names=done_otherFeatures_names,
#          traces_sim_inner=traces_sim_inner,
#          traces_sim_outer=traces_sim_outer,
#          otherFeatures=otherFeatures)
#     print('done saving')
# def load(data,done_trace_names,done_otherFeatures_names,traces_sim_inner,traces_sim_outer,otherFeatures):
#     data = np.load(home_dir+"AllData.npz")
#     done_trace_names=data['done_trace_names']
#     done_otherFeatures_names=data['done_otherFeatures_names']
#     traces_sim_inner=data['traces_sim_inner']
#     traces_sim_outer=data['traces_sim_outer']
#     otherFeatures=data['otherFeatures']
# def freeTemp(traces_sim_inner_temp,traces_sim_outer_temp,otherFeatures_temp,done_trace_names_temp,done_otherFeatures_temp):
#     traces_sim_inner_temp = np.empty((20,2048))
#     traces_sim_outer_temp = np.empty((20,2048))
#     otherFeatures_temp = np.empty((20,4))
#     done_trace_names_temp = np.empty((20,1))
#     done_otherFeatures_temp = np.empty((20,1))


# In[124]:


alloc_size = 100000


# In[125]:


# FREE ALL
n_done_traces = 0
n_done_otherFeatures = 0
n_done_trace_names = 0
n_done_otherFeatures_names = 0
done_trace_names = np.empty((alloc_size),dtype=object)
done_otherFeatures_names = np.empty((alloc_size),dtype=object)
traces_sim_inner = np.empty((alloc_size,2048))
traces_sim_outer = np.empty((alloc_size,2048))
otherFeatures = np.empty((alloc_size,4))
data = None
np.savez(home_dir+"All_Traces",
         done_trace_names=done_trace_names,
         n_done_trace_names=n_done_trace_names,
         traces_sim_inner=traces_sim_inner,
         traces_sim_outer=traces_sim_outer,
         n_done_traces=n_done_traces)
np.savez(home_dir+"All_OtherFeatures",
         done_otherFeatures_names=done_otherFeatures_names,
         n_done_otherFeatures_names=n_done_otherFeatures_names,
         otherFeatures=otherFeatures,
         n_done_otherFeatures=n_done_otherFeatures)


# In[126]:


All_Traces = np.load(home_dir+"All_Traces.npz")
done_trace_names=All_Traces['done_trace_names']
# traces_sim_inner=All_Traces['traces_sim_inner']
# traces_sim_outer=All_Traces['traces_sim_outer']

All_OtherFeatures = np.load(home_dir+"All_OtherFeatures.npz")
done_otherFeatures_names=All_OtherFeatures['done_otherFeatures_names']
# otherFeatures=All_OtherFeatures['otherFeatures']

otherFeatures_names = glob.glob(processed_dir+'*_OtherFeatures.npz')
# print("Total of %d trace files" %len(trace_names))
print("Total of %d dmc files" %len(otherFeatures_names))
otherFeatures_names = list(set(otherFeatures_names)-set(done_otherFeatures_names))
shuffle(otherFeatures_names)
trace_names = [otherFeatures_name[:-17]+"Trace.npz" for otherFeatures_name in otherFeatures_names]
print("Total of %d dmc files to save" %len(otherFeatures_names))


# In[127]:


size_temp = 10000
traces_sim_inner_temp = np.empty((size_temp,2048))
traces_sim_outer_temp = np.empty((size_temp,2048))
otherFeatures_temp = np.empty((size_temp,4))
done_trace_names_temp = ['' for i in range(size_temp)]
done_otherFeatures_names_temp = ['' for i in range(size_temp)]


# In[128]:


print_every_n = 10
freeTempTraces(traces_sim_inner_temp,traces_sim_outer_temp,size_temp)
done_trace_names_temp = ['' for i in range(size_temp)]
i=0
j=0
print("Doing traces:")
for trace_name in trace_names:
    tracesData = np.load(trace_name)
    m_i = tracesData['traces_sim_inner'].shape[0]
    
    otherFeaturesData = np.load(trace_name[:-9]+"OtherFeatures.npz")
    if m_i != otherFeaturesData['otherFeatures'].shape[0]: continue
    
    if (j+m_i)<=size_temp:
        traces_sim_inner_temp[j:j+m_i] = tracesData['traces_sim_inner']
        traces_sim_outer_temp[j:j+m_i] = tracesData['traces_sim_outer']
        j=j+m_i
    else:
        traces_sim_inner_temp[j:] = tracesData['traces_sim_inner'][:size_temp-j]
        traces_sim_outer_temp[j:] = tracesData['traces_sim_outer'][:size_temp-j]
        
        done_trace_names_temp = done_trace_names_temp[:i]
        r = saveTraces(traces_sim_inner_temp, traces_sim_outer_temp, done_trace_names_temp)
        if r==-1: break
        freeTempTraces(traces_sim_inner_temp,traces_sim_outer_temp,size_temp)
        done_trace_names_temp = ['' for i in range(size_temp)]
        
        traces_sim_inner_temp[:m_i-size_temp+j] = tracesData['traces_sim_inner'][:m_i-size_temp+j] 
        traces_sim_outer_temp[:m_i-size_temp+j]  = tracesData['traces_sim_outer'][:m_i-size_temp+j] 
        j = m_i-size_temp+j
        i = 0
    done_trace_names_temp[i] = trace_name
    i+=1 
    if i%print_every_n==0:
        print("[%d,%d]"%(i,j), end=" ")
print("Traces shape:", traces_sim_inner.shape)


# In[129]:


freeTempOtherFeatures(otherFeatures_temp,size_temp)
done_otherFeatures_names_temp = ['' for i in range(size_temp)]
otherFeatures_name = [trace_name[:-9]+"OtherFeatures.npz" for trace_name in trace_names]
i=0
j=0
print("Doing DMC:")
for otherFeatures_name in otherFeatures_names:
    done_otherFeatures_names_temp[i] = otherFeatures_name
    otherFeaturesData = np.load(otherFeatures_name)
    m_i = otherFeaturesData['otherFeatures'].shape[0]
    
    traceData = np.load(otherFeatures_name[:-17]+"Trace.npz")
    if m_i != traceData['traces_sim_inner'].shape[0]: continue
    
    if (j+m_i)<=size_temp:
        otherFeatures_temp[j:j+m_i] = otherFeaturesData['otherFeatures']
        j=j+m_i
    else:
        otherFeatures_temp[j:] = otherFeaturesData['otherFeatures'][:size_temp-j]
        
        done_otherFeatures_names_temp = done_otherFeatures_names_temp[:i]
        r = saveOtherFeatures(otherFeatures_temp, done_otherFeatures_names_temp)
        if r==-1: break
        freeTempOtherFeatures(otherFeatures_temp,size_temp)
        done_otherFeatures_names_temp = ['' for i in range(size_temp)]

        otherFeatures_temp[:m_i-size_temp+j] = otherFeaturesData['otherFeatures'][:m_i-size_temp+j] 
        j = m_i-size_temp+j
        i = 0
    otherFeaturesData = np.load(otherFeatures_name)
    i+=1
    if i%print_every_n==0:
        print("[%d,%d]"%(i,j), end=" ")
print("OtherFeatures shape:", otherFeatures.shape)


# In[132]:


# i = 5
# print(otherFeatures_names[i])
# otherFeaturesData = np.load(otherFeatures_names[i])
# print(otherFeaturesData['otherFeatures'].shape[0])
# print(trace_names[i])
# tracesData = np.load(trace_names[i])
# print(tracesData['traces_sim_inner'].shape[0])


# In[133]:


All_Traces = np.load(home_dir+"All_Traces.npz")
done_trace_names=All_Traces['done_trace_names']
n_done_trace_names=All_Traces['n_done_trace_names']
traces_sim_inner=All_Traces['traces_sim_inner']
traces_sim_outer=All_Traces['traces_sim_outer']
n_done_traces=All_Traces['n_done_traces']
# print(traces_sim_inner.shape)
# print(traces_sim_outer.shape)
print(n_done_traces)
# print(done_trace_names.shape)
print(n_done_trace_names)

All_OtherFeatures = np.load(home_dir+"All_OtherFeatures.npz")
done_otherFeatures_names=All_OtherFeatures['done_otherFeatures_names']
n_done_otherFeatures_names=All_OtherFeatures['n_done_otherFeatures_names']
otherFeatures=All_OtherFeatures['otherFeatures']
n_done_otherFeatures=All_OtherFeatures['n_done_otherFeatures']
# print(otherFeatures.shape)
print(n_done_otherFeatures)
# print(done_otherFeatures_names.shape)
print(n_done_otherFeatures_names)

