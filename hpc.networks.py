# -*- coding: utf-8 -*-

import model as md
import time
import logbin_2020 as logbin
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import random

#%%

mList = [2,4,8,16,32,64,128] # m chosen for asthetic reason for loglog plot -> lines are equidistant

N = 100000

iteration_num = 1

#%%
"""
foldername = "test_dataset"

if not os.path.exists(foldername):
    os.makedirs(foldername)

for m in mList:
    average_k_list = []
    for i in range(iteration_num):
        t = time.time()
        test_model = md.BaraAlbert(N,m)
        test_model.Optimized_PrefAttach()
        elapsed_time = time.time() - t
        print(f"Test model took {elapsed_time} s to run")
        average_k_list.append(np.mean(test_model._degreeList))
    average_k = np.mean(average_k_list)
    average_k_sigma = np.std(average_k_list) / np.sqrt(iteration_num)
    data = dict()
    data["average_k"] = average_k
    data["average_k_sigma"] = average_k_sigma
    
    subfolder = f"{foldername}/{m}"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    with open(f"{subfolder}/{random.randint(0,100000)}.obj","wb") as f0:
        pickle.dump(data, f0)
"""
#%%

foldername = "vary_m_dataset_PrefAttach"

if not os.path.exists(foldername):
    os.makedirs(foldername)

for m in mList:
    degreeLists = md.repeat_method(md.BaraAlbert, N, m, iteration_num, "Optimized_PrefAttach")
    print(f"Finished m = {m}")
    data = dict()
    data["degreeLists"] = degreeLists
    
    subfolder = f"{foldername}/{m}"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    with open(f"{subfolder}/{random.randint(0,100000)}.obj","wb") as f0:
        pickle.dump(data, f0)
        
#%%

NList = [10,100,1000,10000,100000,1000000]

m = 2 # best fit -> fit is worse for larger m

foldername = "vary_N_dataset_PrefAttach"

if not os.path.exists(foldername):
    os.makedirs(foldername)

for N in NList:
    degreeLists = md.repeat_method(md.BaraAlbert, N, m, iteration_num, "Optimized_PrefAttach")
    print(f"Finished N = {N}")
    data = dict()
    data["degreeLists"] = degreeLists
    
    subfolder = f"{foldername}/{N}"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    with open(f"{subfolder}/{random.randint(0,100000)}.obj","wb") as f0:
        pickle.dump(data, f0)
        
#%%

mList = [2,4,8,16,32,64,128] # m chosen for asthetic reason for loglog plot -> lines are equidistant

N = 100000

foldername = "vary_m_dataset_RandAttach"

if not os.path.exists(foldername):
    os.makedirs(foldername)

for m in mList:
    degreeLists = md.repeat_method(md.BaraAlbert, N, m, iteration_num, "Optimized_RandAttach")
    print(f"Finished m = {m}")
    data = dict()
    data["degreeLists"] = degreeLists
    
    subfolder = f"{foldername}/{m}"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    with open(f"{subfolder}/{random.randint(0,100000)}.obj","wb") as f0:
        pickle.dump(data, f0)

#%%

NList = [10,100,1000,10000,100000,1000000]

m = 2 # best fit -> fit is worse for larger m

foldername = "vary_N_dataset_RandAttach"

if not os.path.exists(foldername):
    os.makedirs(foldername)

for N in NList:
    degreeLists = md.repeat_method(md.BaraAlbert, N, m, iteration_num, "Optimized_RandAttach")
    print(f"Finished N = {N}")
    data = dict()
    data["degreeLists"] = degreeLists
    
    subfolder = f"{foldername}/{N}"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    with open(f"{subfolder}/{random.randint(0,100000)}.obj","wb") as f0:
        pickle.dump(data, f0)
        
#%%

mList = [2,4,8,16,32,64,128] # m chosen for asthetic reason for loglog plot -> lines are equidistant

N = 100000

foldername = "vary_m_dataset_ExistAttach"

if not os.path.exists(foldername):
    os.makedirs(foldername)

for m in mList:
    degreeLists = md.repeat_method(md.BaraAlbert, N, m, iteration_num, "Optimized_ExistAttach")
    print(f"Finished m = {m}")
    data = dict()
    data["degreeLists"] = degreeLists
    
    subfolder = f"{foldername}/{m}"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    with open(f"{subfolder}/{random.randint(0,100000)}.obj","wb") as f0:
        pickle.dump(data, f0)
        
#%%

NList = [10,100,1000,10000,100000,1000000]

m = 2 # best fit -> fit is worse for larger m

foldername = "vary_N_dataset_ExistAttach"

if not os.path.exists(foldername):
    os.makedirs(foldername)

for N in NList:
    degreeLists = md.repeat_method(md.BaraAlbert, N, m, iteration_num, "Optimized_ExistAttach")
    print(f"Finished N = {N}")
    data = dict()
    data["degreeLists"] = degreeLists
    
    subfolder = f"{foldername}/{N}"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    with open(f"{subfolder}/{random.randint(0,100000)}.obj","wb") as f0:
        pickle.dump(data, f0)