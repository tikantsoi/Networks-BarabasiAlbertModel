#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 23:00:00 2022

@author: tikantsoi
"""

import model as md
import time
import logbin_2020 as logbin
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from scipy.optimize import curve_fit
import pickle
import os
import glob

plt.rcParams["figure.figsize"] = (7,5)
plt.rcParams["figure.dpi"] = 200
plt.rcParams.update({'font.size': 14})

#%%

def read_data(foldername,testing=False):
    combined_degreeLists = []
    for file in glob.glob(f"{foldername}/*"):
        with open(file,"rb") as f0:
            data = pickle.load(f0) 
            if testing==True:
                average_k = data["average_k"]
                average_k_sigma = data["average_k_sigma"]
            
                return average_k, average_k_sigma
            degreeLists = data["degreeLists"]
            combined_degreeLists.append(degreeLists[0])
        
    return combined_degreeLists
        
#%% check the model

# <k> = 2m for large t, 1 time

# qualitative discussion -> considered the two major errors -> check visually for small k

def continuous_PrefAttach_Eq(k,a,gamma): # try fit againcd 
    prob = a / k**gamma
    return prob

"""
mList = [4,8] # m chosen for asthetic reason for loglog plot -> lines are equidistant

N = 1000

iteration_num = 20

average_k_list = []

if not os.path.exists(foldername):
    os.makedirs(foldername)

for m in mList:
    for i in range(iteration_num):
        t = time.time()
        test_model = md.BaraAlbert(N,m)
        test_model.PrefAttach()
        elapsed_time = time.time() - t
        print(f"Test model took {elapsed_time} s to run")
        average_k_list.append(np.mean(test_model._degreeList))
    average_k = np.mean(average_k_list)
    average_k_sigma = np.std(average_k_list)
    data = dict()
    data["average_k"] = average_k
    data["average_k_sigma"] = average_k_sigma
    with open(f"{foldername}/{m}.obj","wb") as f0:
        pickle.dump(data, f0)
    print(f"Average k for m = {m}: {average_k} \pm {average_k_sigma}")
"""

foldername = "test_dataset"

mList = [2,4,8,16,32,64,128]

N = 100000

#%%

for m in mList:
    average_k, average_k_sigma = read_data(f"{foldername}/{m}",testing=True)
    print(f"Average k for m = {m}: {average_k} \pm {average_k_sigma}")
    
#%% define discrete master equation

def PrefAttach_Eq(m,k):
    prob = 2*m*(m+1) / (k*(k+1)*(k+2))
    return prob

def RandAttach_Eq(m,k):
    prob = (m/(m+1))**(k-m) / (1+m)
    return prob

def ExistAttach_Eq(m,k):
    prob = (3/2*m*(3*m+2)) / ((k+m+2)*(k+m+1)*(k+m))
    return prob

# proof

#%% Phase 1.3 vary m

# generate the models

"""
foldername = "vary_m_dataset_PrefAttach"

if not os.path.exists(foldername):
    os.makedirs(foldername)

for m in mList:
    degreeLists = md.repeat_method(md.BaraAlbert, N, m, iteration_num, "PrefAttach")
    print(f"Finished m = {m}")
    data = dict()
    data["degreeLists"] = degreeLists
    
    with open(f"{foldername}/{m}.obj","wb") as f0:
        pickle.dump(data, f0)
"""
    
#%%

def largestk1_PrefAttach(N,m):
    k1 = - 0.5 + np.sqrt(1+4*N*m*(m+1)) / 2
    return k1

def largestk1_RandAttach(N,m):
    k1 = np.log10((m/(m+1))**m/N) / np.log10(m/(m+1))
    return k1
    
def find_k1_distribution(degreeLists):
    average_k1 = 0
    k1List = []
    for degreeList in degreeLists:
        k1 = max(degreeList)
        k1List.append(k1)
        average_k1 += k1 / len(degreeLists)    
    standard_deviation = np.std(k1List)

    return average_k1, standard_deviation

def find_degree_distribution(degreeLists, scale, N, m, Type="PrefAttach", vary="m", color=""):
    y_list = [] 
    x_list = []
    padded_y_list = []
    padded_x_list = []
    average_x = 0
    average_y = 0
    
    # pad degreeList
    
    #MaxLen = len(max(degreeLists, key = len))
    for degreeList in degreeLists:
        #if len(degreeList) != MaxLen:
            #degreeList += [0] * (MaxLen - len(degreeList))
        x,y = logbin.logbin(degreeList, scale=scale) # same scale for all?, some m will have more data points
        x_list.append(x)
        y_list.append(y)
    """
    # pad x
    #MaxLen_x = len(max(x_list, key = len))
    MinLen_x = len(min(x_list, key = len))
    for x in x_list:
        if len(x) != MinLen_x:
            #x = np.pad(x, (0,MaxLen_x-len(x)))
            x = x[:MinLen_x]
        padded_x_list.append(x)
        average_x += x / len(degreeLists)
    standard_deviation_x = np.std(padded_x_list, axis=0)
    """
    # pad y
    MaxLen_y = len(max(y_list, key = len))
    #MinLen_y = len(min(y_list, key = len))
    for y in y_list:
        if len(y) != MaxLen_y:
            y = np.pad(y, (0,MaxLen_y-len(y)))
            #y = y[:MinLen_y]
        padded_y_list.append(y)
        average_y += y / len(degreeLists)
    standard_deviation_y = np.std(padded_y_list, axis=0)
    
    plt.yscale('log')
    plt.xscale('log')
    
    x = max(x_list, key= len)
    
    #x = np.arange(min(average_x),max(average_x))
    
    if vary == "m":
        plt.errorbar(x, average_y, yerr=standard_deviation_y/np.sqrt(len(degreeLists)), fmt="x", capsize=3,  color=color,label=f"{m}")
        if Type == "PrefAttach":
            plt.plot(x, PrefAttach_Eq(m,x), "--", color=color)
        if Type == "RandAttach":
            plt.plot(x, RandAttach_Eq(m,x), "--", color=color)
        if Type == "ExistAttach":
            plt.plot(x, ExistAttach_Eq(m,x), "--", color=color)
        
        plt.xlabel(r"$k$")
        plt.ylabel(r"$\widetilde{p}_{k}$")
        plt.legend(title=r"$m$", loc=3) 
        
    if vary == "N":
        plt.errorbar(x, average_y, yerr=(standard_deviation_y/np.sqrt(len(degreeLists))), fmt="x", capsize=3, label=f"$10^{str(N).count('0')}$")
        if Type == "PrefAttach":
            plt.plot(x, PrefAttach_Eq(m,x), "--", color="grey")
        if Type == "RandAttach":
            plt.plot(x, RandAttach_Eq(m,x), "--", color="grey")
        plt.xlabel(r"$k$")
        plt.ylabel(r"$\widetilde{p}_k$")
        plt.legend(title=r"$N$", loc=3)    
   # 
    observed_freq = np.array(padded_y_list) * N

    if Type == "PrefAttach":
        theoretical_freq = PrefAttach_Eq(m,x) * N
    if Type == "RandAttach":
        theoretical_freq = RandAttach_Eq(m,x) * N
    if Type == "ExistAttach":
        theoretical_freq = ExistAttach_Eq(m,x) * N

    return theoretical_freq, observed_freq

#%% Visualisation

foldername = "vary_m_dataset_PrefAttach"

colorList = ["tab:blue","tab:orange","tab:green","tab:red", "tab:purple", "tab:brown", "tab:pink"]

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    find_degree_distribution(degreeLists, 1.1, N, m, color=colorList[mList.index(m)])
    
# problem -> fit less well for larger k -> few events in the log bin

#%%

foldername = "vary_m_dataset_PrefAttach"

mList = [4]

colorList = ["tab:blue","tab:orange","tab:green","tab:red", "tab:purple", "tab:brown", "tab:pink"]

for m in mList:
    plt.figure(figsize=(6.4, 4.8))
    degreeLists = read_data(f"{foldername}/{m}")
    find_degree_distribution(degreeLists, 1.3, N, m, color=colorList[mList.index(m)])
    model = md.BaraAlbert(100000,4)
    model.Optimized_PrefAttach()
    x,y = logbin.logbin(model._degreeList,1)
    plt.plot(x,y,'x', zorder=-1, color='gray', alpha=0.6)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\widetilde{p}_{k}$")
    plt.legend(["Unbinned data","Logarithmic binning"])
    
    
#%%

# https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
# k is normally distributed -> normally distributed -> allow chi squared
# chi squared >> 1 -> poor model fit
# chi squared > 1 the fit has not fully captured the data
# chi squared < 1 -> overfitting
# best around 1

# http://physics.ucsc.edu/~drip/133/ch4.pdf
# chi squared a quantity commonly used to test whether any given data are well described by 
# some hypothesized function
# it considers a number of samples -> statistical consideration
# it uses frequencies

# https://hps.org/publicinformation/ate/q7763.html
# reduced chi squared considers the degree of freedom

# whereas ks test should be used on continuous data
# ks test normally only compares 2 samples with each other -> no statistical consideration
# it uses CDF (probability)

# https://www.astroml.org/book_figures/chapter4/fig_chi2_eval.html

foldername = "vary_m_dataset_PrefAttach"

def reduced_chisquared(theoretical_freq, observed_freq):
    mean = np.mean(observed_freq, axis=0)
    standard_deviation = np.std(observed_freq, axis=0)
    
    zList = []
    for i in range(len(theoretical_freq)):
        z = (theoretical_freq[i] - mean[i]) / standard_deviation[i]
        zList.append(z)
    chi2value = np.sum(np.array(zList) ** 2)
 
    reduced_chi2 = chi2value / (len(zList) - 1)
    dof = len(zList) - 1
        
    return reduced_chi2, dof

chisqauredList = []

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.25, N, m, color=colorList[mList.index(m)])
    reduced_chi2, dof = reduced_chisquared(theoretical_freq, observed_freq)
    chisqauredList.append(reduced_chi2)

    print(f"Reduced chi-squared value for {m}: {reduced_chi2}, dof: {dof}")
    
#%%

points_to_remove = 4

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.25, N, m, color=colorList[mList.index(m)])
    reduced_chi2, dof = reduced_chisquared(theoretical_freq[1:-points_to_remove], observed_freq[:,1:-points_to_remove])
    chisqauredList.append(reduced_chi2)

    print(f"Reduced chi-squared value for {m}: {reduced_chi2}, dof: {dof}")
    
#%% chi-squared

# circular argument -> assumed poisson distribution
# independent iterations -> likely normal distribution

def chisquared(theoretical_freq, observed_freq):
    mean = np.mean(observed_freq, axis=0)
    standard_deviation = np.std(observed_freq, axis=0)
    
    zList = []
    for i in range(len(theoretical_freq)):
        z = (theoretical_freq[i] - mean[i]) / standard_deviation[i]
        zList.append(z)
    chi2value = np.sum(np.array(zList) ** 2) # neglect first and last points because of poor statistics
    dof = len(np.array(zList)) - 1

    return chi2value, dof

chisqauredList = []

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.3, N, m, color=colorList[mList.index(m)])
    chi2value, dof = chisquared(theoretical_freq, observed_freq)
    chisqauredList.append(chi2value)
    
    print(f"Chi-squared value for {m}: {chi2value}, degree of freedom: {dof}")

    """
    if pvalue < 0.05:
        print(f"Reject null hypothesis for {m}, {pvalue} < 0.05")
    else:
        print(f"Accept null hypothesis for {m}, {pvalue} > 0.05")
    """

#%%
    
# remove last 5 points

points_to_remove = 5

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.3, N, m, color=colorList[mList.index(m)])
    chi2value, dof = chisquared(theoretical_freq[1:-points_to_remove], observed_freq[:,1:-points_to_remove])
    chisqauredList.append(chi2value)
    
    print(f"Chi-squared value for {m}: {chi2value}, degree of freedom: {dof}")
    """
    if pvalue < 0.05:
        print(f"Reject null hypothesis for {m}, {pvalue} < 0.05")
    else:
        print(f"Accept null hypothesis for {m}, {pvalue} > 0.05")
    """
# null hypothesis: same distribution
# alternative hypothesis: not the same distribution
# alpha = 0.05
#%%

"""
NList = [10,100,1000,10000,100000,1000000]

m = 2 # best fit -> fit is worse for larger m

foldername = "vary_N_dataset_PrefAttach"

if not os.path.exists(foldername):
    os.makedirs(foldername)

for N in NList:
    degreeLists = md.repeat_method(md.BaraAlbert, N, m, iteration_num, "PrefAttach")
    print(f"Finished N = {N}")
    data = dict()
    data["degreeLists"] = degreeLists
    
    with open(f"{foldername}/{N}.obj","wb") as f0:
        pickle.dump(data, f0)
"""

#%%

NList = [10,100,1000,10000,100000,1000000]

m = 2

foldername = "vary_N_dataset_PrefAttach"

average_k1List = []
standard_deviationList = []
absolute_errors = []

for N in NList:
    degreeLists = read_data(f"{foldername}/{N}")
    average_k1, standard_deviation = find_k1_distribution(degreeLists)
    average_k1List.append(average_k1)
    standard_deviationList.append(standard_deviation)
    absolute_error = average_k1 - largestk1_PrefAttach(N, m)
    absolute_errors.append(absolute_error)
    
#%%

plt.yscale('log')
plt.xscale('log')
plt.plot(NList,largestk1_PrefAttach(np.array(NList),m), "--", color="black", label="Theoretical fit")
plt.errorbar(NList,average_k1List,yerr=np.array(average_k1List) / np.sqrt(len(degreeLists)),fmt='x', label="Data")
plt.xlabel(r"$k$")
plt.ylabel(r"$k_{1}$")
plt.legend()

#%%

def linear(m,x,c):
    y = m*x+c
    return y

popt, pcov = curve_fit(linear, np.log10(np.array(NList)), np.log10(np.array(average_k1List)))

print(f"m = {popt[0]} \pm {np.sqrt(pcov[0][0])}")
print(f"c = {popt[1]} \pm {np.sqrt(pcov[1][1])}")
print(f"Percentage difference of m = {(popt[0]-0.5)/0.5*100}")

#%%

#plt.yscale('log')
plt.xscale('log')
plt.plot(NList,largestk1_PrefAttach(np.array(NList),m) /np.sqrt(np.array(NList)), "x", color="black", label="Theoretical fit")
plt.errorbar(NList,average_k1List/np.sqrt(np.array(NList)),yerr=np.array(average_k1List) / np.sqrt(len(degreeLists)) /np.sqrt(np.array(NList)),fmt='x', label="Data")
plt.xlabel(r"$k$", fontsize=16)
plt.ylabel(r"$k_{1} \: / \: \sqrt{N}$", fontsize=16)
plt.legend(loc=4)

#%% plot degree distribution
    
NList = [10,100,1000,10000,100000,1000000]

foldername = "vary_N_dataset_PrefAttach"

for N in NList:
    degreeLists = read_data(f"{foldername}/{N}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.25, N, m, Type="PrefAttach", vary="N", color=colorList[mList.index(m)])
    
#%%
"""
NList = [10,100,1000,10000,100000,1000000]

foldername = "vary_N_dataset_PrefAttach"

for N in NList:
    degreeLists = read_data(f"{foldername}/{N}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.25, N, m, average_k1List[NList.index(N)], Type="PrefAttach", vary="N", color=colorList[mList.index(m)])
"""
#%%

foldername = "vary_m_dataset_RandAttach"

colorList = ["tab:blue","tab:orange","tab:green","tab:red", "tab:purple", "tab:brown", "tab:pink"]

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    find_degree_distribution(degreeLists, 1.25, N, m, Type="RandAttach", color=colorList[mList.index(m)])
    
#%%

chisqauredList = []

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.25, N, m, Type="RandAttach", color=colorList[mList.index(m)])
    chi2value, dof = reduced_chisquared(theoretical_freq, observed_freq)
    chisqauredList.append(chi2value)
    
    print(f"Reduced Chi-squared value for {m}: {chi2value}, dof: {dof}")
    
#%%

points_to_remove = 3

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.25, N, m, Type="RandAttach", color=colorList[mList.index(m)])
    chi2value, dof = reduced_chisquared(theoretical_freq[1:-points_to_remove], observed_freq[:,1:-points_to_remove])
    chisqauredList.append(chi2value)
    
    print(f"Reduced Chi-squared value for {m}: {chi2value}, dof: {dof}")

#%%
    
# remove last 5 points

points_to_remove = 4

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.2, N, m, Type="RandAttach",color=colorList[mList.index(m)])
    chi2value, dof = chisquared(theoretical_freq[1:-points_to_remove], observed_freq[:,1:-points_to_remove])
    chisqauredList.append(chi2value)
    
    print(f"Chi-squared value for {m}: {chi2value}")
    print(f"Degree of freedoms for {m}: {dof}")

#%%

NList = [10,100,1000,10000,100000,1000000]

m = 2

foldername = "vary_N_dataset_RandAttach"

average_k1List = []
standard_deviationList = []
absolute_errors = []

for N in NList:
    degreeLists = read_data(f"{foldername}/{N}")
    average_k1, standard_deviation = find_k1_distribution(degreeLists)
    average_k1List.append(average_k1)
    standard_deviationList.append(standard_deviation)
    absolute_error = average_k1 - largestk1_RandAttach(N, m)
    absolute_errors.append(absolute_error)

#%%

#plt.yscale('log')
plt.xscale('log')
plt.plot(NList,largestk1_RandAttach(np.array(NList),m), "--", color="black", label="Theoretical fit")
plt.errorbar(NList,average_k1List,yerr=np.array(average_k1List) / np.sqrt(len(degreeLists)),fmt='x', label="Data")
plt.xlabel(r"$N$", fontsize=16)
plt.ylabel(r"$k_{1}$", fontsize=16)
plt.legend()

#%%

def linear(m,x,c):
    y = m*x+c
    return y

popt, pcov = curve_fit(linear, np.log10(np.array(NList)), np.array(average_k1List))

print(f"m = {popt[0]} \pm {np.sqrt(pcov[0][0])}")
print(f"c = {popt[1]} \pm {np.sqrt(pcov[1][1])}")
print(f"Percentage difference of m = {(popt[0]-0.5)/0.5*100}")

#%%

#plt.yscale('log')
plt.xscale('log')
plt.errorbar(NList, np.array(average_k1List) / np.log10(np.array(NList)), yerr=np.array(standard_deviationList)/np.sqrt(len(degreeLists)) / np.log10(np.array(NList)), fmt="x", capsize=3, label="Data")
plt.plot(NList, largestk1_RandAttach(np.array(NList), m)  / np.log10(np.array(NList)), 'x', color="black", label="Theoretical Result")
plt.xlabel(r"$N$", fontsize=16)
plt.ylabel(r"$k_{1} \: / \: \log(N)$", fontsize=16)
plt.legend()

#%%



#%% plot degree distribution
    
NList = [10,100,1000,10000,100000,1000000]

foldername = "vary_N_dataset_RandAttach"

for N in NList:
    degreeLists = read_data(f"{foldername}/{N}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.25, N, m, Type="RandAttach", vary="N", color=colorList[mList.index(m)])
    

#%%

foldername = "vary_m_dataset_ExistAttach"
mList=[2,4,8,16,32,64,128]
for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    find_degree_distribution(degreeLists, 1.25, N, m, Type="ExistAttach", vary="m", color=colorList[mList.index(m)])
    
#%%

chisqauredList = []

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.3, N, m, Type="ExistAttach", color=colorList[mList.index(m)])
    chi2value, dof = reduced_chisquared(theoretical_freq, observed_freq)
    chisqauredList.append(chi2value)
    
    print(f"Reduced Chi-squared value for {m}: {chi2value}, dof: {dof}")
    
#%%

points_to_remove = 2

for m in mList:
    degreeLists = read_data(f"{foldername}/{m}")
    theoretical_freq, observed_freq = find_degree_distribution(degreeLists, 1.3, N, m, Type="ExistAttach", color=colorList[mList.index(m)])
    chi2value, dof = reduced_chisquared(theoretical_freq[1:-points_to_remove], observed_freq[:,1:-points_to_remove])
    chisqauredList.append(chi2value)
    
    print(f"Reduced Chi-squared value for {m}: {chi2value}, dof: {dof}")


#%%

