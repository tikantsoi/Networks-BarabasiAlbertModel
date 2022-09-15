# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import cProfile
import logbin_2020 as logbin

#%%

class BaraAlbert():
    def __init__(self, tmax, m):
        self._tinitial = m+1 # tinital needs to be greater than m -> k cannot be < m
        # complete -> unbiased 
        self._G = nx.complete_graph(self._tinitial) 
        self._m = m
        self._tmax = tmax
        self._vertices = list(self._G.nodes)
        self._edges = list(self._G.edges)
        # degree list
        self._PAlist = [vertex for pair in self._edges for vertex in pair]
        self._binCentres = []
        self._degreeList = []

        if self._m > self._tinitial:
            raise ValueError("m cannot exceed number of nodes in initial graph")
            
        if self._tmax <= self._tinitial:
            raise ValueError("tmax has to be greater than tinitial")
    
    def PrefAttach(self, plot=False):
        for i in range(self._tmax-self._tinitial):
            index = self._tinitial + i 
            self._G.add_node(index)
            self._vertices = list(self._G.nodes) # update
            for j in range(self._m):
                randomPA = random.choice(self._PAlist)
                # avoid self-loop
                # avoid connecting to already connected edges 
                while randomPA==index or (randomPA, index) in self._edges:
                    randomPA = random.choice(self._PAlist)
                self._G.add_edge(index,randomPA)
                self._edges = list(self._G.edges)
                self._PAlist.append(index)
                self._PAlist.append(randomPA)
                
            if plot:
                plt.figure()  
                plt.title(f'$t$ = {index+1}')
                nx.draw(self._G, with_labels=True, font_weight='bold')
                plt.show()
                
        hist, binEdges = np.histogram(self._PAlist, np.arange(1,max(self._PAlist)+3))
        # degree k = 1,2,3...
        self._binCentres = list(binEdges[:-1])
        self._degreeList = list(hist)
                
    def Optimized_PrefAttach(self, plot=False):
        for i in range(self._tmax-self._tinitial):
            index = self._tinitial + i 
            #self._G.add_node(index)
            #self._vertices = list(self._G.nodes) # update
            self._vertices.append(index)
            temporary_edges = []
            for j in range(self._m):
                #randomPA = random.randint(min(self._PAlist),max(self._PAlist)+1)
                randomPA = random.choice(self._PAlist)
                # avoid self-loop
                # avoid connecting to already connected edges 
                while randomPA==index or (randomPA, index) in temporary_edges:
                    #randomPA = random.randint(min(self._PAlist),max(self._PAlist)+1)
                    randomPA = random.choice(self._PAlist)
                #self._G.add_edge(index,randomPA)
                #self._edges = list(self._G.edges)
                #self._edges.append((randomPA,index))
                temporary_edges.append((randomPA, index))
                self._PAlist.extend([index,randomPA])
                #self._PAlist.append(index)
                #self._PAlist.append(randomPA)
                
        hist, binEdges = np.histogram(self._PAlist, np.arange(1,max(self._PAlist)+3))
        # degree k = 1,2,3...
        self._binCentres = list(binEdges[:-1])
        self._degreeList = list(hist)
                
    def RandAttach(self, plot=False):
        for i in range(self._tmax-self._tinitial):
            index = self._tinitial + i 
            self._G.add_node(index)
            self._vertices = list(self._G.nodes) # update
            for j in range(self._m):
                # randomly choose a vertex
                random_choice = random.choice(self._vertices)
                # avoid self-loop
                # avoid connecting to already connected edges 
                while random_choice==index or (random_choice, index) in self._edges:
                    random_choice = random.choice(self._vertices)
                self._G.add_edge(index,random_choice)
                self._edges = list(self._G.edges)
                    
            if plot:
                plt.figure()  
                plt.title(f'$t$ = {index+1}')
                nx.draw(self._G, with_labels=True, font_weight='bold')
                plt.show()
                
        hist, binEdges = np.histogram(self._PAlist, np.arange(1,max(self._PAlist)+3))
        # degree k = 1,2,3...
        self._binCentres = list(binEdges[:-1])
        self._degreeList = list(hist)
                    
    def Optimized_RandAttach(self, plot=False):
        for i in range(self._tmax-self._tinitial):
            index = self._tinitial + i 
            #self._G.add_node(index)
            #self._vertices = list(self._G.nodes) # update
            self._vertices.append(index)
            temporary_edges = []
            for j in range(self._m):
                # randomly choose a vertex
                #random_choice = random.randint(min(self._vertices),max(self._vertices)+1)
                random_choice = random.choice(self._vertices)
                # avoid self-loop
                # avoid connecting to already connected edges 
                while random_choice==index or (random_choice, index) in temporary_edges:
                    #random_choice = random.randint(min(self._vertices),max(self._vertices)+1)
                    random_choice = random.choice(self._vertices)
                #self._G.add_edge(index,random_choice)
                #self._edges = list(self._G.edges)
                temporary_edges.append((random_choice,index))
                self._PAlist.extend([index,random_choice])
                    
        hist, binEdges = np.histogram(self._PAlist, np.arange(1,max(self._PAlist)+3))
        # degree k = 1,2,3...
        self._binCentres = list(binEdges[:-1])
        self._degreeList = list(hist)

    def Optimized_ExistAttach(self, plot=False):
        r = int(self._m / 2)
        for i in range(self._tmax-self._tinitial):
            index = self._tinitial + i 
            self._vertices.append(index)
            # Random attachment for r edges
            for j in range(r):
                # randomly choose a vertex
                random_choice = random.choice(self._vertices)
                #while random_choice==index or (random_choice, index) in temporary_edges:
                    #random_choice = random.choice(self._vertices)
                #temporary_edges.append((random_choice,index))
                self._PAlist.extend([random_choice,index])
            for j in range(self._m-r):
                # choose a vertex from PA list
                randomPA_1 = random.choice(self._PAlist)
                randomPA_2 = random.choice(self._PAlist)
                # avoid self loop
                while randomPA_2==randomPA_1 or randomPA_1==index or randomPA_2==index:
                    randomPA_1 = random.choice(self._PAlist)
                    randomPA_2 = random.choice(self._PAlist)
                self._PAlist.extend([randomPA_1,randomPA_2])
                
        hist, binEdges = np.histogram(self._PAlist, np.arange(1,max(self._PAlist)+3))
        # degree k = 1,2,3...
        self._binCentres = list(binEdges[:-1])
        self._degreeList = list(hist)   
        
"""
    def Optimized_ExistAttach(self, plot=False):
        # re-defined the network here
        r = int(self._m / 2)
        for i in range(self._tmax-self._tinitial):
            index = self._tinitial + i 
            self._vertices.append(index)
            # Random attachment for r edges
            for j in range(r):
                # randomly choose a vertex
                random_choice = int(len(self._vertices) * random.random())
                while random_choice==index or (random_choice, index) in self._edges:
                    random_choice = int(len(self._vertices) * random.random())
                self._edges.append((random_choice,index))
                self._PAlist.append(random_choice)
            for j in range(self._m-r):
                # choose a vertex from PA list
                randomPA_1 = round(max(self._PAlist) * random.random())
                randomPA_2 = round(max(self._PAlist) * random.random())
                randomPAList = [randomPA_1,randomPA_2]
                # avoid self loop
                while randomPA_2==randomPA_1 or (randomPA_1,randomPA_2) in self._edges or (randomPA_2,randomPA_1) in self._edges:
                    randomPA_1 = round(max(self._PAlist) * random.random())
                    randomPA_2 = round(max(self._PAlist) * random.random())
                randomPAList = [randomPA_1,randomPA_2]
                self._edges.append((min(randomPAList),max(randomPAList)))
                self._PAlist.extend(randomPAList)
            self._PAlist.append(index)
                
        hist, binEdges = np.histogram(self._PAlist, np.arange(1,max(self._PAlist)+3))
        # degree k = 1,2,3...
        self._binCentres = list(binEdges[:-1])
        self._degreeList = list(hist)        
"""                 

def repeat_method(model_class, N, m, iteration_num, method=""):
    degreeLists = []
    for i in range(iteration_num):
        t=time.time()
        model = model_class(N,m)
        if method == "PrefAttach":
            model.PrefAttach()
        if method == "Optimized_PrefAttach":
            model.Optimized_PrefAttach()     
        if method == "RandAttach":
            model.RandAttach()       
        if method == "Optimized_RandAttach":
            model.Optimized_RandAttach()  
        elapsed_time = time.time() - t
        if method == "Optimized_ExistAttach":
            model.Optimized_ExistAttach()  
        elapsed_time = time.time() - t
        print(f"repeat_{i} took {elapsed_time} s to run")
        degreeList = model._degreeList
        degreeLists.append(degreeList)
        
    return degreeLists

