# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:07:44 2020

@author: KOUSIK ROY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import math


#file=input("enter data file name:")
file="iris.csv"
Raw_data=open(file,"r")
data=pd.read_csv(Raw_data)
data=preprocessing.normalize(data,norm="l2")#normalization
data=data[:,:2]#This is just for visualization of iris using two features only
ax=plt.scatter(data[:,0],data[:,1])
plt.show(ax)
m=data.shape[0]
k=math.ceil(5*m/100)+5#value of k for knn
knn=NearestNeighbors(n_neighbors=k,algorithm="ball_tree").fit(data)
distances,indices=knn.kneighbors(data)
p=np.zeros((m,1)) #initialize density vector

for i in range(m):
    p[i]=1/(np.sum(distances[i,:],axis=0))
#ax1=plt.scatter(data.X,p)
#plt.show(ax1)
#plt.scatter(np.arange(0,m),p)
p_idx=np.concatenate((p,np.arange(0,m).reshape(m,1)),axis=1)#adding index column
p_sort=p_idx[p_idx[:,0].argsort()][::-1] #basically sorting in decreasing order based on first column values
p_fil=p_sort
i=0
while i < p_fil.shape[0]:
    try:
        ind=indices[int(p_fil[i,1])][1:]#indices of knn neighbors#[1:] to make sure it doesnot add its own index
        _,_,d_idx=np.intersect1d(ind,p_fil[:,1],return_indices=True)#indices of knn neighbors that are present in p_sort
    
        p_fil=np.concatenate((p_fil[:i,:],np.delete(p_fil[i:,:],d_idx,axis=0)),axis=0)#deleting all neighbors which lie below the point in p_sort
    except IndexError:#to handle the possible error for now , will think more about this later
        continue
    i=i+1
    
i=p_fil.shape[0]-1# set i to start loop from bottom
delta=np.zeros((p_fil.shape[0],1))# to keep min distances of higher density points
gamma=np.zeros((p_fil.shape[0],1))#delta * p_fil[:,0]
temp=[]#to hold distances for a iteration
while i > 0:
    for j in range(i):
        temp.append((np.sum((data[int(p_fil[i,1]),:]-data[int(p_fil[j,1]),:])**2))**0.5)#calculating distances of points according to p_fil starting from bottom and only going up in each iteration
    delta[i]=min(temp) # min distance 
    temp=[]#clear temp for next iteration
    i=i-1   
delta[0] = np.max(delta) # setting a high relative distance for the highest density point
gamma=np.multiply(delta,p_fil[:,0].reshape(p_fil.shape[0],1))

ax2=plt.scatter(np.arange(0,gamma.shape[0]),gamma)  
plt.show(ax2)  

color=['red','yellow','green','blue','black']   
for i in range(m):
    ax3=plt.scatter(data[int(p_sort[i,1]),0],data[int(p_sort[i,1]),1],c=color[math.ceil(i/(m/5))-1])#for visualization of density as colors red is most densed and black is least densed
    
plt.show(ax3)

