import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def feature_normalize(data): #needs improvement
    mean=np.mean(data)
    std=np.sqrt(np.sum((data-mean)**2)/np.ptp(data))
    data=(data-mean)/std
    return data;

def loading_data(file):
    raw=open(file,'r')
    data=np.loadtxt(raw, dtype='float', delimiter=',', skiprows=1)
    return data;
def initialize_centroids(data, k):
    (m,n)=np.shape(data)
    centroids=np.zeros((k,n))
    np.random.shuffle(data)
    centroids=data[:k,:]
    return centroids;
def find_closest_centroids(data,centroids,k):
    (m,n)=np.shape(data)
    idx=np.zeros((m,1),dtype='int')
    dist=np.zeros((k,1))
    for i in range(0,m):
        for j in range(0,k):
            dist[j]=np.sqrt(np.sum((data[i,:]-centroids[j,:])**2))
        mark=np.argmin(dist)
        idx[i]=mark
    return idx;
def compute_centroids(data,centroids,idx, k):
    (m,n)=np.shape(data)
    for i in range(0,k):
        count=0
        s=np.zeros((1,n))
        for j in range(0,m):
            if idx[j]==i:
                s=s+data[j,:]
                count+=1
        mean=s/count
        cost=math.inf
        for b in range(0,m):
            if idx[b]==i:
                c=np.sum((data[b,:]-mean)**2)
            if c<cost:
                cost=c
                w=b
        medoid=data[w,:]
        centroids[i,:]=medoid
    return centroids;
def plotting_3D(data,idx):
    (m,n)=np.shape(data)
    fig=plt.figure(figsize=(15,15))
    ax=fig.add_subplot(1,1,1,projection='3d')
    a1=np.empty((1,n))
    a2=np.empty((1,n))
    a3=np.empty((1,n))
    for i in range(0,m):
        if idx[i]==0:
            a1=np.append(a1,[data[i,:]],axis=0)
        elif idx[i]==1:
            a2=np.append(a2,[data[i,:]],axis=0)
        elif idx[i]==2:
            a3=np.append(a3,[data[i,:]],axis=0)
    x1=a1[:,0]
    y1=a1[:,1]
    z1=a1[:,2]
    figure=ax.scatter(x1,y1,z1,color='red')
    x2=a2[:,0]
    y2=a2[:,1]
    z2=a2[:,2]
    figure=ax.scatter(x2,y2,z2,color='green')
    x3=a3[:,0]
    y3=a3[:,1]
    z3=a3[:,2]
    figure=ax.scatter(x3,y3,z3,color='blue')
    plt.title("3d visualization of clusters taking 3 features")
    plt.xlabel("sepal length")
    plt.ylabel("sepal width")
    plt.xlim(-.25,.25)
    plt.ylim(-.25,.25)
    ax.set_zlim(-.25,.25)
    ax.set_zlabel("petal length")
    plt.show()
    return;
def plotting_2D(data,idx):
    (m,n)=np.shape(data)
    fig=plt.figure(figsize=(15,15))
    ax=fig.add_subplot(1,1,1)
    a1=np.empty((1,n))
    a2=np.empty((1,n))
    a3=np.empty((1,n))
    for i in range(0,m):
        if idx[i]==0:
            a1=np.append(a1,[data[i,:]],axis=0)
        elif idx[i]==1:
            a2=np.append(a2,[data[i,:]],axis=0)
        elif idx[i]==2:
            a3=np.append(a3,[data[i,:]],axis=0)
    x1=a1[:,0]
    y1=a1[:,1]
    
    figure=ax.scatter(x1,y1,color='red')
    x2=a2[:,0]
    y2=a2[:,1]
    
    figure=ax.scatter(x2,y2,color='green')
    x3=a3[:,0]
    y3=a3[:,1]
    
    figure=ax.scatter(x3,y3,color='blue')
    plt.title("2d visualization of clusters taking 2 features")
    plt.xlabel("sepal length")
    plt.ylabel("sepal width")
    plt.xlim(-.25,.25)
    plt.ylim(-.25,.25)
    plt.show()
    return;






def main():
    k=int(input("enter no of clusters:"))
    file=input("enter data file name:")
    data=loading_data(file)
    data=feature_normalize(data) 
    centroids=initialize_centroids(data,k)
    iter=int(input("enter no of iterations you want:"))
    for i in range(0,iter):
        idx=find_closest_centroids(data,centroids,k)
        centroids=compute_centroids(data,centroids,idx,k)
    print(data)
    print(centroids)
    plotting_3D(data,idx)
    plotting_2D(data,idx)
    
    return;
main()






    
                           
