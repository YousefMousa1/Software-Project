from sklearn import metrics
from symnmf import *
import math
import sys
import copy
import numpy as np
import pandas as pd
import mysymnmf as mf

np.random.seed(0)

#the function calculates the euclidean distance between p1 and p2.
#p1, p2 are arrays with float values, representing d dimentional points
def distance(p1, p2, d):
    sum_of_squares = 0.0
    for i in range(d):
        sum_of_squares += math.pow(float(p1[i])-float(p2[i]), 2)
    distance= math.sqrt(sum_of_squares)
    return distance

#this function returns the index of the centroid closest to x.
#x is a point in R^d, centroids is an array with k points in R^d
def clusterAssign(x, centriods, k, d):
   min_distance = distance(x, centriods[0], d)
   cluster = 0
   for i in range(1, k):
       dist = distance(x, centriods[i], d)
       if dist <= min_distance:
           min_distance = dist
           cluster = i
   return cluster 

#this function calculates the new centroids from the given clusters.
#clusters is a 2D array of sum of the points in each cluster
#size of clusters is an array of the size of each cluster
#k is the number of clusters
#d is the number of coordinates of each point
def centriodUpdate(clusters, size_of_clusters, k: int, d: int):
    for i in range(k):
        for j in range(d):
            if size_of_clusters[i]==0:
                print("An Error Has Occurred")
                exit(0)
            clusters[i][j] =  clusters[i][j]/size_of_clusters[i]
    return

#this function performs the kmeans algorithm from the datapoints elements.
#there are num_of_elements d dimentional datapoints, and k required clusters.
# returns an array with num_of_elements cells, where the i'th cell contains
# the cluster that the i'th datapoint belongs to.
def kmeans(elements, num_of_elements, k, d):
    centroids = []
    clusters = [[0.0 for i in range(d)] for j in range(k)] 
    size_of_clusters = [0 for i in range(k)]

    for i in range(k):
        centroids.append(copy.deepcopy(elements[i]))

    assigned_centriod=0
    for i in range(300):
        for j in range(num_of_elements):
            assigned_centriod = clusterAssign(elements[j],centroids, k, d)
            for l in range(d):
                clusters[assigned_centriod][l] += elements[j][l]
            size_of_clusters[assigned_centriod]+=1
        centriodUpdate(clusters, size_of_clusters, k, d)
        j=0
        end=1
        for j in range(k):
            if distance(centroids[j], clusters[j], d)>=0.0001:
                end=0
                break
        if end:
            break  
        j=0
        l=0
        for j in range(k):
            for l in range(d):
                centroids[j][l] = float(clusters[j][l])
                clusters[j][l] = 0.0
            size_of_clusters[j] = 0

    final_clusters = [clusterAssign(elements[x], centroids, k, d) for x in range(num_of_elements)]
    return final_clusters

#this function returns an array with num_of_elements cells, where the i'th cell contains
# the cluster that the i'th datapoint belongs to, from the result of symNMF.
# matrix is the result matrix of symNMF.    
def symnmfClusterAssign(matrix, num_of_elements):
    clusters = []
    for i in range(num_of_elements):
        max_i = max(matrix[i])
        clusters.append(matrix[i].index(max_i))
    return clusters


def main():
    file_name=""
    elements = []
    num_of_elements=0
    d=0
    
    if len(sys.argv) == 3:

        k = int(sys.argv[1])
        file_name = str(sys.argv[2])
    else:
        print("An Error Has Occurred")
        exit(0)

    file = pd.read_csv(file_name, header=None)

    elements = file.values #this is the array of the datapoints from the file

    num_of_elements=len(elements)
    d = len(elements[0])

    if k<1 or k>num_of_elements:
        print("An Error Has Occurred")
        exit(0)

    #calculating the silhouette score of kmeans
    kmeans_clusters = kmeans(elements, num_of_elements, k, d)
    kmeans_silhouette = metrics.silhouette_score(elements, kmeans_clusters)

    #calculating the silhouette score of symNMF
    W = mf.norm(elements.tolist(), num_of_elements, d)
    H = initialize_H(W, num_of_elements, k)
    new_H = mf.symnmf(H.tolist(), W, k, num_of_elements)
    symnmf_clusters = symnmfClusterAssign(new_H, num_of_elements)
    symnmf_silhouette = metrics.silhouette_score(elements, symnmf_clusters)

    print("nmf: %.4f" % symnmf_silhouette)
    print("kmeans: %.4f"  % kmeans_silhouette)

if __name__ == "__main__":
    main()

    

