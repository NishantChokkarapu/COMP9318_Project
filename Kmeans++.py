## import modules here
import numpy as np
import pandas as pd
################# Question 1 #################

class KMeans:
    def __init__(self, k=256, tolerance = 0.001, max_iterations = 20):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iterations):
            self.result = []
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
#                 print(classification)
                self.result.append(classification)
                self.classes[classification].append(features)

            previous = dict(self.centroids)
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis = 0)

            isOptimal = True
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            if isOptimal:
                break

def kmean(data,k):
    km = KMeans(k)
    km.fit(data)
    final_list = km.result
    classes = km.classes
    return(final_list)
    
    
##############################
import pickle
import time
with open('./toy_example/Data_File', 'rb') as f:
    Data_File = pickle.load(f, encoding = 'bytes')
start = time.time()
print(kmean(Data_File, 20))
end = time.time()
print('Time: ',end - start)
