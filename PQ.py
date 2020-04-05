import pickle
import time
import numpy as np
import random

with open('./datasets/Data_File', 'rb') as f:
    Data_File = pickle.load(f, encoding = 'bytes')
with open('./datasets/Centroids_File', 'rb') as f:
    Centroids_File = pickle.load(f, encoding = 'bytes')
start = time.time()
# codebooks, codes = submission.pq(data, P=2, init_centroids=centroids, max_iter = 20)
end = time.time()
time_cost_1 = end - start


# How to run your implementation for Part 2
with open('./datasets/Query_File', 'rb') as f:
    Query_File = pickle.load(f, encoding = 'bytes')
# queries = pickle.load(Query_File, encoding = 'bytes')
start = time.time()
# candidates = submission.query(queries, codebooks, codes, T=10)
end = time.time()
time_cost_2 = end - start

# output for part 2.
# print(candidates)

print(Data_File.shape)
print(Centroids_File.shape)

def pq(input_data, p, centroids, iterations):

    #Splitting the input data into subvectors and it returns a list of arrays
    sub_vector_list = np.split(input_data, p)
    print(len(sub_vector_list[0]))


pq(Data_File, 2, Centroids_File, 20)


# #Manhattan Distance
# def L1(v1,v2):
#     if(len(v1)!=len(v2)):
#         print('error')
#         return -1
#     # return sum([abs(v1[i]-v2[i]) for i in range(len(v1))])
#
#     return np.linalg.norm(v1-v2, 1)


# # kmeans with L1 distance.
# # rows refers to the NxM feature vectors
# def kcluster(rows,distance=L1,k=4):# Cited from Programming Collective Intelligence
#     # Determine the minimum and maximum values for each point
#     ranges=[(min([row[i] for row in rows]),max([row[i] for row in rows])) for i in range(len(rows[0]))]
#
#     # Create k randomly placed centroids
#     clusters=[[random.random( )*(ranges[i][1]-ranges[i][0])+ranges[i][0] for i in range(len(rows[0]))] for j in range(k)]
#
#     lastmatches=None
#     for t in range(100):
#         print 'Iteration %d' % t
#         bestmatches=[[] for i in range(k)]
#         # Find which centroid is the closest for each row
#         for j in range(len(rows)):
#             row=rows[j]
#             bestmatch=0
#             for i in range(k):
#                 d=distance(clusters[i],row)
#                 if d<distance(clusters[bestmatch],row):
#                     bestmatch=i
#             bestmatches[bestmatch].append(j)
#         ## If the results are the same as last time, this is complete
#         if bestmatches==lastmatches:
#             break
#         lastmatches=bestmatches
#
#         # Move the centroids to the average of their members
#         for i in range(k):
#             avgs=[0.0]*len(rows[0])
#             if len(bestmatches[i])>0:
#                 for rowid in bestmatches[i]:
#                     for m in range(len(rows[rowid])):
#                         avgs[m]+=rows[rowid][m]
#                 for j in range(len(avgs)):
#                     avgs[j]/=len(bestmatches[i])
#                 clusters[i]=avgs
#     return bestmatches