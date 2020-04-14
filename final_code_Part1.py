import numpy as np
from collections import deque
from scipy.spatial import distance
import pickle
import time

with open('datasets/Data_File', 'rb') as f:
    Data_File = pickle.load(f, encoding='bytes')
with open('datasets/Centroids_File', 'rb') as f:
    Centroids_File = pickle.load(f, encoding='bytes')


def pq(data, P, init_centroids, max_iter=20):
    final_codes_list = []
    final_code_book = []
    sub_vector_list = np.split(data, P, axis=1)

    for i in range(P):
        if i == 0:
            first_code_book, first_codes_list = k_means(sub_vector_list[i], init_centroids[i], max_iter)
            final_codes_list.append(first_codes_list)
            final_code_book.append(first_code_book)
        else:
            temp_code_book, temp_codes_list = k_means(sub_vector_list[i], init_centroids[i], max_iter)

            final_codes_list.append(temp_codes_list)
            final_code_book.append(temp_code_book)

    final_codes = np.array(final_codes_list)
    final_codes = np.transpose(final_codes)
    final_codes = final_codes.astype('uint8')

    final_code_book = np.array(final_code_book)

    return final_code_book, final_codes


def k_means(obs, code_book, iter=20):
    difference = np.inf
    distance_mean = deque([difference], maxlen=2)

    iteration = 0
    while (iteration < iter):
        # compute the codes and distances between each observation and code_book when the query == "PQ"
        obs_code, min_distace = distance_cal(obs, code_book, "PQ")
        distance_mean.append(min_distace.mean(axis=-1))

        # updating and creating new code books from the given observations
        code_book = update_code_book(obs, obs_code, code_book)
        # difference = distance_mean[0] - distance_mean[1]

        iteration += 1

    return code_book, obs_code


def distance_cal(obs, code_book, query_type):
    distance_array = distance.cdist(obs, code_book, 'cityblock')

    if query_type == 'PQ':
        code = distance_array.argmin(axis=1)
        min_dist = distance_array[np.arange(len(code)), code]

        return code, min_dist

    else:
        d_list = distance_array.tolist()
        distance_list = list(enumerate(d_list[0]))
        sorted_distace = sorted(distance_list, key=lambda x: x[1])

        return sorted_distace


def update_code_book(obs, codes, code_book):
    code_book_list = []
    cluster = dict_list(codes)
    missing_centorid = sorted(set(range(codes[0], codes[-1])) - set(codes))

    for j in range(code_book.shape[0]):
        centroid_cal = 0
        # new_centroid = []

        for i in cluster[j]:
            centroid_cal += obs[i]

        new_centroid = centroid_cal / len(cluster[j])
        code_book_list.append(list(new_centroid))

    new_code_book = np.array(code_book_list)

    if len(missing_centorid) != 0:
        for i in missing_centorid:
            new_code_book = np.insert(new_code_book, i, code_book[i], 0)

    return new_code_book


def dict_list(codes):
    enum_list = list(enumerate(codes))
    clusters = {}
    for index, cluster_no in enum_list:
        if cluster_no in clusters:
            clusters[cluster_no].append(index)
        else:
            clusters[cluster_no] = [index]

    return clusters


start = time.time()
codebooks, codes = pq(Data_File, P=2, init_centroids=Centroids_File, max_iter=20)
end = time.time()
time_cost_1 = end - start
print(time_cost_1)

print("Codebook shape: ",codebooks.shape)
print(" Codebook type: ",codebooks.dtype)
print("   Codes shape: ",codes.shape)
print("    Codes type: ",codes.dtype)

