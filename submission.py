import numpy as np
from collections import deque
from scipy.spatial import distance
import itertools
import copy
import heapq


def pq(data, P, init_centroids, max_iter):
    final_codes_list = []
    final_code_book = []
    sub_vector_list = np.split(data, P, axis=1)
    k_list = [i for i in range(init_centroids.shape[1])]

    for i in range(P):
        if i == 0:
            first_code_book, first_codes_list = k_means(sub_vector_list[i], init_centroids[i], k_list, max_iter)
            final_codes_list.append(first_codes_list)
            final_code_book.append(first_code_book)
        else:
            temp_code_book, temp_codes_list = k_means(sub_vector_list[i], init_centroids[i], k_list, max_iter)

            final_codes_list.append(temp_codes_list)
            final_code_book.append(temp_code_book)

    final_codes = np.array(final_codes_list)
    final_codes = np.transpose(final_codes)
    final_codes = final_codes.astype('uint8')

    final_code_book = np.array(final_code_book)

    return final_code_book, final_codes


def k_means(obs, code_book, k_list, iter=20):
    difference = np.inf
    distance_mean = deque([difference], maxlen=2)

    iteration = 0
    while (iteration < iter):
        # compute the codes and distances between each observation and code_book when the query == "PQ"
        obs_code, min_distace = distance_cal(obs, code_book, "PQ")
        distance_mean.append(min_distace.mean(axis=-1))

        # updating and creating new code books from the given observations
        code_book = update_code_book(obs, obs_code, code_book, k_list)
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
        d_codes = [c for c, dis in sorted_distace]
        d_cost = [dis for c, dis in sorted_distace]

        return d_codes, d_cost


def update_code_book(obs, codes, code_book, k_list):
    code_book_list = []
    cluster = dict_list(codes)
    missing_centorid = sorted(set(k_list) - set(codes))

    for j in range(code_book.shape[0]):
        centroid_cal = 0

        if j in cluster:
            for i in cluster[j]:
                centroid_cal += obs[i]

            new_centroid = centroid_cal / len(cluster[j])
            code_book_list.append(list(new_centroid))

    new_code_book = np.array(code_book_list)

    if len(missing_centorid) != 0:
        for i in missing_centorid:
            new_code_book = np.insert(new_code_book, i, code_book[i], 0)

    return code_book


def dict_list(codes):
    enum_list = list(enumerate(codes))
    clusters = {}
    for index, cluster_no in enum_list:
        if cluster_no in clusters:
            clusters[cluster_no].append(index)
        else:
            clusters[cluster_no] = [index]

    return clusters


def query(queries, codebooks, codes, T=10):
    q_number = queries.shape[0]
    P = codebooks.shape[0]
    codebook_len = codebooks.shape[1]
    code_list = []
    f_candidates = []

    codes_vectors = np.split(codes, P, axis=1)
    for i in range(P):
        merged = list(itertools.chain.from_iterable(codes_vectors[i].tolist()))
        code_list.append(merged)

    for k in range(q_number):
        query = np.reshape(queries[k], (1, -1))
        sub_query_list = np.split(query, P, axis=1)  # Splitting the query into P parts
        final_sorted_dist = []
        final_cost_list = []
        cost_coor = {}
        queue = []
        ded_up = []

        for i in range(P):
            distance_list, cost_list = distance_cal(sub_query_list[i], codebooks[i], "Query")
            final_sorted_dist.append(distance_list)
            final_cost_list.append(cost_list)

        code_distance = np.transpose(np.array(final_sorted_dist))
        code_cost = np.transpose(np.array(final_cost_list))

        coor = [0 for _ in range(P)]
        first_cost = sum([code_cost[0][i] for i in range(P)])
        cost_coor[first_cost] = [coor]

        T_check = 0
        first_loop_check = True
        codes_check_list = [[] for _ in range(P)]
        w_candidates = set()
        while T_check < T:
            if first_loop_check:
                queue.append(first_cost)
                queue, ded_up, cost_coor, coor_check = cost_neighbours(queue, ded_up, cost_coor, code_cost, P)
                first_loop_check = False

            else:
                queue, ded_up, cost_coor, coor_check = cost_neighbours(queue, ded_up, cost_coor, code_cost, P)

            for column in range(P):
                row = coor_check[column]
                code_check = code_distance[row][column]

                if code_check not in codes_check_list[column]:  # Checkin if a code of a partic
                    candidate = set([i for i, val in enumerate(code_list[column]) if val == code_check])

                w_candidates.update(candidate)

            T_check = len(w_candidates)

        f_candidates.append(w_candidates)

    return f_candidates


def cost_neighbours(queue, ded_up, cost_coor, code_cost, P):
    key = queue[0]  # Getting the first cost from the queue

    if len(cost_coor[key]) == 1:
        coordinates = cost_coor[key].pop(0)  # Getting the coordinates of the first element in queue
        del cost_coor[key]  # Deleting the key from the dictionary
        heapq.heappop(queue)
    else:
        coordinates = cost_coor[key].pop(0)

    if coordinates not in ded_up:
        neighbours = []

        for i in range(P):
            new_coordinates = copy.deepcopy(coordinates)
            new_coordinates[i] = coordinates[i] + 1

            neigh_cost = cal_cost(new_coordinates, code_cost)  # Calculating cost of new neighbour

            if neigh_cost in cost_coor:
                cost_coor[neigh_cost].append(new_coordinates)
            else:
                cost_coor[neigh_cost] = [new_coordinates]
                queue.append(neigh_cost)

    heapq.heapify(queue)
    ded_up.append(coordinates)  # appending the coordinates to dedup to make we have already visisted

    return queue, ded_up, cost_coor, coordinates


def cal_cost(new_coordinates, code_cost):
    cost = 0

    for column in range(len(new_coordinates)):
        row = new_coordinates[column]
        cost += code_cost[row][column]

    return cost