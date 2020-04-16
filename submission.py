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

        return sorted_distace


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
    f_candidates = []
    code_list = []

    codes_vectors = np.split(codes, P, axis=1)
    for i in range(P):
        merged = list(itertools.chain.from_iterable(codes_vectors[i].tolist()))
        code_list.append(merged)

    for k in range(q_number):
        query = np.reshape(queries[k], (1, -1))
        sub_query_list = np.split(query, P, axis=1)  # Splitting the query into P parts
        final_sorted_dist = []
        cost_coord = {}
        index_positions = []

        for i in range(P):
            distance_list = distance_cal(sub_query_list[i], codebooks[i], "Query")
            final_sorted_dist.append(distance_list)
            ''' final sorted list consists of list of tuples of [[((2, 64.0),(3, 128.0),(1, 256.0),
                                                                (4, 320.0),(0, 448.0),(5, 512.0),(6, 704.0),
                                                                (7, 896.0),............)]]'''

        counter = 0
        for i in range(codebook_len):
            coordinates = [i for _ in range(P)]
            index_positions.append(coordinates)
            cost, orginal_coord = cost_func(final_sorted_dist, coordinates)
            cost_coord[counter] = [cost, coordinates, orginal_coord]
            counter += 1

            if i != codebook_len - 1:
                for j in range(P):
                    coordinates = [i for _ in range(P)]
                    coordinates[j] += 1
                    index_positions.append(coordinates)
                    cost, orginal_coord = cost_func(final_sorted_dist, coordinates)
                    cost_coord[counter] = [cost, coordinates, orginal_coord]
                    counter += 1

        queue = []
        final_candidates = set()
        candidates_list = []
        T_check = 0
        ded_check = []
        while T_check < T:
            # Finding the minimum cost
            if T_check == 0:
                element = cost_coord[0][1]
            else:
                element, key = find_element(queue, cost_coord)

            queue, new_ded_check = find_next(queue, element, index_positions, ded_check)

            candidate, queue, final_ded_check = get_Candidates(queue, cost_coord, code_list, new_ded_check)

            ded_check = final_ded_check
            final_candidates.update(candidate)

            T_check = len(final_candidates)

        f_candidates.append(final_candidates)

    return f_candidates


def find_next(Queue, coordinates, positions_list, check):
    for i in range(len(coordinates)):
        new_coordinates = copy.deepcopy(coordinates)
        new_coordinates[i] += 1

        # Finding the position of coordinate in the dictionary
        if new_coordinates in positions_list:
            pos = positions_list.index(new_coordinates)

            # Checking if coorinates already in queue and for dedup
            # The position is 1 means it has already been visited
            # We assign to position in position list 0 when we have found the new coordinate
            # Since the dictionary and position lists are arranged in same order we take new coordinates
            # position from position list as dicitonary key
            if pos not in check:
                Queue.append(pos)
                heapq.heapify(Queue)

    return Queue, check


def get_Candidates(Queue, cost_dict, codes, check):
    final_candidates = []
    coordinates, key = find_element(Queue, cost_dict)
    heapq.heappop(Queue)
    check.append(key)

    original_coord = cost_dict[key][2]

    for i in range(len(original_coord)):
        coor = original_coord[i]
        z = [i for i, x in enumerate(codes[i]) if x == coor]
        final_candidates.append(z)

    final_candidates = set(list(itertools.chain.from_iterable(final_candidates)))

    return final_candidates, Queue, check


def find_element(Queue, cost_dict):
    #     ele_key = min(Queue)
    ele_key = Queue[0]
    element = cost_dict[ele_key][1]

    return element, ele_key


def cost_func(final_list, coord):
    '''
    What does cost funciton do?

    '''
    cost = 0
    orginal_coord = []

    for i in range(len(coord)):
        number = coord[i]
        cost += final_list[i][number][1]
        orginal_coord.append(final_list[i][number][0])

    return cost, orginal_coord