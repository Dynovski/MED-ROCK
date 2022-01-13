import numpy as np

from sortedcontainers import SortedList
from multiprocessing import Pool
from typing import List, Set, Tuple
from tqdm import tqdm

import config
from cluster import Cluster
from utils import euclidean_distance, jaccard_coefficient, encode_categorical


class Rock:
    def __init__(
            self,
            data: np.ndarray,
            sample_size: float,
            num_clusters: int,
            theta: float,
    ):
        """

        :param data: numpy.ndarray
            all elements to cluster. Each column in array represents attribute, each row represents item
        :param sample_size: float
            value between 0 and 1, tells how much percent of all data elements will be in sampled data
        :param num_clusters: int
            number of clusters to output from algorithm
        :param theta: float
            value between 0 and 1, the closer to 1 the harder it is for two elements to be neighbours
        """
        self.result: List[List[int]] = []
        self.assign_rest_of_data: bool = sample_size < 1.0
        self.data: np.ndarray = data
        num_elements_to_sample: int = int(len(data) * sample_size)
        self.random_indices: np.ndarray = np.random.default_rng().choice(
            self.data.shape[0],
            size=num_elements_to_sample,
            replace=False
        )
        self.sample: np.ndarray = self.data[self.random_indices, :]
        self.num_clusters: int = num_clusters
        self.theta: float = theta
        self.links: np.ndarray = self.compute_links()
        self.goodness_exponent: float = 1 + 2 * (1.0 - self.theta) / (1.0 + self.theta)

        clusters_by_idx: List[Cluster] = []

        for i in range(num_elements_to_sample):
            clusters_by_idx.append(Cluster([i]))

        for i in range(num_elements_to_sample):
            cluster: Cluster = clusters_by_idx[i]
            similar_clusters: np.ndarray = np.nonzero(self.links[i, :])[0]
            for cluster_idx in similar_clusters:
                similar_cluster: Cluster = clusters_by_idx[cluster_idx]
                goodness: float = self.goodness_measure(cluster, similar_cluster)
                cluster.add_linked_cluster(similar_cluster, goodness)

        self.all_clusters: SortedList[Cluster] = SortedList(clusters_by_idx)

    @property
    def best_cluster(self) -> Cluster:
        return self.all_clusters[0]

    def compute_similarity(self, point: np.ndarray, points: np.ndarray, point_in_all_points: bool = True) -> np.ndarray:
        raise NotImplementedError()

    def compute_adjacency_matrix(self) -> np.ndarray:
        raise NotImplementedError()

    def compute_links(self) -> np.ndarray:
        matrix = self.compute_adjacency_matrix()
        result = matrix.dot(matrix)
        np.fill_diagonal(result, 0)
        return result

    def compute_num_neighbours(self, point: np.ndarray, points: np.ndarray) -> float:
        raise NotImplementedError()

    def goodness_measure(self, c1: Cluster, c2: Cluster) -> float:
        num_links: int = 0
        for i in c1.data_indices:
            for j in c2.data_indices:
                num_links += self.links[i, j]
        normalize_factor = (
                (c1.size + c2.size) ** self.goodness_exponent -
                c1.size ** self.goodness_exponent -
                c2.size ** self.goodness_exponent
        )
        if c1.size == 0 and c2.size == 0:
            return 0
        return -num_links / normalize_factor

    def get_best_cluster(self) -> Cluster:
        return self.all_clusters.pop(0)

    def run(self):
        bar = tqdm(desc='Processing clusters', total=len(self.all_clusters) - self.num_clusters)
        while len(self.all_clusters) > self.num_clusters:
            if len(self.best_cluster.linked_clusters) == 0:
                # No more possible clusters to merge
                break

            # pops u from all_clusters
            u = self.get_best_cluster()
            v = u.best_linked_cluster
            self.all_clusters.discard(v)

            w = Cluster(sorted(u.data_indices + v.data_indices))

            similar_clusters: Set[Cluster] = set()
            for i in u.linked_clusters:
                similar_clusters.add(i)
            for i in v.linked_clusters:
                similar_clusters.add(i)
            similar_clusters.discard(u)
            similar_clusters.discard(v)

            for x in similar_clusters:
                self.all_clusters.discard(x)

                x.remove_linked_cluster(u)
                x.remove_linked_cluster(v)

                goodness = self.goodness_measure(w, x)

                x.add_linked_cluster(w, goodness)
                w.add_linked_cluster(x, goodness)

                self.all_clusters.add(x)

            self.all_clusters.add(w)
            bar.update()

        for cluster in self.all_clusters:
            real_indices = []
            for index in cluster.data_indices:
                real_indices.append(self.random_indices[index])
            self.result.append(sorted(real_indices))

        if self.assign_rest_of_data:
            cluster_data_subsets: List[np.ndarray] = []
            for indices in self.result:
                indices_subset: np.ndarray = np.random.default_rng().choice(
                    indices,
                    size=int(0.7 * len(indices)),
                    replace=False
                )
                cluster_data_subsets.append(self.data[indices_subset, :])

            for i in range(self.data.shape[0]):
                if i in self.random_indices:
                    continue
                point: np.ndarray = self.data[i]
                best_cluster_index: int = 0
                best_score: float = 0.0
                for j in range(len(cluster_data_subsets)):
                    score = self.compute_num_neighbours(point, cluster_data_subsets[j])
                    if score > best_score:
                        best_cluster_index = j
                        best_score = score
                self.result[best_cluster_index].append(i)


class DistanceRock(Rock):
    def __init__(
            self,
            data: np.ndarray,
            sample_size: float,
            num_clusters: int,
            theta: float,
            max_distance: float):
        self.max_distance: float = max_distance
        super(DistanceRock, self).__init__(data, sample_size, num_clusters, theta)

    def compute_similarity(self, point: np.ndarray, points: np.ndarray, point_in_all_points: bool = True) -> np.ndarray:
        return euclidean_distance(point, points, point_in_all_points)

    def compute_adjacency_matrix(self) -> np.ndarray:
        adjacency_rows: List[np.ndarray] = []
        num_points: int = self.sample.shape[0]
        for i in range(num_points):
            point: np.ndarray = self.sample[i]
            sim_matrix: np.ndarray = self.compute_similarity(point, self.sample)
            adjacency_rows.append(sim_matrix <= self.max_distance)
        return np.stack(adjacency_rows).astype(int)

    def compute_num_neighbours(self, point: np.ndarray, points: np.ndarray) -> float:
        similarity: np.ndarray = self.compute_similarity(point, points, False)
        return (
                np.sum(similarity <= self.max_distance).item() /
                (points.shape[0] + 1) ** ((1.0 - self.theta) / (1.0 + self.theta))
        )


class CategoricalRock(Rock):
    def __init__(self, data: np.ndarray, sample_size: float, num_clusters: int, theta: float):
        categorical_data = encode_categorical(data)
        super(CategoricalRock, self).__init__(categorical_data, sample_size, num_clusters, theta)

    def compute_similarity(self, point: np.ndarray, points: np.ndarray, point_in_all_points: bool = True) -> np.ndarray:
        return jaccard_coefficient(point, points, point_in_all_points)

    def compute_adjacency_matrix(self) -> np.ndarray:
        adjacency_rows: List[np.ndarray] = []
        num_points: int = self.sample.shape[0]
        if config.USE_PARALLEL:
            parallel_computation_data: List[Tuple[int, np.ndarray]] = []
            unsorted_rows: List[Tuple[int, np.ndarray]] = []
            for i in range(num_points):
                point: np.ndarray = self.sample[i]
                parallel_computation_data.append((i, point))
            with Pool() as pool:
                unsorted_rows = pool.starmap(self.parallel_computations, parallel_computation_data)
            adjacency_rows: List[np.ndarray] = [tup[1] for tup in sorted(unsorted_rows)]
        else:
            bar = tqdm(desc='Computing similarity matrix', total=num_points)
            for i in range(num_points):
                point: np.ndarray = self.sample[i]
                sim_matrix: np.ndarray = self.compute_similarity(point, self.sample)
                adjacency_rows.append(sim_matrix >= self.theta)
                bar.update()
        return np.stack(adjacency_rows).astype(int)

    def parallel_computations(self, i, point):
        sim_matrix = self.compute_similarity(point, self.sample)
        return i, sim_matrix >= self.theta

    def compute_num_neighbours(self, point: np.ndarray, points: np.ndarray) -> float:
        similarity: np.ndarray = self.compute_similarity(point, points, False)
        return (
                np.sum(similarity >= self.theta).item() /
                (points.shape[0] + 1) ** ((1.0 - self.theta) / (1.0 + self.theta))
        )
