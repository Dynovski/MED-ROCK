import numpy as np

from sortedcontainers import SortedList
from typing import List, Set
from tqdm import tqdm

from cluster import Cluster
from utils import euclidean_distance


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
        self.data: np.ndarray = data
        num_elements_to_sample: int = int(len(data) * sample_size)
        self.random_indices: np.ndarray = np.random.default_rng().choice(
            self.data.shape[0],
            size=num_elements_to_sample,
            replace=False
        )
        self.sample: np.ndarray = self.data[self.random_indices, :]
        print('Sampled data for detailed computation')
        self.num_clusters: int = num_clusters
        self.theta: float = theta
        self.links: np.ndarray = self.compute_links()
        print('Computed links')
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

        print('Calculated initial goodness measure and linked clusters')

        self.all_clusters: SortedList[Cluster] = SortedList(clusters_by_idx)

    @property
    def best_cluster(self) -> Cluster:
        return self.all_clusters[0]

    @property
    def result(self) -> List[List[int]]:
        result: List[List[int]] = []
        for cluster in self.all_clusters:
            real_indices = []
            for index in cluster.data_indices:
                real_indices.append(self.random_indices[index])
            result.append(sorted(real_indices))
        return result

    def compute_similarity(self, point: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def compute_adjacency_matrix(self) -> np.ndarray:
        raise NotImplementedError()

    def compute_links(self):
        matrix = self.compute_adjacency_matrix()
        result = matrix.dot(matrix)
        np.fill_diagonal(result, 0)
        return result

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

    def compute_similarity(self, point: np.ndarray) -> np.ndarray:
        return euclidean_distance(point, self.sample)

    def compute_adjacency_matrix(self) -> np.ndarray:
        adjacency_rows: List[np.ndarray] = []
        num_points: int = self.sample.shape[0]
        for i in range(num_points):
            point: np.ndarray = self.sample[i]
            sim_matrix: np.ndarray = self.compute_similarity(point)
            adjacency_rows.append(sim_matrix <= self.max_distance)
        return np.stack(adjacency_rows).astype(int)

