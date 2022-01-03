import copy
import numpy as np

from sortedcontainers import SortedList
from typing import List, Set

from cluster import Cluster
from utils import euclidean_distance


class Rock:
    def __init__(
            self,
            data: np.ndarray,
            sample_size: float,
            num_clusters: int,
            theta: float,
            max_distance: float
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
        random_indices: np.ndarray = np.random.randint(self.data.shape[0], size=num_elements_to_sample)
        self.sample: np.ndarray = self.data[random_indices, :]
        self.num_clusters: int = num_clusters
        self.theta: float = theta
        self.max_distance: float = max_distance
        self.links: np.ndarray = self.compute_links()
        self.goodness_exponent: float = 1 + 2 * (1.0 - self.theta) / (1.0 + self.theta)
        self.all_clusters: SortedList[Cluster] = SortedList()

        self.clusters_by_idx: List[Cluster] = []

        for i in range(num_elements_to_sample):
            self.clusters_by_idx.append(Cluster([i]))

        for i in range(num_elements_to_sample):
            cluster: Cluster = self.clusters_by_idx[i]
            similar_clusters: np.ndarray = np.nonzero(self.links[i, :])[0]
            for cluster_idx in similar_clusters:
                similar_cluster: Cluster = self.clusters_by_idx[cluster_idx]
                goodness: float = self.goodness_measure(cluster, similar_cluster)
                cluster.add_linked_cluster(similar_cluster, goodness)

        self.all_clusters: SortedList[Cluster] = SortedList(copy.copy(self.clusters_by_idx))

    def compute_similarity_matrix(self, point: np.ndarray) -> np.ndarray:
        return euclidean_distance(point, self.data)

    def find_neighbours(self) -> List[np.ndarray]:
        neighbours: List[np.ndarray] = []
        num_points: int = self.data.shape[0]
        for i in range(num_points):
            point: np.ndarray = self.data[i]
            sim_matrix: np.ndarray = self.compute_similarity_matrix(point)
            neighbours.append(np.where(sim_matrix <= self.max_distance))
        return neighbours

    def compute_links(self) -> np.ndarray:
        neighbors: List[np.ndarray] = self.find_neighbours()
        num_data = self.sample.shape[0]
        links_matrix: np.ndarray = np.zeros((num_data, num_data), dtype=int)
        for i in range(num_data):
            i_neighbors = neighbors[i]
            for j in range(i_neighbors.shape[0] - 1):
                for k in range(j + 1, i_neighbors.shape[0]):
                    links_matrix[i_neighbors[j], i_neighbors[k]] += 1
                    links_matrix[i_neighbors[k], i_neighbors[j]] += 1
            return links_matrix

    def goodness_measure(self, c1: Cluster, c2: Cluster) -> float:
        num_links: int = c1.num_links + c2.num_links
        normalize_factor = (
                (c1.size + c2.size) ** self.goodness_exponent -
                c1.size ** self.goodness_exponent -
                c2.size ** self.goodness_exponent
        )
        return num_links / normalize_factor

    @property
    def best_cluster(self) -> Cluster:
        return self.all_clusters[0]

    def get_best_cluster(self) -> Cluster:
        return self.all_clusters.pop(0)

    def run(self):
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
            similar_clusters.remove(u)
            similar_clusters.remove(v)

            for x in similar_clusters:
                self.all_clusters.remove(x)

                try:
                    x.linked_clusters.discard(u)
                    del x.cluster_to_goodness_d[u]
                    x.linked_clusters.discard(v)
                    del x.cluster_to_goodness_d[v]
                except KeyError:
                    pass

                goodness = self.goodness_measure(w, x)

                x.add_linked_cluster(w, goodness)
                w.add_linked_cluster(x, goodness)

                self.all_clusters.add(x)

            self.all_clusters.add(w)
            print(f'Clusters left: {len(self.all_clusters)}')
