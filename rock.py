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
        self.random_indices: np.ndarray = np.random.randint(self.data.shape[0], size=num_elements_to_sample)
        self.sample: np.ndarray = self.data[self.random_indices, :]
        print('Sampled data for detailed computation')
        self.num_clusters: int = num_clusters
        self.theta: float = theta
        self.max_distance: float = max_distance
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

    def compute_similarity_matrix(self, point: np.ndarray) -> np.ndarray:
        return euclidean_distance(point, self.sample)

    def find_neighbours(self) -> List[np.ndarray]:
        neighbours: List[np.ndarray] = []
        num_points: int = self.sample.shape[0]
        for i in range(num_points):
            point: np.ndarray = self.sample[i]
            sim_matrix: np.ndarray = self.compute_similarity_matrix(point)
            neighbours.append(np.where(sim_matrix <= self.max_distance)[0])
        return neighbours

    def compute_links(self) -> np.ndarray:
        neighbors: List[np.ndarray] = self.find_neighbours()
        print('Computed neighbours list')
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
            print(f'Clusters left: {len(self.all_clusters)}')
