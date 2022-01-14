import heapq
from typing import List, Tuple


class Cluster:
    def __init__(self, data_indices: List[int]):
        self.data_indices: List[int] = data_indices
        self.linked_clusters: List[Tuple[float, 'Cluster']] = []

    @property
    def size(self) -> int:
        return len(self.data_indices)

    @property
    def best_linked_cluster(self) -> 'Cluster':
        return self.linked_clusters[0][1]

    def add_linked_cluster(self, cluster: 'Cluster', goodness: float):
        self.linked_clusters.append((goodness, cluster))
        heapq.heapify(self.linked_clusters)

    def remove_linked_cluster(self, cluster: 'Cluster') -> None:
        if cluster in self.cluster_to_goodness_d.keys():
            self.linked_clusters.discard(cluster)
            self.cluster_to_goodness_d.pop(cluster)

    def __lt__(self, other: 'Cluster'):
        if self.linked_clusters and other.linked_clusters:
            this_goodness = self.linked_clusters[0][0]
            other_goodness = other.linked_clusters[0][0]
            if this_goodness == other_goodness:
                return self.data_indices[0] < other.data_indices[0]
            return this_goodness < other_goodness
        elif other.linked_clusters:
            return False
        else:
            return True

    def __eq__(self, other):
        return self.data_indices[0] == other.data_indices[0]

    def __hash__(self):
        return self.data_indices[0]

    def __str__(self):
        return f'Cluster{self.data_indices}'

    def __repr__(self):
        return f'Cluster{self.data_indices}'
