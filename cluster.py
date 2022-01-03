from sortedcontainers import SortedList
from typing import Optional, List, Dict


class Cluster:
    def __init__(self, data_indices: List[int], sorted_list: Optional[SortedList['Cluster']] = None):
        self.data_indices: List[int] = data_indices
        self.linked_clusters: SortedList['Cluster'] = (
            sorted_list if sorted_list else SortedList()
        )
        self.cluster_to_goodness_d: Dict['Cluster', float] = {}

    @property
    def size(self) -> int:
        return len(self.data_indices)

    @property
    def num_links(self) -> int:
        return len(self.linked_clusters)

    @property
    def best_linked_cluster(self) -> 'Cluster':
        return self.linked_clusters[0]

    def add_linked_cluster(self, cluster: 'Cluster', goodness: float):
        self.linked_clusters.add(cluster)
        self.cluster_to_goodness_d[cluster] = goodness

    def get_goodness(self, cluster: 'Cluster') -> float:
        return self.cluster_to_goodness_d.get(cluster)

    def __lt__(self, other: 'Cluster'):
        return self.get_goodness(self.best_linked_cluster) < other.get_goodness(other.best_linked_cluster)

    def __eq__(self, other: 'Cluster'):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)
