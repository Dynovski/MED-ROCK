from sortedcontainers import SortedKeyList
from typing import Optional, List, Dict


class Cluster:
    def __init__(self, data_indices: List[int], sorted_list: Optional[SortedKeyList['Cluster']] = None):
        self.data_indices: List[int] = data_indices
        self.linked_clusters: SortedKeyList['Cluster'] = (
            sorted_list if sorted_list else SortedKeyList(key=lambda c: self.get_goodness(c))
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
        self.cluster_to_goodness_d[cluster] = goodness
        self.linked_clusters.add(cluster)

    def get_goodness(self, cluster: 'Cluster') -> float:
        return self.cluster_to_goodness_d.get(cluster)

    def __lt__(self, other: 'Cluster'):

        return self.get_goodness(self.best_linked_cluster) < other.get_goodness(other.best_linked_cluster)

    def __eq__(self, other: 'Cluster'):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)

    def __str__(self):
        return f'Cluster{self.data_indices}'

    def __repr__(self):
        return f'Cluster{self.data_indices}'
