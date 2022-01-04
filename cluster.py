from sortedcontainers import SortedKeyList
from typing import List, Dict


class Cluster:
    def __init__(self, data_indices: List[int]):
        self.data_indices: List[int] = data_indices
        self.linked_clusters: SortedKeyList['Cluster'] = SortedKeyList(
            key=lambda c: (self.get_goodness(c), tuple(c.data_indices))
        )
        self.cluster_to_goodness_d: Dict['Cluster', float] = {}

    @property
    def size(self) -> int:
        return len(self.data_indices)

    @property
    def best_linked_cluster(self) -> 'Cluster':
        return self.linked_clusters[0]

    def add_linked_cluster(self, cluster: 'Cluster', goodness: float):
        self.cluster_to_goodness_d[cluster] = goodness
        self.linked_clusters.add(cluster)

    def get_goodness(self, cluster: 'Cluster') -> float:
        return self.cluster_to_goodness_d.get(cluster, 0.0)

    def remove_linked_cluster(self, cluster: 'Cluster') -> None:
        if cluster in self.cluster_to_goodness_d.keys():
            self.linked_clusters.discard(cluster)
            self.cluster_to_goodness_d.pop(cluster)

    def __lt__(self, other: 'Cluster'):
        # zdarzają się elementy, które mają identyczne goodness, trzeba posłużyć się czymnś więcej przy sortowaniu
        this_goodness = self.get_goodness(self.best_linked_cluster)
        other_goodness = other.get_goodness(other.best_linked_cluster)
        if this_goodness == other_goodness:
            return tuple(self.data_indices) < tuple(other.data_indices)
        return self.get_goodness(self.best_linked_cluster) < other.get_goodness(other.best_linked_cluster)

    def __eq__(self, other: 'Cluster'):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)

    def __str__(self):
        return f'Cluster{self.data_indices}'

    def __repr__(self):
        return f'Cluster{self.data_indices}'
