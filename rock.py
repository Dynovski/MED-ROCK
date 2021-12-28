import numpy as np

from sortedcontainers import SortedList
from typing import List

from cluster import Cluster


class Rock:
    def __init__(
            self,
            data: np.ndarray,
            sample_size: float,
            num_clusters: int,
            theta: float
    ):
        """

        :param data: numpy.ndarray
            all elements to cluster
        :param sample_size: float
            value between 0 and 1, tells how much percent of all data elements will be in sampled data
        :param num_clusters: int
            number of clusters to output from algorithm
        :param theta: float
            value between 0 and 1, the closer to 1 the harder it is for two elements to be neighbours
        """
        self.data: np.ndarray = data
        num_elements_to_sample: int = int(len(data) * sample_size)
        self.sample: np.ndarray = np.random.choice(data, num_elements_to_sample)
        self.num_clusters: int = num_clusters
        self.theta: float = theta
        self.goodness_exponent = 1 + 2 * (1.0 - self.theta) / (1.0 + self.theta)
        self.linked_clusters: List[SortedList[Cluster]] = []
        self.all_clusters: SortedList[Cluster] = SortedList()
