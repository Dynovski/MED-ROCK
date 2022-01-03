import pandas as pd
import numpy as np

from scipy.io import arff
from typing import List
from sortedcontainers import SortedList

import config as cfg

from plotting import plot_2d_dataframe, plot_2d_result
from rock import Rock
from cluster import Cluster


def test():
    data = arff.loadarff(f'{cfg.DATA_PATH}/2d-3c-no123.arff')
    data_df: pd.DataFrame = pd.DataFrame(data[0])
    data_array: np.ndarray = data_df[cfg.DATA_ATTRIBUTES].values.tolist()

    plot_2d_dataframe(data_df, 'test')

    rock: Rock = Rock(data_array, data_array.shape[0], 3, 0.6, 0.6)

    rock.run()

    clusters: SortedList[Cluster] = rock.result

    cluster_data: List[pd.DataFrame] = []
    for cluster in clusters:
        data_indices = data_df.index.isin(cluster.data_indices)
        cluster_data.append(data_df[data_indices])

    concatenated_df: pd.DataFrame = pd.concat([c.append(dataset=f'c{i}') for (i, c) in enumerate(cluster_data)])
    print(concatenated_df.head())

    plot_2d_result(concatenated_df, 'test')


if __name__ == '__main__':
    test()
