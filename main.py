import pandas as pd
import numpy as np

from scipy.io import arff
from typing import List
from sortedcontainers import SortedList

import config as cfg

from plotting import plot_2d_dataframe_by_class, plot_2d_dataframe_by_dataset
from rock import Rock


def test():
    np.random.seed(42)
    data = arff.loadarff(f'{cfg.DATA_PATH}/2d-3c-no123.arff')
    print('Loaded data')
    data_df: pd.DataFrame = pd.DataFrame(data[0])
    data_array: np.ndarray = data_df[cfg.DATA_ATTRIBUTES].values

    plot_2d_dataframe_by_class(data_df, 'test4')

    rock: Rock = Rock(data_array, 1.0, 3, 0.6, 0.3)

    rock.run()

    clusters: List[List[int]] = rock.result

    cluster_data: List[pd.DataFrame] = []
    for cluster in clusters:
        data_indices = data_df.index.isin(cluster)
        cluster_data.append(data_df[data_indices])

    concatenated_df: pd.DataFrame = pd.concat([c.assign(dataset=f'c{i}') for (i, c) in enumerate(cluster_data)])

    plot_2d_dataframe_by_dataset(concatenated_df, 'test4')


if __name__ == '__main__':
    test()
