import pandas as pd
import numpy as np

from scipy.io import arff
from typing import List

import config as cfg

from plotting import plot_2d_dataframe_by_class, plot_2d_dataframe_by_dataset, plot_categorical_data_clusters
from rock import DistanceRock, CategoricalRock


def test():
    np.random.seed(42)
    data = arff.loadarff(f'{cfg.DATA_PATH}/2d-3c-no123.arff')
    data_df: pd.DataFrame = pd.DataFrame(data[0])
    data_array: np.ndarray = data_df[cfg.DATA_ATTRIBUTES].values

    plot_2d_dataframe_by_class(data_df, 'test4')

    rock: DistanceRock = DistanceRock(data_array, 0.5, 3, 0.0, 0.6)

    rock.run()

    clusters: List[List[int]] = rock.result

    cluster_data: List[pd.DataFrame] = []
    for cluster in clusters:
        data_indices = data_df.index.isin(cluster)
        cluster_data.append(data_df[data_indices])

    concatenated_df: pd.DataFrame = pd.concat([c.assign(dataset=f'c{i}') for (i, c) in enumerate(cluster_data)])

    plot_2d_dataframe_by_dataset(concatenated_df, 'test4')

    cat_data = np.loadtxt("agaricus-lepiota.data", dtype=str, delimiter=",")
    cat_data_labels = np.asarray(cat_data[:, 0])
    cat_data_array = np.asarray(cat_data[:, 1:])

    cat_rock = CategoricalRock(cat_data_array, 0.1, 20, 0.8)

    cat_data_df = pd.DataFrame(columns=[cfg.DATA_TARGET, cfg.CATEGORICAL_DATA_ATTRIBUTE])
    cat_rock.run()
    res = cat_rock.result
    for i, cluster in enumerate(res):
        d = cat_data_labels[cluster]
        for j in range(d.shape[0]):
            cat_data_df = cat_data_df.append({cfg.DATA_TARGET: d[j], cfg.CATEGORICAL_DATA_ATTRIBUTE: i}, ignore_index=True)
        labels_cluster_data = cat_data_labels[cluster]
        unique, counts = np.unique(labels_cluster_data, return_counts=True)
        print(f'Cluster {i}: {np.asarray((unique, counts)).T}')
    plot_categorical_data_clusters(cat_data_df, 'test')


if __name__ == '__main__':
    test()
