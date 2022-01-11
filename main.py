import pandas as pd
import numpy as np

from typing import List

import config as cfg

from plotting import plot_2d_dataframe_by_class, plot_2d_dataframe_by_dataset, plot_categorical_data_clusters
from rock import DistanceRock, CategoricalRock
from data_loader import DataLoader


def test():
    np.random.seed(42)

    loader: DataLoader = DataLoader(f'{cfg.DATA_PATH}/2d-3c-no123.arff')
    data_df: pd.DataFrame = loader.load_from_arff()
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

    cat_loader: DataLoader = DataLoader('agaricus-lepiota.data')
    cat_data: np.ndarray = cat_loader.load_from_csv()
    cat_data_labels: np.ndarray = np.asarray(cat_data[:, 0])
    cat_data_array: np.ndarray = np.asarray(cat_data[:, 1:])

    cat_rock: CategoricalRock = CategoricalRock(cat_data_array, 0.25, 20, 0.8)
    cat_rock.run()

    cat_clusters: List[List[int]] = cat_rock.result
    cat_data_df: pd.DataFrame = pd.DataFrame(columns=[cfg.DATA_TARGET, cfg.CATEGORICAL_DATA_ATTRIBUTE])
    for i, cluster in enumerate(cat_clusters):
        labels_in_cluster: np.ndarray = cat_data_labels[cluster]
        for k in range(labels_in_cluster.shape[0]):
            cat_data_df = cat_data_df.append(
                {
                    cfg.DATA_TARGET: labels_in_cluster[k],
                    cfg.CATEGORICAL_DATA_ATTRIBUTE: i
                }, ignore_index=True)
        print(f'Cluster {i}:\n {np.asarray(np.unique(labels_in_cluster, return_counts=True)).T}')
    plot_categorical_data_clusters(cat_data_df, 'test')


if __name__ == '__main__':
    test()
