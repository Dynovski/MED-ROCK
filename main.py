import pandas as pd
import numpy as np

from typing import List
from multiprocessing import Pool

import config as cfg

from plotting import plot_2d_dataframe_by_class, plot_2d_dataframe_by_dataset, plot_categorical_data_clusters
from rock import DistanceRock, CategoricalRock
from data_loader import DataLoader


def test_nominal(test_name: str, num_clusters: int, threshold: float, max_distance: float) -> None:
    loader: DataLoader = DataLoader(f'{cfg.DATA_PATH}/{test_name}')
    data_df: pd.DataFrame = loader.load_from_arff()
    data_array: np.ndarray = data_df[cfg.DATA_ATTRIBUTES].values
    plot_2d_dataframe_by_class(
        data_df,
        f'ncl_{num_clusters}thr_{threshold}dst_{max_distance}__{test_name}'.replace('.', '')[:-4]
    )

    rock: DistanceRock = DistanceRock(data_array, 0.5, num_clusters, threshold, max_distance)
    rock.run()

    clusters: List[List[int]] = rock.result
    cluster_data: List[pd.DataFrame] = []
    for cluster in clusters:
        data_indices = data_df.index.isin(cluster)
        cluster_data.append(data_df[data_indices])
    concatenated_df: pd.DataFrame = pd.concat([c.assign(dataset=f'c{i}') for (i, c) in enumerate(cluster_data)])
    plot_2d_dataframe_by_dataset(
        concatenated_df,
        f'ncl_{num_clusters}thr_{threshold}dst_{max_distance}__{test_name}'.replace('.', '')[:-4]
    )


def test_categorical(test_name: str, num_clusters: int, threshold: float) -> None:
    loader: DataLoader = DataLoader(f'{cfg.DATA_PATH}/{test_name}')
    data: np.ndarray = loader.load_from_csv()
    labels_array: np.ndarray = np.asarray(data[:, 0])
    data_array: np.ndarray = np.asarray(data[:, 1:])

    rock: CategoricalRock = CategoricalRock(data_array, 0.25, num_clusters, threshold)
    rock.run()

    clusters: List[List[int]] = rock.result
    for i, cluster in enumerate(clusters):
        labels_in_cluster: np.ndarray = labels_array[cluster]
        print(f'Cluster {i}:\n {np.asarray(np.unique(labels_in_cluster, return_counts=True)).T}')


if __name__ == '__main__':
    np.random.seed(42)
    nominal_data = []
    for nominal_test, num_clusters in zip(cfg.NOMINAL_TEST_FILENAMES, cfg.NOMINAL_NUM_CLUSTERS):
        for threshold in cfg.THRESHOLDS:
            for max_distance in cfg.DISTANCES:
                nominal_data.append((nominal_test, num_clusters, threshold, max_distance))
    with Pool() as pool:
        pool.starmap(test_nominal, nominal_data)

    for categorical_test, num_clusters in zip(cfg.CATEGORICAL_TEST_FILENAMES, cfg.CATEGORICAL_NUM_CLUSTERS):
        test_categorical(categorical_test, num_clusters, 0.8)
