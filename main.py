import pandas as pd
import numpy as np

from typing import List, Optional

import config as cfg

from plotting import plot_2d_dataframe_by_class, plot_2d_dataframe_by_dataset
from rock import DistanceRock, CategoricalRock
from data_loader import DataLoader


def run_nominal(test_name: str, num_clusters: int, threshold: float, max_distance: float, ratio: float) -> None:
    loader: DataLoader = DataLoader(f'{cfg.NOMINAL_DATA_PATH}/{test_name}')
    data_df: pd.DataFrame = loader.load_from_arff()
    data_array: np.ndarray = data_df.iloc[:, :-1].values
    plot_2d_dataframe_by_class(
        data_df,
        f'{test_name}'.replace('.', '')[:-4]
    )

    rock: DistanceRock = DistanceRock(data_array, ratio, num_clusters, threshold, max_distance)
    rock.run()

    clusters: List[List[int]] = rock.result
    cluster_data: List[pd.DataFrame] = []
    for cluster in clusters:
        data_indices = data_df.index.isin(cluster)
        cluster_data.append(data_df[data_indices])
    concatenated_df: pd.DataFrame = pd.concat([c.assign(dataset=f'c{i}') for (i, c) in enumerate(cluster_data)])
    plot_2d_dataframe_by_dataset(
        concatenated_df,
        f'CL_{num_clusters}THR_{threshold}MD_{max_distance}__{test_name}'.replace('.', '')[:-4]
    )


def run_categorical(test_name: str, num_clusters: int, threshold: float, label_first: bool, ratio: float) -> None:
    loader: DataLoader = DataLoader(f'{cfg.CATEGORICAL_DATA_PATH}/{test_name}')
    data: np.ndarray = loader.load_from_csv()
    labels_array: Optional[np.ndarray] = None
    data_array: Optional[np.ndarray] = None
    if label_first:
        labels_array: np.ndarray = np.asarray(data[:, 0])
        data_array: np.ndarray = np.asarray(data[:, 1:])
    else:
        labels_array: np.ndarray = np.asarray(data[:, -1])
        data_array: np.ndarray = np.asarray(data[:, :-1])

    rock: CategoricalRock = CategoricalRock(data_array, ratio, num_clusters, threshold)
    rock.run()

    clusters: List[List[int]] = rock.result
    for i, cluster in enumerate(clusters):
        labels_in_cluster: np.ndarray = labels_array[cluster]
        print(f'Cluster {i}:\n {np.asarray(np.unique(labels_in_cluster, return_counts=True)).T}')


if __name__ == '__main__':
    if cfg.IS_NOMINAL:
        run_nominal(cfg.DATA_FILENAME, cfg.NUM_CLUSTERS, cfg.THETA, cfg.MAX_DISTANCE, cfg.RATIO)
    else:
        run_categorical(cfg.DATA_FILENAME, cfg.NUM_CLUSTERS, cfg.THETA, cfg.IS_LABEL_FIRST, cfg.RATIO)
