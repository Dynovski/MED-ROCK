import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

import config as cfg


def plot_2d_dataframe_by_class(df: pd.DataFrame, filename: str) -> None:
    sns.scatterplot(data=df, x=cfg.DATA_ATTRIBUTES[0], y=cfg.DATA_ATTRIBUTES[1], hue=cfg.DATA_TARGET, legend=False)
    plt.savefig(f'{cfg.TARGET_PATH}/{filename}')
    plt.clf()


def plot_2d_dataframe_by_dataset(df: pd.DataFrame, filename: str) -> None:
    sns.scatterplot(data=df, x=cfg.DATA_ATTRIBUTES[0], y=cfg.DATA_ATTRIBUTES[1], hue='dataset', legend=False)
    plt.savefig(f'{cfg.OUTPUT_PATH}/{filename}')
    plt.clf()


def plot_categorical_data_clusters(df: pd.DataFrame, filename: str) -> None:
    sns.catplot(data=df, y=cfg.CATEGORICAL_DATA_ATTRIBUTE, hue=cfg.DATA_TARGET, kind='count', height=30, aspect=1)
    plt.savefig(f'{cfg.CATEGORICAL_PLOTS_PATH}/{filename}')
    plt.clf()
