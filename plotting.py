import pandas as pd
import seaborn as sns

import config as cfg


def plot_2d_dataframe_by_class(df: pd.DataFrame, filename: str) -> None:
    plot = sns.scatterplot(data=df, x=cfg.DATA_ATTRIBUTES[0], y=cfg.DATA_ATTRIBUTES[1], hue=cfg.DATA_TARGET)
    plot.get_figure().savefig(f'{cfg.TARGET_PATH}/{filename}')


def plot_2d_dataframe_by_dataset(df: pd.DataFrame, filename: str) -> None:
    plot = sns.scatterplot(data=df, x=cfg.DATA_ATTRIBUTES[0], y=cfg.DATA_ATTRIBUTES[1], hue='dataset', legend='brief')
    plot.get_figure().savefig(f'{cfg.OUTPUT_PATH}/{filename}')
