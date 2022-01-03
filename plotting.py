import pandas as pd
import seaborn as sns

import config as cfg


def plot_2d_dataframe(df: pd.DataFrame, filename: str) -> None:
    plot = df.plot.scatter(x=cfg.DATA_ATTRIBUTES[0], y=cfg.DATA_ATTRIBUTES[1], c=cfg.DATA_TARGET, colormap='viridis')
    plot.figure.savefig(f'{cfg.TARGET_PATH}/{filename}')


def plot_2d_result(df: pd.DataFrame, filename: str) -> None:
    plot = sns.scatterplot(x=cfg.DATA_ATTRIBUTES[0], y=cfg.DATA_ATTRIBUTES[1], data=df, hue='dataset', legend='brief')
    plot.get_figure().savefig(f'{cfg.OUTPUT_PATH}/{filename}')
