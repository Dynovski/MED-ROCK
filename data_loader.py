import numpy as np
import pandas as pd

from scipy.io import arff


class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_from_arff(self) -> pd.DataFrame:
        data = arff.loadarff(self.path)
        return pd.DataFrame(data[0])

    def load_from_csv(self) -> np.ndarray:
        return np.loadtxt(self.path, dtype=str, delimiter=",")
