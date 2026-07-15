import pandas as pd

from . import config


def load_housing_panel(path=None):
    return pd.read_csv(path or config.EXTERNAL / "housing_panel.csv")
