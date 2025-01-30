import os

from msticpy.vis.data_viewer import DataViewer

import pandas as pd

if os.path.exists("results.pkl"):
    df = pd.read_pickle("results.pkl")

print("null percents", df.isnull().sum() * 100 / len(df), sep="\n")

x = DataViewer(df)
