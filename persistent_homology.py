import numpy as np
import pandas as pd
from ripser import ripser
from ripser import Rips
from persim import plot_diagrams
import matplotlib.pyplot as plt
from visualize import get_week_day_of_month

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def compute_persistent_homology(df):
    friday_df = get_week_day_of_month(df, 6)
    friday_df = friday_df.drop(columns=["traj", "user", "transport", "altitude"])
    friday_df["timestamp"] = (pd.to_datetime(friday_df["timestamp"]).dt.hour) * 0.085
    
    data = friday_df.to_numpy()
    diagrams = ripser(data, n_perm=2000, metric="manhattan")
    plot_diagrams(diagrams["dgms"], show=True)
    indices = diagrams["idx_perm"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data[indices, 0], data[indices, 1], data[indices, 2], c=data[indices, 2], cmap="copper")
    plt.show()

df = pd.read_csv("university_area_day_2008_october.csv").dropna()
compute_persistent_homology(df)