from datetime import datetime
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_traj_start_end_time(trajectory):
    start = trajectory["timestamp"].iloc[0]
    end = trajectory["timestamp"].iloc[len(trajectory)-1]
    traj = trajectory["traj"].iloc[0]

    return pd.DataFrame({"traj": [traj], "start": [start], "end": [end]})

def get_trajectories_time_window(df):
    grouped_df = df.groupby(by = ["traj"], as_index=False).apply(compute_traj_start_end_time)

    return grouped_df


def compute_traj_average_speed(trajectory):
    timestamps = pd.to_datetime(trajectory['timestamp'], format='%Y-%m-%d %H:%M:%S')

    vx = trajectory["lat"].diff()
    vy = trajectory["long"].diff()
    vt = timestamps.diff().dt.total_seconds()
    mean_traj_speed = (np.sqrt((vx ** 2) + vy ** 2)/vt).replace(np.inf, 0).mean()
    transport = trajectory["transport"].iloc[0]
    traj = trajectory["traj"].iloc[0]
    dist = (np.sqrt((vx ** 2) + vy ** 2)).replace(np.inf, 0).sum()

    return pd.DataFrame({"traj": [traj], "mean": [mean_traj_speed], "transport": [transport], "distance": [dist]})

def compute_trajectories(df):
    grouped_df = df.groupby(by = ["traj"], as_index=False).apply(compute_traj_average_speed)

    return grouped_df

if __name__ == '__main__':
    df = compute_trajectories(pd.read_csv("university_area_day_2008_october.csv"))
    df.to_csv("university_area_speed_per_traj_day.csv", index=False)
    # df = get_trajectories_time_window(pd.read_csv("university_area_night_2008_october.csv"))
    # df.to_csv("./TrajectoryData/university_area_traj_start_end_time.csv", index=False)