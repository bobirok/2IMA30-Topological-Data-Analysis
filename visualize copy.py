from datetime import datetime
from pickle import TRUE
from turtle import color
from numpy import size
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# Beijing bounding box
# bounding_box = {
#     "x1": 116.21200561523438,
#     "x2": 116.54159545898438,
#     "y1": 39.77265852521458,
#     "y2": 40.02340800226773
# }

# University bounding box
bounding_box = {
    "x1": 116.27655029296875,
    "x2":  116.3723373413086,
    "y1": 39.98027708862265,
    "y2": 40.024459635387906
}

def read_files():
    schema = ["lat", "long", "zeroes", "altitude", "dateNr", "date", "time"]
    path = os.getcwd()

    df = None
    for i in range(161, 163):
        directory = './Data/{:03d}/Trajectory'.format(i)
        csv_files = glob.glob(os.path.join(os.path.join(path, directory), "*.plt"))
        print("User {} has {} files".format(i, len(csv_files)))
        for file in csv_files:
            if df is not None:
                df = pd.concat([df.copy(), (pd.read_csv(file, skiprows=6, names=schema))], axis=0, ignore_index=True)
            else:
                df = pd.read_csv(file, skiprows=6, names=schema)
    return df

def read_combined_data():
    schema = ["traj", "lat", "long", "altitude", "timestamp", "user", "transport"]

    df = pd.read_csv('test_data.csv', skiprows=1, names=schema)
    return df
    
def main():
    df = read_combined_data()
    x1 = bounding_box["x1"]
    x2 = bounding_box["x2"]
    y1 = bounding_box["y1"]
    y2 = bounding_box["y2"]

    new_df = df.loc[((df["lat"] >= y1) & (df["lat"] <= y2)) & ((df["long"] >= x1) & (df["long"] <= x2))]

    # & (df['transport'] == 'car') | (df['transport'] == 'walk')

    new_df.to_csv(r'university_area.csv', index=False)

    # put_df_on_plot(new_df)
    heatmap(new_df)

def put_df_on_plot(df):
    plt.scatter(x=df['long'], y=df['lat'], color="red", s=.3**2)

# Can get for specific day, month or year
def filter_by_date(df, date):
    date_df = df[df["timestamp"].str.contains(date)]
    return date_df

def get_night_hours_for_date(df, date):
    start_night_hour = 0
    end_night_hour = 8
    date_df = filter_by_date(df,date)
    date_df["timestamp"] = pd.to_datetime(date_df["timestamp"])
    night_hours_df = date_df.loc[(date_df["timestamp"].dt.hour > start_night_hour) & (date_df["timestamp"].dt.hour <= end_night_hour)]
    return night_hours_df

def get_day_hours_for_date(df, date):
    start_day_hour = 8
    end_day_hour = 23
    date_df = filter_by_date(df,date)
    date_df["timestamp"] = pd.to_datetime(date_df["timestamp"])
    day_hours_df = date_df.loc[(date_df["timestamp"].dt.hour >= start_day_hour) & (date_df["timestamp"].dt.hour < end_day_hour)]
    return day_hours_df

def get_week_day_of_month(df, week_day):
    day_df = df.loc[(pd.to_datetime(df["timestamp"]).dt.dayofweek == week_day)]
    return day_df

def heatmap(df):
    # df = df.loc[(df['transport'] == 'car')]
    # H = plt.hist2d((df['long']).astype(float), (df['lat']).astype(float), bins=280)[0]
    # H = np.flip(H.T, 0)
    # fig = plt.figure(figsize=(50, 50))
    # H = np.ma.masked_where(H == 0, H)
    # cmap = plt.get_cmap('cool')
    # cmap.set_bad(color='black')
    # plt.imshow(H, cmap=cmap)
    # plt.show()
    plt.scatter(df['long'], df['lat'])
    plt.show()

# main()
if __name__ == '__main__':
    # df = pd.read_csv("university_area_day_2008_october.csv")
    # friday_df = get_week_day_of_month(df, 4)
    
    df_day_october = get_day_hours_for_date(pd.read_csv("university_area.csv"), "2008-10")
    df_day_october.to_csv(r'university_area_day_2008_october.csv', index=False)

    df_night_october = get_night_hours_for_date(pd.read_csv("university_area.csv"), "2008-10")
    df_night_october.to_csv(r'university_area_night_2008_october.csv', index=False)