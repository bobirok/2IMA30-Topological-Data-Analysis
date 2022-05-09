from pickle import TRUE
from turtle import color
from numpy import size
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import seaborn

bounding_box = {
    "x1": 116.21200561523438,
    "x2": 116.54159545898438,
    "y1": 39.77265852521458,
    "y2": 40.02340800226773
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

    new_df = df.loc[((df["lat"] >= y1) & (df["lat"] <= y2)) & ((df["long"] >= x1) & (df["long"] <= x2)) &
                    (df['transport'] == 'car')]

    new_df.to_csv(r'test_data_filtered.csv', index=False)

    # put_df_on_plot(new_df)
    heatmap(new_df)



def put_df_on_plot(df):
    plt.scatter(x=df['long'], y=df['lat'], color="red", s=.3**2)


def heatmap(df):
    H = plt.hist2d(df['long'], df['lat'], bins=280)[0]
    H = np.flip(H.T, 0)
    plt.close()
    fig = plt.figure(figsize=(50, 50))
    H = np.ma.masked_where(H == 0, H)
    cmap = plt.get_cmap('cool')
    cmap.set_bad(color='black')
    plt.imshow(H, cmap=cmap)
    plt.show()



main()
