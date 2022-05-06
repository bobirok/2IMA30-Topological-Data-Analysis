from pickle import TRUE
from turtle import color
from numpy import size
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

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
            if(df is not None):
                df = pd.concat([df.copy(), (pd.read_csv(file, skiprows=6, names=schema))], axis=0, ignore_index=True)
            else:
                df = pd.read_csv(file, skiprows=6, names=schema)
    return df

def main():
    df = read_files()
    x1 = bounding_box["x1"]
    x2 = bounding_box["x2"]
    y1 = bounding_box["y1"]
    y2 = bounding_box["y2"]

    new_df = df.loc[((df["lat"] >= y1) & (df["lat"] <= y2)) & ((df["long"] >= x1) & (df["long"] <= x2))]
    print(len(df), len(new_df))


def putDFOnPlot(df):
    plt.scatter(x=df['long'], y=df['lat'], color="red", s=.3**2)


main()