from turtle import color
from numpy import size
import pandas
import matplotlib.pyplot as plt

def main():
    schema = ["lat", "long", "zeroes", "altitude", "dateNr", "date", "time"]
    for i in range(51):
        print(f"reading: {i}")
        df = pandas.read_table(f"./sample_trajectories/{i+1}.plt", names=schema, quotechar = "\"", skiprows = 7, sep = ",")
        putDFOnPlot(df)
    plt.show()

def putDFOnPlot(df):
    plt.scatter(x=df['long'], y=df['lat'], color="red", s=.3**2)


main()