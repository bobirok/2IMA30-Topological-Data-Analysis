import os
import kmapper as km
import pandas as pd
import sklearn
from sklearn import datasets
import haversine as hs
from datetime import datetime
import numpy as np
from tqdm import tqdm


def calc_speed(pointA, pointB):
    distance = hs.haversine((pointA[0],pointA[1]), (pointB[0], pointB[1]))

    time = datetime.strptime(pointA[2], '%Y-%m-%d %H:%M:%S')
    new_time = datetime.strptime(pointB[2], '%Y-%m-%d %H:%M:%S')
    diff = new_time - time
    delta_t = diff.total_seconds() / 3600

    if delta_t != 0:
        speed = distance/delta_t
    else:
        speed = 0
    return speed


def create_lenses(df):
    X = np.empty((0, 4))
    row = 0
    p = df.iloc[row]
    point = [p['lat'], p['long'], p['timestamp']]
    speed = []
    row = 1
    pbar = tqdm(total=len(df) + 1)
    while row < len(df):
        new_p = df.iloc[row]
        new_point = [new_p['lat'], new_p['long'], new_p['timestamp']]
        if p['traj'] == new_p['traj'] and p['transport'] == new_p['transport'] and row != len(df)-1:
            speed.append(calc_speed(point, new_point))
        else:
            s = np.array([speed])
            X = np.append(X, np.array([[p['traj'], np.mean(s), np.std(s), str(p['user'])+'_'+str(p['transport'])]]), axis=0)
            p = new_p
            speed = []

        point = new_point
        row += 1
        pbar.update(1)
    pbar.close()
    return X


data_path = "lens_data.csv"
if os.path.exists(data_path):
    #data = np.genfromtxt(data_path, delimiter=',',dtype=None) #, dtype=None
    data = pd.read_csv("lens_data.csv", names=['traj', 'mean', 'std', 'label'])
else:
    print('load has failed')
    data = pd.read_csv("test_data_filtered.csv")
    data = create_lenses(data)
    np.savetxt(data_path, data,  fmt="%s, %s, %s, %s", delimiter=",")

print(data)
#

data_names = data['label'].to_numpy()
# data = data[:, [0, 1, 2]]

# print(lens1)

test_data = data[['mean', 'std']].astype({"mean": float, 'std': float}, errors='raise').to_numpy()
test_data = np.nan_to_num(test_data)
lens1 = test_data[:, [0, 1]]
print(data_names)
print(type(data_names[0]))

# Initialize
mapper = km.KeplerMapper(verbose=2)
lens2 = mapper.fit_transform(test_data, projection=[0,1])

# Fit to and transform the data
#projected_data = mapper.fit_transform(data, projection=[0,1])
# lens = np.c_[lens1, lens2]

clusterer = sklearn.cluster.DBSCAN(eps=1, min_samples=2)
#clusterer=sklearn.cluster.KMeans(n_clusters=2,random_state=3471)
cover = km.Cover(n_cubes=15, perc_overlap=0.7)

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(lens1, test_data, cover=cover, clusterer=clusterer)

labels = test_data[:, 0]
print(graph)
# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                 title="Torus Map Y", color_values=labels, color_function_name="labels", custom_tooltips=data_names, X=test_data, X_names=['mean', 'std'])

if __name__ == '__main__':
    pass
