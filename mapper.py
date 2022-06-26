
import kmapper as km
import pandas as pd
import sklearn
import numpy as np

df = pd.read_csv("university_area_speed_per_traj_day.csv").dropna()

df = df[df["transport"] != "unknown"]

# data = df.drop(columns=["transport"]).to_numpy()
# transports = {'unknown': 1, 'taxi': 2, 'bus': 3, 'train': 4, 'bike': 5, 'car': 6, 'walk': 7, 'subway': 8, 'run': 9}
# df['transport'] = df['transport'].apply(lambda x: transports[x])
# df["traj"] = (df["traj"] - np.min(df["traj"]))/(np.max(df["traj"]) - np.min(df["traj"]))
df["distance"] = (df["distance"] - np.min(df["distance"]))/(np.max(df["distance"]) - np.min(df["distance"]))

data = df.drop(columns=["transport", "traj"]).to_numpy()
print(df)

# Initialize
mapper = km.KeplerMapper(verbose=2)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0])
print(projected_data)

# clusterer = sklearn.cluster.DBSCAN(eps=1, min_samples=1)
clusterer = sklearn.cluster.KMeans(n_clusters=4)
# clusterer = sklearn.cluster.DBSCAN(eps=0.3, min_samples=4)

cover = km.Cover(n_cubes=6, perc_overlap=0.5)

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data, cover=cover, clusterer=clusterer)

# labels = data[:, 2]
labels = np.array(df["transport"].tolist(), dtype=object)
colors = []
for cluster in graph['nodes']:
    colors.append(np.nanmean(graph['nodes'][cluster]))
    #print(np.nanmean(graph['nodes'][cluster]))

# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                 title="Cylinder Map Y", custom_tooltips=labels, color_function_name="colors", color_values=projected_data)

if __name__ == '__main__':
    pass