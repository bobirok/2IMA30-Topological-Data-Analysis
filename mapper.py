import kmapper as km
import pandas as pd
import sklearn

df = pd.read_csv("torus_coordinates_1000.csv")

data = df.drop(columns=["timestamp"]).to_numpy()


# Initialize
mapper = km.KeplerMapper(verbose=2)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0,1])


clusterer = sklearn.cluster.DBSCAN(eps=0.3, min_samples=5)
cover = km.Cover(n_cubes=3, perc_overlap=0.3)

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, data, cover=cover, clusterer=clusterer)

labels = data[:, 2]
# Visualize it
mapper.visualize(graph, path_html="make_circles_keplermapper_output.html",
                 title="Torus Map Y", color_values=labels, color_function_name="labels")

if __name__ == '__main__':
    pass
