# import required module
import os
import pandas as pd
import numpy as np
from datetime import datetime


def load_data(folder):
    # assign directory
    directory = folder

    schema = ["traj", "lat", "long", "altitude", "timestamp", "user", "transport"]

    # iterate over files in
    # that directory
    data = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        print(f[-3:])
        labels_present = False
        for file in os.listdir(f):
            f_2 = os.path.join(f, file)
            # checking if it is a file
            if os.path.isfile(f_2):
                labels_present = True
        data.extend(construct_matrix(f, labels_present))
    df = pd.DataFrame(data, columns=schema)
    df.to_csv(r'test_data.csv', index=False)


def construct_labels(path):
    file = os.path.join(path, 'labels.txt')
    convertfunc = lambda x: datetime.strptime(x[2:], '%y/%m/%d %H:%M:%S')
    data = np.genfromtxt(file, dtype=None, delimiter='\t', skip_header=1, encoding=None,
                         converters={0: convertfunc, 1: convertfunc})
    return data


def construct_matrix(path, labels_present):
    trajectories = os.path.join(path, 'Trajectory')
    matrix = []
    labels = []
    label =[]
    if labels_present:
        labels = construct_labels(path)
        if np.size(labels) == 1:
            label = labels.item(0)
        else:
            label = labels[0]
        labels = np.delete(labels, 0)

    for filename in os.listdir(trajectories):
        f = os.path.join(trajectories, filename)
        id = f.split('\\')[-1].split('.')[0]
        user = f.split('\\')[2]
        data = np.genfromtxt(f, dtype=None, delimiter=',', skip_header=6, encoding=None, usecols=(0, 1, 3, 5, 6))
        for i in data:
            time_str = i[3][2:].replace('-', '/') + ' ' + i[4]
            timestamp = datetime.strptime(time_str, '%y/%m/%d %H:%M:%S')
            if labels_present:
                while timestamp > label[1]:
                    if np.size(labels) == 0:
                        label = [label[0], label[1],'unknown']
                        labels_present = False
                        break
                    label = labels[0]
                    labels = np.delete(labels, 0)
                if timestamp < label[0]:
                    row = [id, i[0], i[1], i[2], timestamp, user, 'unknown']
                else:
                    row = [id, i[0], i[1], i[2], timestamp, user, label[2]]
            else:
                row = [id, i[0], i[1], i[2], timestamp, user, 'unknown']
            matrix.append(row)
    return matrix


load_data('Geolife Trajectories 1.3\Data')