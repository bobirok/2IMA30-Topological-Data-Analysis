import kmapper as km
import pandas as pd
import sklearn
import haversine as hs
from datetime import datetime
import numpy as np


# df = pd.read_csv("test_data_filtered.csv")
#
# time_str = df['timestamp'][0]
# time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
#
# print(time)
# print(type(time))
#
# new_time = datetime.strptime(df['timestamp'][1], '%Y-%m-%d %H:%M:%S')
# print(new_time)
# diff = new_time - time
# print(diff)
# diff_in_hours = diff.total_seconds() / 3600
# print('Difference between two datetimes in hours:')
# print(diff_in_hours)

a = np.asarray([ [1.3454235,2,'hello'], [4,5,'world'], [7,8,'!!!'] ])
print(a)
np.savetxt("foo.csv", a,  fmt="%s, %s, %s", delimiter=",")

b = np.genfromtxt('foo.csv', delimiter=',')
print(b)
print(b[0][0])
print(type(b[0][0]))

