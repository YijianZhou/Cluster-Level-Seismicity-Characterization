import os
import numpy as np
from obspy import UTCDateTime

# i/o paths
fclust = 'output/db-seis_4-manual-cluster_Nmin-50.csv'
fout = 'output/db-seis_4-manual-cluster_Nmin-50.npy'
dtype = [('ot','O'),('lat','O'),('lon','O'),('dep','O'),('mag','O'),('evid','O')]

events_clustered = []
f=open(fclust); lines=f.readlines(); f.close()
for line in lines:
    if line[0]=='#': events_clustered.append([]); continue
    codes = line.split(',')
    ot = UTCDateTime(codes[0])
    lat, lon, dep, mag = [float(code) for code in codes[1:5]]
    #if mag>5: print(line[:-1])
    evid = int(codes[-1])
    events_clustered[-1].append([ot, lat, lon, dep, mag, evid])
#events_clustered = [np.array(clust, dtype=dtype) for clust in events_clustered]
events_clustered = [np.array(clust) for clust in events_clustered]

num_evt = np.array([len(clust) for clust in events_clustered])
print(fclust)
print('num events & clusters', np.sum(num_evt), len(events_clustered))
print('N>100, 500, 1000, 5000, 10000', np.sum(num_evt>100), np.sum(num_evt>500), np.sum(num_evt>1000), np.sum(num_evt>5000), np.sum(num_evt>10000))
np.save(fout, np.array(events_clustered, dtype=object), allow_pickle=True)
