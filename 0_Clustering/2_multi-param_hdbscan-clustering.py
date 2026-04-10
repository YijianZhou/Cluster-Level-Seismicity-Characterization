""" Install HDBSCAN 
"""
import os, glob
from hdbscan import HDBSCAN
#from sklearn.cluster import HDBSCAN
import numpy as np
from obspy import UTCDateTime

# i/o paths
fctlg = 'output/db-seis_1-init-cluster_2-20.npy'
fout_root = 'output/db-seis_2-hdbscan-cluster'
# clustering params
lon0, lat0 = -104., 31.5
cos_lat = np.cos(lat0*np.pi/180)
min_cluster_size_list = list(range(20,101,20))
min_samples_list = [10, None]

# read fctlg
print('read catalog file')
events_clustered = np.load(fctlg, allow_pickle=True)
events_org = np.concatenate(events_clustered[:-1], axis=0)
lat = events_org[:, 1].astype(float)
lon = events_org[:, 2].astype(float)
dep = events_org[:, 3].astype(float)
distx = 111.0 * (lon - lon0) * cos_lat
disty = 111.0 * (lat - lat0)
events_euc = np.column_stack((distx, disty, dep))  # (N,3)
print(len(events_org), 'events')

print('start multi-param HDBSCAN clustering')
for min_samples in min_samples_list:
  for min_cluster_size in min_cluster_size_list:
    fout = fout_root+'_%s-%s.npy'%(min_samples, min_cluster_size)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    clusterer.fit(events_euc)
    num_clust = np.amax(clusterer.labels_) + 1
    print('HDBSCAN %s %s: %s clusters found'%(min_samples, min_cluster_size, num_clust))
    events_clustered = []
    for clust_id in range(num_clust):
        events_clustered.append(events_org[np.where(clusterer.labels_==clust_id)])
    events_clustered.append(events_org[np.where(clusterer.labels_==-1)])
    np.save(fout, np.array(events_clustered, dtype=object), allow_pickle=True)

