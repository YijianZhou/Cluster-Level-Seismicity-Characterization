import os, glob
from sklearn.cluster import DBSCAN
import numpy as np
from obspy import UTCDateTime

# i/o paths
fctlg = 'input/nanometrics-delaware-basin_reloc_2019-2024.csv'
fout = 'output/db-seis_1-init-cluster_2-20.npy'
# clustering params
ot_rng = '20190401-20250501'
ot_rng = [UTCDateTime(code) for code in ot_rng.split('-')]
lon_rng =  [-104.8, -102.9]
lat_rng = [30.75, 32.8]
dep_rng = [-1,11]
mag_rng = [0.3, 6.0]
lon0, lat0 = -104., 31.5
cos_lat = np.cos(lat0*np.pi/180)
eps = 2
min_samples=20

# read fctlg
print('read catalog file')
events_org, events_euc = [],[]
f=open(fctlg); lines=f.readlines(); f.close()
for evid,line in enumerate(lines):
    codes = line.split(',')
    ot = UTCDateTime(codes[0])
    lat, lon, dep, mag = [float(code) for code in codes[1:5]]
    if not (ot_rng[0]<=ot<=ot_rng[1] and lon_rng[0]<lon<lon_rng[1] and lat_rng[0]<lat<lat_rng[1] and dep_rng[0]<dep<dep_rng[1] and mag_rng[0]<=mag<=mag_rng[1]): continue
    distx, disty = 111*(lon-lon0)*cos_lat, 111*(lat-lat0)
    events_org.append([ot, lat, lon, dep, mag, evid])
    events_euc.append([distx, disty, dep])
events_org = np.array(events_org)
events_euc = np.array(events_euc)
print(len(events_org), 'events')

print('start DBSCAN clustering')
clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
clusterer.fit(events_euc)
num_clust = np.amax(clusterer.labels_) + 1
print('%s clusters found'%num_clust)

events_clustered = []
for clust_id in range(num_clust):
    events_clustered.append(events_org[np.where(clusterer.labels_==clust_id)])
events_clustered.append(events_org[np.where(clusterer.labels_==-1)])
print('%s noise events'%len(events_clustered[-1]))
np.save(fout, np.array(events_clustered, dtype=object), allow_pickle=True)

