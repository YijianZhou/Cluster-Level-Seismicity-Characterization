import os, glob
import numpy as np

# i/o paths
min_cluster_size_list = list(range(20,101,20))
min_samples_list = [10, None]
fclust_list = []
for min_samples in min_samples_list:
  for min_cluster_size in min_cluster_size_list:
    fclust_list.append('output/db-seis_2-hdbscan-cluster_%s-%s.npy'%(min_samples, min_cluster_size))
fout_npy = 'output/db-seis_3-merged-cluster.npy'
fout_csv = open('output/db-seis_3-merged-cluster.csv','w')
min_events = 50

def merge_clusters(cluster_set1, cluster_set2):
    # format checks
    if len(cluster_set1[0][0]) != 6: print('f1 wrong format')
    if len(cluster_set2[0][0]) != 6: print('f2 wrong format')
    # ---- flatten inputs into one list of clusters ----
    clusters = list(cluster_set1) + list(cluster_set2)
    n = len(clusters)
    # ---- Union-Find (Disjoint Set Union) ----
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # ---- union clusters that share any evid (order-independent) ----
    evid_owner = {}  # evid -> cluster_index that first had it
    for i, cl in enumerate(clusters):
        # evid is last column
        evids = cl[:, -1]
        for evid in evids:
            # keep evid hashable/stable (int if possible, else str)
            try:
                key = int(evid)
            except Exception:
                key = str(evid)

            j = evid_owner.get(key)
            if j is None:
                evid_owner[key] = i
            else:
                union(i, j)

    # ---- group clusters by connected component ----
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    merged = []
    for idxs in groups.values():
        # merge raw rows
        merged_cluster = np.vstack([clusters[i] for i in idxs])

        # ---- de-duplicate events inside merged cluster ----
        # If evid is truly unique per event, dedup-by-evid is the cleanest and fastest.
        uniq = {}
        for ev in merged_cluster:
            evid = ev[-1]
            try:
                key = int(evid)
            except Exception:
                key = str(evid)
            # keep the first occurrence (or replace if you prefer latest)
            if key not in uniq:
                uniq[key] = ev

        merged.append(np.array(list(uniq.values()), dtype=object))

    # return as object array of clusters (ragged)
    return np.array(merged, dtype=object)


events_clusters_merged = np.load(fclust_list[0], allow_pickle=True)[:-1]
for fclust in fclust_list[1:]:
    print('merging %s'%fclust)
    events_clusters = np.load(fclust, allow_pickle=True)[:-1]
    print('number of clusters:', len(events_clusters), len(events_clusters_merged))
    events_clusters_merged = merge_clusters(events_clusters_merged, events_clusters)
np.save(fout_npy, events_clusters_merged)

# write csv file
clust_id = 0
for events in events_clusters_merged:
    if len(events)<min_events: continue
    fout_csv.write('# cluster %s\n'%clust_id)
    for [ot, lat, lon, dep, mag, evid] in events:
        fout_csv.write('%s,%s,%s,%s,%s,%s\n'%(ot, lat, lon, dep, mag, evid))
    clust_id += 1
fout_csv.close()

