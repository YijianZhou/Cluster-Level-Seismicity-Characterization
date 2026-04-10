""" run plot_cluster3d in MATLAB for manual refinement before this step
"""
import os, glob
from obspy import UTCDateTime

# i/o paths
n_min = 50
ot_rng = '20190401-20250501'
ot_rng = [UTCDateTime(code) for code in ot_rng.split('-')]

fclust_list = sorted(glob.glob('output/manual_clusters/cluster-*'))
fout = open('output/db-seis_4-manual-cluster_Nmin-%s.csv'%n_min,'w')
num_clust = 0

for fclust in fclust_list:
    f=open(fclust); lines=f.readlines(); f.close()
    #if len(lines)<n_min: continue
    lines_to_write = []
    for line in lines: 
        ot = UTCDateTime(line.split(',')[0])
        if ot_rng[0]<=ot<=ot_rng[1]: lines_to_write.append(line)
    if len(lines_to_write)<n_min: continue
    fname = os.path.basename(fclust)
    fout.write('# %s\n'%fname[:-4])
    for line in lines_to_write: fout.write(line)
    num_clust += 1
print('%s clusters selected'%num_clust)
fout.close()
