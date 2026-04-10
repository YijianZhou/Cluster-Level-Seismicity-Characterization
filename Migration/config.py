import numpy as np

# I/O
FCLUST = 'input/db-seis_4-manual-cluster_Nmin-100.csv'
OUTPUT_DIR = 'output/db20'
FOUT_ROOT = f'{OUTPUT_DIR}/db_seis-mig'
FOUT_CSV = f'{OUTPUT_DIR}/db-seis_cluster-mig.csv'

# Optional magnitude threshold while reading
MC = 0.3

# Contour estimation parameters
QUANTILES = np.arange(0.05, 1.00, 0.05)
WIN_LEN_CONTOUR = 100 * 86400          # contour sliding window length in seconds
WIN_STEP_CONTOUR = 10 * 86400          # contour sliding step in seconds
MIN_EVENTS_CONTOUR = 10                # windows with fewer events become blank windows and get zero weight
N_BOOTSTRAP = 200                      # bootstrap realizations for Dxx and s_ij
RANDOM_STATE = 12345                   # deterministic bootstrap seed
N_JOBS = 10                          # parallel workers for contour bootstrap; None = auto
S0_FRAC = 0.02                         # weight floor s0 = S0_FRAC * L_ref
DXX_PLOT_REL_UNC_MAX = 0.25            # if set, hide Dxx points with s_ij / L_ref above this threshold in plots

# Migration fitting parameters
MIN_MIG_LEN = 400 * 86400              # minimum migration duration in seconds
WIN_STEP_SLOPE = WIN_STEP_CONTOUR      # sliding step for migration-velocity measurement
NUM_WIN_MIG = int(MIN_MIG_LEN / WIN_STEP_SLOPE) + 1
MIN_FIT_WINDOWS = 10                    # minimum number of valid contour windows needed to fit 4 parameters

# Migration classification thresholds (m/day)
MIN_VELO_EXP = 1.0                     # minimum expansion speed to declare migration front
MIN_VELO_SHR = 1.0                     # maximum allowed relative shrink speed; beyond this, classify as shrink
MAX_VELO_EXP = 10                      # None disables this filter
MAX_OFF_RATIO = None

# Cluster-level screening
MIN_CLUSTER_EVENTS = 50
MIN_CLUSTER_DURATION = MIN_MIG_LEN
PLOT_MAX_CLUSTERS = None               # None = all clusters

# Figure style
FIG_DPI = 300
FIGSIZE_SUMMARY = (12, 8)
FIGSIZE_DETAIL = (8, 8)
LABEL_FONTSIZE = 14
TITLE_FONTSIZE = 16
SCATTER_SIZE = 8
SCATTER_ALPHA = 0.8
CONTOUR_LINEWIDTH = 1.5
BLANK_LINEWIDTH = 1.0

# More detailed candidate/post-check screen output for selected cluster ids
DEBUG_CLUSTER_IDS = []
