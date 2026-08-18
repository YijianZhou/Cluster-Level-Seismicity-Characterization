# Cluster-Level Seismicity Characterization

Research scripts for identifying earthquake clusters and characterizing their
temporal, hydraulic-fracturing, nearest-neighbor, and migration behavior. The
workflow was developed for its first application to Delaware Basin seismicity,
but its individual stages can be adapted to other regional earthquake catalogs.

This repository is a script-based research workflow rather than an installed
Python package. File paths and analysis parameters are intentionally exposed near
the top of each script so that they can be reviewed and adjusted for a new
catalog.

## Capabilities

- Multi-parameter DBSCAN/HDBSCAN clustering followed by consensus merging and
  optional manual refinement.
- Cluster-level characteristic timescale and Fano-factor calculation.
- Association of earthquake clusters with hydraulic-fracturing disclosures.
- Nearest-neighbor rescaled time-distance analysis and declustering diagnostics.
- Detection and quality control of depth and three-dimensional-distance
  seismicity migration.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `0_Clustering/` | Initial clustering, multi-parameter HDBSCAN runs, consensus merging, manual-cluster consolidation, and MATLAB inspection tools. |
| `HF_analysis/` | Hydraulic-fracturing association, characteristic-time/Fano-factor analysis, and nearest-neighbor diagnostics. |
| `Migration/` | Sliding quantile-contour calculation, migration detection, post-checking, and diagnostic plotting. |

The input catalogs and generated output are not distributed with the source
code. By default, scripts expect project-level `input/` and `output/`
directories relative to the directory from which they are run.

## Requirements

The Python workflow requires Python 3 and the following third-party packages:

- NumPy
- pandas
- Matplotlib
- SciPy
- scikit-learn
- hdbscan
- ObsPy

One possible environment setup is:

```bash
conda create -n cluster-characterization python=3.11
conda activate cluster-characterization
python -m pip install numpy pandas matplotlib scipy scikit-learn hdbscan obspy
```

MATLAB is optional and is used only for the interactive three-dimensional
cluster inspection/refinement step in `0_Clustering/`.

## Input data

### Earthquake catalog

The initial clustering script expects a headerless CSV with one event per row:

```text
origin_time,latitude,longitude,depth_km,magnitude
```

The default filename is:

```text
input/nanometrics-delaware-basin_reloc_2019-2024.csv
```

During initial clustering, each event receives an integer `event_id` equal to
its zero-based source-row index. Keep the source catalog order unchanged across
all stages because this identifier is used to merge, de-duplicate, and match
events.

### Cluster catalog

Cluster files use the following format:

```text
# cluster-name
origin_time,latitude,longitude,depth_km,magnitude,event_id
...
# another-cluster
...
```

Several downstream scripts currently default to:

```text
input/db-seis_4-manual-cluster_Nmin-100.csv
```

If a different minimum cluster size or filename is used, update `FCLUST` and
related path variables in the analysis scripts.

### Hydraulic-fracturing disclosures

`HF_analysis/format_disclosure.py` reads the FracFocus-style disclosure table
specified by `fin` (default `input/DisclosureList_1.csv`) and writes a cleaned
well/job table used by `HF_analysis/analyze_hf_triggering.py`. Review its column
mapping before applying it to a disclosure export with a different schema.

## Workflow

Run commands from the repository root so that all relative `input/` and
`output/` paths resolve consistently.

### 1. Detect and merge earthquake clusters

```bash
python 0_Clustering/1_init_dbscsn-clustering.py
python 0_Clustering/2_multi-param_hdbscan-clustering.py
python 0_Clustering/3_merge-clusters.py
```

The stages perform the following operations:

1. `1_init_dbscsn-clustering.py` applies an initial three-dimensional DBSCAN
   clustering after catalog filtering.
2. `2_multi-param_hdbscan-clustering.py` repeats HDBSCAN over user-defined
   `min_cluster_size` and `min_samples` combinations.
3. `3_merge-clusters.py` joins clusters that share event IDs across runs using a
   union-find merge and removes duplicate events.

The clustering thresholds, catalog bounds, and output names are defined near
the top of each script and should be validated for the target catalog.

For optional manual refinement, inspect the merged clusters with:

```matlab
plot_cluster3d
```

Save accepted or split clusters as individual files under
`output/manual_clusters/cluster-*.csv`, then combine them with:

```bash
python 0_Clustering/4_merge-manual-clusters.py
```

`0_Clustering/csv2npy.py` provides an optional CSV-to-NumPy conversion utility.

### 2. Calculate characteristic timescale and Fano factor

```bash
python HF_analysis/calc_char-time-ff.py
python HF_analysis/plot_tau_vs_bin.py
python HF_analysis/plot_tau-ff_summary.py
```

For each cluster, `calc_char-time-ff.py` scans count-window lengths, estimates
the characteristic timescale from the first autocorrelation lag below `1/e`,
and calculates the Fano factor from sliding event counts. Its default summary is
written to:

```text
output/acf-tau_clusters/cluster_tau-ff_summary.csv
```

Some plotting scripts currently point to
`output/cluster_tau-ff_summary.csv`. Either copy the generated summary there or
update each script's `summary_file` setting before running the plots.

### 3. Associate clusters with hydraulic fracturing

```bash
python HF_analysis/format_disclosure.py
python HF_analysis/analyze_hf_triggering.py
```

The association analysis uses configurable spatial radius, event-count,
catalog-coverage, and time-lag criteria. These thresholds are specific to the
catalog and disclosure coverage and should not be treated as universal values.

### 4. Compute nearest-neighbor statistics

```bash
python HF_analysis/nn_compute.py
python HF_analysis/plot_NN-FF-tau.py
```

The nearest-neighbor metric follows the rescaled time-distance form

```text
eta_ij = t_ij * r_ij^d * 10^(-b * m_i)
```

The distributed configuration uses two-dimensional epicentral distance with
`d = 1.6`. This pairing is deliberate. Do not switch `distance_mode` to
three-dimensional hypocentral distance while retaining `d = 1.6` unless the
fractal dimension has been independently estimated from the corresponding 3-D
hypocenter distribution. Results are saved by default to
`output/nn_results_all.npz`.

### 5. Detect cluster migration

Review `Migration/config.py`, then run:

```bash
python Migration/detect_mig_clust.py
```

The migration workflow:

1. Computes bootstrapped D05-D95 depth and 3-D-distance contours in sliding
   windows.
2. Fits the quantiles jointly to estimate expanding-front velocities.
3. Merges compatible migration segments into candidate periods.
4. Applies whole-period and sliding-subsegment quality controls.
5. Writes a summary CSV and candidate/final-period diagnostic figures.

Important settings in `Migration/config.py` include the contour window and step,
bootstrap count and random seed, minimum migration duration, velocity thresholds,
fit-window requirements, and parallel worker count. Diagnostic messages report
candidate screening, boundary tuning, strict final checks, and accepted
migration periods.

## Main outputs

Depending on the enabled stages, the workflow produces:

- NumPy and CSV cluster catalogs.
- Per-cluster autocorrelation/Fano-factor figures and a cluster summary CSV.
- Cleaned hydraulic-fracturing disclosure data and association diagnostics.
- Nearest-neighbor arrays and rescaled time-distance figures.
- Migration summary tables, cached contour calculations, and detailed migration
  plots.

Output filenames are configurable in each script. Existing files may be
overwritten, so preserve publication-ready results separately.

## Reproducibility and interpretation

- Keep event IDs stable by preserving the input catalog row order.
- Record all edited path and threshold values for each run.
- Migration bootstrapping uses the random seed in `Migration/config.py`; keep the
  seed fixed when reproducing a result.
- HDBSCAN consensus and manual refinement are analysis choices. Archive the
  accepted manual-cluster files with any reported result.
- Completeness magnitude, catalog resolution, location uncertainty, and spatial
  coverage affect every characterization stage and must be reassessed for a new
  study area.
- These scripts support scientific interpretation; they do not replace visual
  quality control or uncertainty analysis.

## References

- Zhou et al. (Under review). Manuscript describing the first application of
  this workflow to Delaware Basin seismicity.

## Issues and contributions

Bug reports and focused pull requests are welcome through the GitHub issue
tracker. When reporting a result discrepancy, include the script, parameter
values, dependency versions, and the relevant console output.
