# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from obspy import UTCDateTime
from datetime import datetime

# ============================
# User-defined variables
# ============================
fname = 'input/DisclosureList_1.csv'
lat_rng = [30.55, 33.]       # Delaware Basin latitude range
lon_rng = [-105, -102.7]   # Delaware Basin longitude range
ot_rng = [UTCDateTime("2019-04-01"), UTCDateTime("2025-05-01")]  # Study period filter
outname = 'output/cleaned_DisclosureList_well_info.csv'

# ============================
# Load and process
# ============================
df = pd.read_csv(fname, low_memory=False)
fields_needed = ["WellName", "Latitude", "Longitude", "APINumber",
                 "JobStartDate", "JobEndDate", "TotalBaseWaterVolume"]

missing = [f for f in fields_needed if f not in df.columns]
if missing:
    raise ValueError(f"Missing required fields in DisclosureList file: {missing}")

def parse_datetime_safe(date_str):
    try:
        # Handle formats like "9/26/2025 10:01:00 AM"
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y %I:%M:%S %p")
        offset = 5*3600 if dt.month in [4,5,6,7,8,9,10] else 6*3600
        return str(UTCDateTime(dt) + offset)
        #return str(UTCDateTime(dt))
    except Exception as e:
        print(f"[⚠️ Bad date] Cannot parse: {date_str} — {e}")
        return ""

records = []
for idx, row in df.iterrows():
    lat, lon = row["Latitude"], row["Longitude"]

    # Skip wells outside region of interest
    if not (lat_rng[0] <= lat <= lat_rng[1] and lon_rng[0] <= lon <= lon_rng[1]):
        continue

    # Convert job start/end to UTCDateTime strings
    job_start = parse_datetime_safe(row["JobStartDate"]) if pd.notna(row["JobStartDate"]) else ""
    job_end = parse_datetime_safe(row["JobEndDate"]) if pd.notna(row["JobEndDate"]) else ""

    if job_start and job_end:
        t0 = UTCDateTime(job_start)
        t1 = UTCDateTime(job_end)
        if t1 < ot_rng[0] or t0 > ot_rng[1]: continue

    # Skip rows without lat/lon and print them
    if pd.isna(lat) or pd.isna(lon):
        print(f"[DisclosureList: Missing lat/lon] Row {idx}: {row.to_dict()}")
        continue

    records.append({
        "well_name": row.get("WellName", ""),
        "lat": lat,
        "lon": lon,
        "api_number": row.get("APINumber", ""),
        "job_start": job_start,
        "job_end": job_end,
        "fluid_volume_gal": row.get("TotalBaseWaterVolume", "")
    })

out_df = pd.DataFrame(records)
out_df.to_csv(outname, index=False)
print(f"✅ Saved {len(out_df)} wells to {outname}")
