# Bike Network Yearly Dynamics Dashboard

An interactive dashboard for exploring how origin-destination (OD) flow patterns of a
Bicycle Sharing System (BSS) change over the course of a year — from system-wide
activity, down to individual zones, stations, and routes.


## Data

This project is designed to work with trip-level bikeshare data in the standard
Lyft/Motivate format (`started_at`, `start_station_id`, `end_station_id`, etc.), such as
the data published by Citi Bike (New York):

🔗 **[citibikenyc.com/system-data](https://citibikenyc.com/system-data)**

The raw trip data itself is **not included in this repository** and is not redistributed
here, in line with the data provider's license terms. To run the pipeline:

1. Download the monthly trip data CSV(s) you want from the link above.
2. Place them in an `input_data/` folder (or pass a custom path, see below).
3. Run the preprocessing pipeline, then launch the dashboard.


## Project structure

- `data_preprocessing.py` — reads raw monthly trip CSVs, cleans them, computes daily
  station-to-station and zone-to-zone flows, detects zones via Louvain community
  detection, and writes the aggregated Parquet files used by the dashboard.
- `app.py` — the Dash application itself.

## Running

```bash
# 2. Preprocess raw trip data into the format the dashboard expects
python data_preprocessing.py --input-dir input_data --output-dir preprocessed_data

# 3. Launch the dashboard
python app.py
```

## License note

The data linked above is provided by the BSS operator under its own license terms,
which restrict redistributing the raw data as a stand-alone dataset. This repository
only links to the original source and does not host or redistribute the raw trip data.
