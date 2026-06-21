import argparse
import math
import networkx as nx
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from sklearn.cluster import KMeans


USECOLS = [
    "started_at", "start_station_id", "start_station_name", "start_lat", "start_lng",
    "end_station_id", "end_station_name", "end_lat", "end_lng"
]


def clean_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def process_chunk(df: pd.DataFrame):
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df = df.dropna(subset=["started_at"])
    df["date"] = df["started_at"].dt.floor("D")

    df["start_station_id"] = clean_id(df["start_station_id"])
    df["end_station_id"] = clean_id(df["end_station_id"])
    df = df.dropna(subset=["start_station_id", "end_station_id"])
    df = df[df["start_station_id"] != df["end_station_id"]]

    start = df[["start_station_id", "start_station_name", "start_lat", "start_lng"]].rename(
        columns={"start_station_id": "station_id",
                 "start_station_name": "name", "start_lat": "lat", "start_lng": "lng"})
    end = df[["end_station_id", "end_station_name", "end_lat", "end_lng"]].rename(
        columns={"end_station_id": "station_id",
                 "end_station_name": "name", "end_lat": "lat", "end_lng": "lng"})

    meta = pd.concat([start, end], ignore_index=True).drop_duplicates(subset="station_id")
    flows = df.groupby(["date", "start_station_id", "end_station_id"]).size().reset_index(name="count").rename(
        columns={"start_station_id": "station_i", "end_station_id": "station_j"})

    return meta, flows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("input_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("preprocessed_data"))
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--resolution", type=float, default=1.6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_chunks, flow_chunks = [], []

    for m in range(1, 13):
        file = args.input_dir / f"{args.year}{m:02d}-capitalbikeshare-tripdata.csv"
        if not file.exists():
            continue

        print(f"Processing {file}")
        for chunk in pd.read_csv(file, usecols=USECOLS, chunksize=args.chunksize):
            meta, flows = process_chunk(chunk)
            meta_chunks.append(meta)
            flow_chunks.append(flows)

    if not meta_chunks:
        raise SystemExit("No data found.")

    stations = pd.concat(meta_chunks, ignore_index=True).groupby("station_id").agg(
        name=("name", "first"), lat=("lat", "median"), lng=("lng", "median")).reset_index()

    station_flows = pd.concat(flow_chunks, ignore_index=True).groupby(
        ["date", "station_i", "station_j"], as_index=False)["count"].sum()

    static_edges = station_flows.groupby(["station_i", "station_j"], as_index=False)["count"].sum()
    G = nx.from_pandas_edgelist(
        static_edges, source="station_i", target="station_j", edge_attr="count", create_using=nx.Graph)
    G.add_nodes_from(stations["station_id"])

    # Community detection
    base_communities = nx.community.louvain_communities(G, weight="count", resolution=args.resolution, seed=0)
    queue = deque([list(community) for community in base_communities])
    final_communities = []

    while queue:
        c_nodes = queue.popleft()
        if len(c_nodes) <= 32:
            final_communities.append(c_nodes)
        else:
            sub_stations = stations[stations["station_id"].isin(c_nodes)]
            n_splits = math.ceil(len(c_nodes) / 30)
            km = KMeans(n_clusters=n_splits, random_state=0, n_init=10)
            labels = km.fit_predict(sub_stations[["lat", "lng"]])

            for g in range(n_splits):
                split_nodes = sub_stations.iloc[labels == g]["station_id"].tolist()
                if split_nodes:
                    if len(split_nodes) == len(c_nodes):
                        final_communities.append(split_nodes)
                    else:
                        queue.append(split_nodes)

    station_to_zone = {sid: z_idx for z_idx, comm in enumerate(final_communities) for sid in comm}
    stations["zone"] = stations["station_id"].map(station_to_zone).fillna(-1)
    zone_centers = stations.groupby("zone").agg(lat=("lat", "mean"), lng=("lng", "mean")).reset_index()

    # Time series metrics
    curr = station_flows.rename(columns={"count": "curr"})
    prev = station_flows.copy()
    prev["date"] += pd.Timedelta(days=1)
    prev = prev.rename(columns={"count": "prev"})

    merged = curr.merge(prev, how="outer", on=["date", "station_i", "station_j"]).fillna(0)
    merged["absdiff"] = (merged["curr"] - merged["prev"]).abs()
    merged["sumflow"] = merged["curr"] + merged["prev"]

    daily_summary = merged.groupby("date").agg(diff=("absdiff", "sum"), total=("sumflow", "sum")).reset_index()
    daily_summary["change_score"] = (daily_summary["diff"] / daily_summary["total"]).replace([np.inf, -np.inf],
                                                                                             np.nan).fillna(1.0)
    daily_summary = daily_summary[["date", "change_score"]]

    zone_map = stations.set_index("station_id")["zone"]
    station_flows["zone_i"] = station_flows["station_i"].map(zone_map)
    station_flows["zone_j"] = station_flows["station_j"].map(zone_map)

    zone_flows = station_flows.groupby(["date", "zone_i", "zone_j"], as_index=False)["count"].sum()

    total = zone_flows.groupby(["date", "zone_i"], as_index=False)["count"].sum().rename(
        columns={"zone_i": "zone", "count": "total_outgoing"})
    internal = zone_flows[zone_flows["zone_i"] == zone_flows["zone_j"]][["date", "zone_i", "count"]].rename(
        columns={"zone_i": "zone", "count": "internal"})

    zone_daily = total.merge(internal, how="left", on=["date", "zone"]).fillna(0)
    zone_daily["internal_ratio"] = zone_daily["internal"] / zone_daily["total_outgoing"]
    zone_daily = zone_daily.merge(zone_centers, on="zone", how="left")

    od = station_flows.copy()
    od["station_1"] = np.minimum(od["station_i"], od["station_j"])
    od["station_2"] = np.maximum(od["station_i"], od["station_j"])

    forward_mask = od["station_i"] == od["station_1"]
    od["forward"] = np.where(forward_mask, od["count"], 0)
    od["reverse"] = np.where(forward_mask, 0, od["count"])

    od_pair_daily = od.groupby(["date", "station_1", "station_2"], as_index=False).agg(
        forward=("forward", "sum"), reverse=("reverse", "sum"))
    od_pair_daily["difference"] = od_pair_daily["forward"] - od_pair_daily["reverse"]

    # Explicit type casting for Parquet performance
    stations["station_id"] = stations["station_id"].astype(str)
    stations["zone"] = stations["zone"].astype("int32")
    station_flows["station_i"] = station_flows["station_i"].astype(str)
    station_flows["station_j"] = station_flows["station_j"].astype(str)
    station_flows["zone_i"] = station_flows["zone_i"].astype("int32")
    station_flows["zone_j"] = station_flows["zone_j"].astype("int32")
    zone_flows["zone_i"] = zone_flows["zone_i"].astype("int32")
    zone_flows["zone_j"] = zone_flows["zone_j"].astype("int32")
    zone_daily["zone"] = zone_daily["zone"].astype("int32")
    od_pair_daily["station_1"] = od_pair_daily["station_1"].astype(str)
    od_pair_daily["station_2"] = od_pair_daily["station_2"].astype(str)

    # Save to disk
    stations.to_parquet(args.output_dir / "stations.parquet", index=False)
    daily_summary.to_parquet(args.output_dir / "daily_summary.parquet", index=False)
    zone_daily.to_parquet(args.output_dir / "zone_daily.parquet", index=False)
    zone_flows.to_parquet(args.output_dir / "zone_flows_daily.parquet", index=False)
    station_flows[["date", "station_i", "station_j", "count"]].to_parquet(
        args.output_dir / "station_flows_daily.parquet", index=False)
    od_pair_daily.to_parquet(args.output_dir / "od_pair_daily.parquet", index=False)

    print(f"\nProcessed data saved to: {args.output_dir}")

    actual_sizes = [len(c) for c in final_communities]
    print("\n" + "-" * 40)
    print("PIPELINE METRICS SUMMARY:")
    print(f"Total Unique Stations:     {len(stations)}")
    print(f"Total Daily Flow Records:  {len(station_flows):,}")
    print(f"Total Zones Created:       {len(actual_sizes)}")
    print(f"Maximum Cluster Size:      {max(actual_sizes)} stations")
    print(f"Minimum Cluster Size:      {min(actual_sizes)} stations")
    print(f"Average Cluster Size:      {sum(actual_sizes) / len(actual_sizes):.1f} stations")
    print("-" * 40)


if __name__ == "__main__":
    main()
