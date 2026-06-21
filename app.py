import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# UI Constants
BG, SURFACE, BORDER = "#0d1117", "#161b22", "#30363d"
TEXT_PRI, TEXT_SEC = "#c9d1d9", "#8b949e"
GREEN_BRT, ACCENT, ORANGE_ACC = "#39d353", "#58a6ff", "#ffa657"
FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"

COLORSCALE = [[0.00, "#0d1117"], [0.10, "#003d1f"], [0.30, "#007a33"], [0.55, "#00b347"], [0.80, "#39d353"],
              [1.00, "#adffb5"]]
MAP_STYLE = "carto-darkmatter"
BASE_LAYOUT = dict(paper_bgcolor=BG, plot_bgcolor=BG, font=dict(family=FONT, color=TEXT_PRI))

ZONE_COLORS = ["#58a6ff", "#f78166", "#ffa657", "#7ee787", "#d2a8ff", "#39d353", "#ff7b72", "#79c0ff", "#e3b341",
               "#56d364"]
COMP_PALETTE = ["rgba(255, 75, 115, 1)", "rgba(0, 212, 255, 1)", "rgba(255, 163, 0, 1)", "rgba(157, 78, 221, 1)"]
COMP_BG = ["rgba(255, 75, 115, 0.18)", "rgba(0, 212, 255, 0.15)", "rgba(255, 163, 0, 0.15)", "rgba(157, 78, 221, 0.18)"]

CUSTOM_CSS = """
html, body { margin: 0; padding: 0; height: 100vh; overflow: hidden; background-color: #0d1117; }
.snap-container { height: 100vh; overflow-y: scroll; scroll-snap-type: y mandatory; scroll-behavior: smooth; scrollbar-width: none; }
.snap-container::-webkit-scrollbar { display: none; }
.section-35 { height: 35vh; scroll-snap-align: start; box-sizing: border-box; padding: 2vh 28px 1vh 28px; border-bottom: 1px solid #30363d; }
.section-65 { height: 65vh; scroll-snap-align: start; box-sizing: border-box; padding: 2vh 28px 1vh 28px; border-bottom: 1px solid #30363d; }
.reset-badge { font-size: 11px; cursor: pointer; color: #58a6ff; text-decoration: underline; margin-left: 12px; font-weight: normal; text-transform: none; }
.reset-badge:hover { color: #79c0ff; }
"""


def fmt_date(dt):
    day = dt.day
    sfx = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    return f"{dt.strftime('%b')} {day}{sfx}, {dt.strftime('%Y')}"


def section_header(label, task_tag, reset_id=None):
    children = [
        html.Span(task_tag,
                  style={"color": TEXT_SEC, "fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.08em",
                         "marginRight": "10px"}),
        html.Span(label, style={"color": TEXT_PRI, "fontSize": "14px", "fontWeight": "600"}),
    ]

    if reset_id: children.append(html.Span("Reset Selection", id=reset_id, className="reset-badge"))

    return html.Div(children, style={"marginBottom": "2px"})


def interaction_hint(text, id=None):
    return html.Div(text, id=id,
                    style={"color": TEXT_SEC, "fontSize": "11px",
                           "marginBottom": "6px"}) if id else html.Div(text,style={"color": TEXT_SEC,
                                                                                   "fontSize": "11px",
                                                                                   "marginBottom": "6px"})


def empty_fig(msg=""):
    fig = go.Figure()
    fig.update_layout(
        **BASE_LAYOUT, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=10, b=10),
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                          font=dict(color=TEXT_SEC, size=12))] if msg else [])

    return fig


def empty_map(msg=""):
    fig = go.Figure(go.Scattermap(lat=[], lon=[], mode="markers"))
    fig.update_layout(
        **BASE_LAYOUT, map=dict(style=MAP_STYLE, center=dict(lat=38.9, lon=-77.04), zoom=10),
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                          font=dict(color=TEXT_SEC, size=12))] if msg else [])

    return fig


def canonical_pair(id_a, id_b):
    s1, s2 = sorted([id_a, id_b])
    return s1, s2, (id_a == s1)


def roll7(arr):
    return pd.Series(arr).rolling(7, min_periods=1).mean().values


# Load Data
DATA_DIR = "preprocessed_data"
stations = pd.read_parquet(f"{DATA_DIR}/stations.parquet")
daily_summary = pd.read_parquet(f"{DATA_DIR}/daily_summary.parquet")
zone_daily = pd.read_parquet(f"{DATA_DIR}/zone_daily.parquet")
zone_flows_df = pd.read_parquet(f"{DATA_DIR}/zone_flows_daily.parquet")
station_flows = pd.read_parquet(f"{DATA_DIR}/station_flows_daily.parquet")
od_pair_daily = pd.read_parquet(f"{DATA_DIR}/od_pair_daily.parquet")

target_year = int(daily_summary["date"].dt.year.mode()[0])
daily_summary = daily_summary[daily_summary["date"].dt.year == target_year].copy()

ALL_DATES = pd.date_range(start=f"{target_year}-01-01", end=f"{target_year}-12-31", freq="D")
zone_ids = sorted(stations["zone"].unique())
zone_meta = stations.groupby("zone").agg(lat=("lat", "mean"), lng=("lng", "mean"),
                                         n=("station_id", "count")).reset_index()
zone_meta["color"] = [ZONE_COLORS[i % len(ZONE_COLORS)] for i in zone_meta["zone"]]
sid_to_name = stations.set_index("station_id")["name"].to_dict()

cal = daily_summary.copy()
cal["weekday"] = cal["date"].dt.weekday
min_date = cal["date"].min()
min_weekday = min_date.weekday()
cal["week"] = ((cal["date"] - min_date).dt.days + min_weekday) // 7
cal["date_str"] = cal["date"].apply(fmt_date)

n_weeks = int(cal["week"].max()) + 1
z_grid = np.full((7, n_weeks), np.nan)
cd_grid = np.empty((7, n_weeks), dtype=object)

for _, r in cal.iterrows():
    z_grid[int(r["weekday"]), int(r["week"])] = r["change_score"]
    cd_grid[int(r["weekday"]), int(r["week"])] = r["date_str"]

jan1 = pd.Timestamp(f"{target_year}-01-01")
jan1_match = cal[cal["date"] == jan1]
if not jan1_match.empty:
    z_grid[int(jan1_match.iloc[0]["weekday"]), int(jan1_match.iloc[0]["week"])] = np.nan

month_labels = {r["date"].strftime("%b"): int(r["week"]) for _, r in cal.iterrows()}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Bike Network Yearly Dynamics"
app.index_string = app.index_string.replace('{%css%}', '{%css%}<style>' + CUSTOM_CSS + '</style>')

app.layout = html.Div(className="snap-container", children=[
    dbc.Container(fluid=True, style={"backgroundColor": BG, "padding": "0", "fontFamily": FONT}, children=[
        html.Div(className="section-35", children=[
            dbc.Row(dbc.Col(section_header("System-level change activity", "T1 - V1", reset_id="btn-reset-date"))),
            dbc.Row(dbc.Col(
                interaction_hint("Click a day to explore zone and station flows below. Double-click to deselect."))),
            dbc.Row(dbc.Col(dcc.Graph(id="v1-calendar", config={"displayModeBar": False}, style={"height": "25vh"}))),
            dcc.Store(id="store-date"), dcc.Store(id="store-zone-click"), dcc.Store(id="store-v2-matrix-click"),
            dcc.Store(id="store-active-flows", data=[]),
        ]),
        html.Div(className="section-65", children=[
            dbc.Row(dbc.Col(section_header("Zone-level OD flow map", "T2 - V2", reset_id="btn-reset-zone"))),
            dbc.Row(dbc.Col(interaction_hint(id="hint-t2", text="Select a day in the calendar above."))),
            dbc.Row([
                dbc.Col(dcc.Graph(id="v2-zone-map", config={"displayModeBar": False}, style={"height": "54vh"}),
                        width=5),
                dbc.Col(dcc.Graph(id="v2-zone-matrix", config={"displayModeBar": False}, style={"height": "54vh"}),
                        width=7),
            ]),
        ]),
        html.Div(className="section-35", children=[
            dbc.Row(
                dbc.Col(section_header("Station-level OD flows — selected zone", "T3 - V3", reset_id="btn-reset-arc"))),
            dbc.Row(dbc.Col(interaction_hint(id="hint-t3", text="Select a zone on the map above."))),
            dbc.Row([
                dbc.Col(dcc.Graph(id="v3-station-map", config={"displayModeBar": False}, style={"height": "25vh"}),
                        width=5),
                dbc.Col(dcc.Graph(id="v3-station-matrix", config={"displayModeBar": False}, style={"height": "25vh"}),
                        width=7),
            ]),
        ]),
        html.Div(className="section-65", children=[
            dbc.Row(dbc.Col(section_header("OD pair temporal dynamics & directional balance", "T4,T5 - V4",
                                           reset_id="btn-reset-v4"))),
            dbc.Row(dbc.Col(interaction_hint(
                "Click up to four distinct matrix cells or map arcs in V3 to overlay trace comparisons. Double-click to clear."))),
            dbc.Row(
                dbc.Col(dcc.Graph(id="v4-od-timeseries", config={"displayModeBar": False}, style={"height": "54vh"}))),
        ]),
    ])
])


@app.callback(Output("store-date", "data"),
              Input("v1-calendar", "clickData"),
              Input("btn-reset-date", "n_clicks"),
              prevent_initial_call=True)
def store_date(click_data, _n_clicks):
    if dash.ctx.triggered_id == "btn-reset-date" or not click_data:
        return None

    pt = click_data["points"][0]
    match = cal[(cal["week"] == int(pt["x"])) & (
                cal["weekday"] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(pt["y"]))]

    return None if match.empty else str(match.iloc[0]["date"].date())


@app.callback(Output("store-zone-click", "data"),
              Input("v2-zone-map", "clickData"),
              Input("btn-reset-zone", "n_clicks"),
              Input("btn-reset-date", "n_clicks"),
              State("store-zone-click", "data"), prevent_initial_call=True)
def store_zone_click(click_data, _reset_z, _reset_d, current):
    if dash.ctx.triggered_id in ("btn-reset-zone", "btn-reset-date") or not click_data or not click_data.get("points"):
        return None if dash.ctx.triggered_id in ("btn-reset-zone", "btn-reset-date") else current

    raw = click_data["points"][0].get("customdata")

    if raw is None: return current
    new_zone = int(raw[0]) if isinstance(raw, (list, tuple)) else int(raw)

    return None if current == new_zone else new_zone


@app.callback(Output("store-v2-matrix-click", "data"),
              Input("v2-zone-matrix", "clickData"),
              Input("btn-reset-zone", "n_clicks"),
              Input("btn-reset-date", "n_clicks"),
              State("store-v2-matrix-click", "data"), prevent_initial_call=True)
def store_v2_matrix_click(click_data, _reset_z, _reset_d, current):
    if dash.ctx.triggered_id in ("btn-reset-zone", "btn-reset-date") or not click_data or not click_data.get(
        "points"): return None
    pt = click_data["points"][0]

    if "x" not in pt or "y" not in pt: return None
    zi, zj = int(pt["y"].lstrip("Z")), int(pt["x"].lstrip("Z"))

    return None if current and current.get("zi") == zi and current.get("zj") == zj else {"zi": zi, "zj": zj}


@app.callback(Output("store-active-flows", "data"),
              Input("v3-station-matrix", "clickData"),
              Input("v3-station-map", "clickData"),
              Input("btn-reset-v4", "n_clicks"),
              Input("btn-reset-arc", "n_clicks"),
              Input("btn-reset-zone", "n_clicks"),
              Input("btn-reset-date", "n_clicks"),
              State("store-active-flows", "data"), prevent_initial_call=True)
def update_active_flows(mat_click, map_click, _r_v4, _r_arc, _r_z, _r_d, current):
    trigger = dash.ctx.triggered_id
    if trigger in ("btn-reset-v4", "btn-reset-arc", "btn-reset-zone", "btn-reset-date"): return []
    current = current or []
    new_pair = None

    click_src = mat_click if trigger == "v3-station-matrix" else map_click
    if click_src and click_src.get("points"):
        cust = click_src["points"][0].get("customdata")
        if cust and "||" in str(cust): new_pair = str(cust)

    if new_pair:
        si, sj = new_pair.split("||")
        if si != sj:
            current.remove(new_pair) if new_pair in current else current.append(new_pair)
            if len(current) > 4: current.pop(0)

    return current


@app.callback(Output("v1-calendar", "figure"), Input("store-date", "data"))
def render_v1(sel_date):
    fig = go.Figure(go.Heatmap(
        z=z_grid, x=list(range(n_weeks)), y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        customdata=cd_grid, hovertemplate="<b>%{customdata}</b><br>Change score: %{z:.3f}<extra></extra>",
        colorscale=COLORSCALE, showscale=True, xgap=8, ygap=8,
        colorbar=dict(thickness=10, len=1, bgcolor=BG, bordercolor=BORDER,
                      title=dict(text="value", font=dict(color=TEXT_SEC, size=10), side="right"),
                      tickfont=dict(color=TEXT_SEC, size=9))))

    if sel_date:
        match = cal[cal["date"] == pd.Timestamp(sel_date)]
        if not match.empty:
            wd, wk = int(match.iloc[0]["weekday"]), int(match.iloc[0]["week"])
            fig.add_shape(type="rect", x0=wk - 0.34, x1=wk + 0.34, y0=wd - 0.34, y1=wd + 0.34,
                          line=dict(color="#f0f0f0", width=1.5), fillcolor="rgba(0,0,0,0)")

    fig.update_layout(
        margin=dict(l=35, r=70, t=5, b=20),
        xaxis=dict(showgrid=False, zeroline=False, tickvals=list(month_labels.values()),
                   ticktext=list(month_labels.keys()), tickfont=dict(color=TEXT_SEC, size=10), scaleanchor="y",
                   scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, autorange="reversed", tickfont=dict(color=TEXT_SEC, size=9)),
        **BASE_LAYOUT)

    return fig


@app.callback(Output("hint-t2", "children"),
              Output("v2-zone-map", "figure"),
              Output("v2-zone-matrix", "figure"),
              Input("store-date", "data"),
              Input("store-zone-click", "data"),
              Input("store-v2-matrix-click", "data"))
def render_v2(sel_date, clicked_zone, matrix_click):
    if sel_date is None: return "Select a day in the calendar above.", empty_map(), empty_fig()

    date_ts = pd.Timestamp(sel_date)
    day_zd = zone_daily[zone_daily["date"] == date_ts]
    day_zf = zone_flows_df[zone_flows_df["date"] == date_ts]

    zod = pd.DataFrame(0.0, index=zone_ids, columns=zone_ids)
    pivot_df = day_zf.pivot(index="zone_i", columns="zone_j", values="count")
    zod.update(pivot_df)

    mat_zi, mat_zj = (matrix_click.get("zi"), matrix_click.get("zj")) if matrix_click else (None, None)
    map_fig = go.Figure()
    max_flow = max(zod.values.max(), 1)
    arc_thresh = max_flow * 0.05

    for zi in zone_ids:
        for zj in zone_ids:
            if zi == zj or zod.loc[zi, zj] < arc_thresh: continue
            flow = zod.loc[zi, zj]
            lat_i, lng_i = zone_meta.loc[zone_meta["zone"] == zi, ["lat", "lng"]].values[0]
            lat_j, lng_j = zone_meta.loc[zone_meta["zone"] == zj, ["lat", "lng"]].values[0]
            is_selected_arc = (mat_zi is not None and mat_zj is not None and mat_zi != mat_zj and (
                        (zi == mat_zi and zj == mat_zj) or (zi == mat_zj and zj == mat_zi)))
            line_width = (1.5 + 4.0 * (flow / max_flow)) if is_selected_arc else (0.5 + 3.0 * (flow / max_flow))
            line_color = f"rgba(255,166,87,{0.75 + 0.25 * (flow / max_flow):.2f})" if is_selected_arc else f"rgba(88,166,255,{0.12 + 0.55 * (flow / max_flow):.2f})"
            map_fig.add_trace(go.Scattermap(lat=[lat_i, lat_j, None], lon=[lng_i, lng_j, None], mode="lines",
                                            line=dict(width=line_width, color=line_color), hoverinfo="skip",
                                            showlegend=False))

    total_all = max(day_zd["total_outgoing"].sum(), 1)
    for _, zr in zone_meta.iterrows():
        z, col, n_members = int(zr["zone"]), zr["color"], int(zr["n"])
        zmatch = day_zd[day_zd["zone"] == z]
        total_out = float(zmatch["total_outgoing"].values[0]) if not zmatch.empty else 0.0
        int_ratio = float(zmatch["internal_ratio"].values[0]) if not zmatch.empty else 0.0
        is_sel = (clicked_zone == z) or (mat_zi is not None and mat_zj is not None and mat_zi == mat_zj and mat_zi == z)
        msize = 10 + 30 * (total_out / total_all) * (0.5 + int_ratio * 1.5)

        map_fig.add_trace(go.Scattermap(
            lat=[zr["lat"]], lon=[zr["lng"]], mode="markers",
            marker=dict(size=msize, color=col, opacity=1.0 if is_sel else 0.75), customdata=[z],
            hovertemplate=f"<b>Zone {z}</b><br>Stations: {n_members}<br>Outbound: {int(total_out)}<br>Internal: {int_ratio:.0%}<extra></extra>",
            showlegend=False
        ))
        if is_sel:
            map_fig.add_trace(go.Scattermap(lat=[zr["lat"]], lon=[zr["lng"]], mode="markers",
                                            marker=dict(size=msize + 20, color=ORANGE_ACC if mat_zi == mat_zj else col,
                                                        opacity=0.45), hoverinfo="skip", showlegend=False))

    map_fig.update_layout(
        map=dict(style=MAP_STYLE, center=dict(lat=stations["lat"].mean(), lon=stations["lng"].mean()), zoom=11.2),
        margin=dict(l=0, r=0, t=0, b=0), **BASE_LAYOUT)

    labels = [f"Z{z}" for z in zone_ids]
    mat_vals = zod.values.astype(float)
    hover_text = [[f"Z{zi}→Z{zj}: {int(mat_vals[i, j])} trips" for j, zj in enumerate(zone_ids)] for i, zi in
                  enumerate(zone_ids)]

    mat_fig = go.Figure(go.Heatmap(
        z=mat_vals, x=labels, y=labels, text=[[f"{int(v)}" if v > 0 else "" for v in row] for row in mat_vals],
        texttemplate="%{text}", hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text, colorscale=COLORSCALE, showscale=True,
        colorbar=dict(thickness=12, len=0.9, bgcolor=BG, bordercolor=BORDER,
                      title=dict(text="trips", font=dict(color=TEXT_SEC, size=10), side="right"),
                      tickfont=dict(color=TEXT_SEC, size=10)), xgap=2, ygap=2,
    ))

    if clicked_zone in zone_ids:
        idx = zone_ids.index(clicked_zone)
        mat_fig.add_shape(type="rect", x0=-0.5, x1=len(zone_ids) - 0.5, y0=idx - 0.5, y1=idx + 0.5,
                          line=dict(color=ACCENT, width=2), fillcolor="rgba(0,0,0,0)")
        mat_fig.add_shape(type="rect", x0=idx - 0.5, x1=idx + 0.5, y0=-0.5, y1=len(zone_ids) - 0.5,
                          line=dict(color=ACCENT, width=2), fillcolor="rgba(0,0,0,0)")
    if mat_zi in zone_ids and mat_zj in zone_ids:
        xi, yi = zone_ids.index(mat_zj), zone_ids.index(mat_zi)
        mat_fig.add_shape(type="rect", x0=xi - 0.5, x1=xi + 0.5, y0=yi - 0.5, y1=yi + 0.5,
                          line=dict(color=ORANGE_ACC, width=2.5), fillcolor="rgba(255,166,87,0.15)")

    mat_fig.update_layout(**BASE_LAYOUT, margin=dict(l=45, r=55, t=15, b=45),
                          xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=TEXT_SEC, size=10),
                                     title=dict(text="Destination Zone", font=dict(color=TEXT_SEC, size=10))),
                          yaxis=dict(showgrid=False, zeroline=False, autorange="reversed",
                                     tickfont=dict(color=TEXT_SEC, size=10),
                                     title=dict(text="Origin Zone", font=dict(color=TEXT_SEC, size=10))))

    return f"{fmt_date(date_ts)} — Click a zone bubble or matrix cell to inspect.", map_fig, mat_fig


@app.callback(Output("hint-t3", "children"),
              Output("v3-station-map", "figure"),
              Output("v3-station-matrix", "figure"),
              Input("store-date", "data"),
              Input("store-zone-click", "data"),
              Input("v3-station-map", "clickData"),
              Input("store-active-flows", "data"))
def render_v3(sel_date, sel_zone, station_click, active_flows):
    if not sel_date: return "Select a day in the calendar above.", empty_map(), empty_fig()
    if sel_zone is None: return "Select a zone on the map to view station-level flows.", empty_map(), empty_fig()

    zone_sids = stations[stations["zone"] == sel_zone]["station_id"].tolist()
    if not zone_sids: return f"Zone {sel_zone} contains no station markers.", empty_map(), empty_fig()

    core_sf = station_flows[
        (station_flows["date"] == pd.Timestamp(sel_date)) & station_flows["station_i"].isin(zone_sids) & station_flows[
            "station_j"].isin(zone_sids)]
    show_ids = sorted(zone_sids)

    # Vectorized Matrix Population
    sub_mat = pd.DataFrame(0.0, index=show_ids, columns=show_ids)
    pivot_sf = core_sf.pivot(index="station_i", columns="station_j", values="count")
    sub_mat.update(pivot_sf)

    core_meta = stations[stations["station_id"].isin(zone_sids)].set_index("station_id")
    clicked_station = None
    if dash.ctx.triggered_id == "v3-station-map" and station_click and station_click.get("points"):
        pt_cust = station_click["points"][0].get("customdata")
        if pt_cust and "||" not in str(pt_cust): clicked_station = str(pt_cust)

    map_fig = go.Figure()
    max_sf = max(sub_mat.values.max(), 1)

    for si in show_ids:
        for sj in show_ids:
            if si == sj or sub_mat.loc[si, sj] < (
                    max_sf * 0.10) or si not in core_meta.index or sj not in core_meta.index: continue
            flow = sub_mat.loc[si, sj]

            matched_idx = -1
            if active_flows:
                for idx, f_str in enumerate(active_flows):
                    hi_i, hi_j = f_str.split("||")
                    if (si == hi_i and sj == hi_j) or (si == hi_j and sj == hi_i):
                        matched_idx = idx
                        break

            line_width, line_color = (1.2 + 3.0 * (flow / max_sf),
                                      COMP_PALETTE[matched_idx]) if matched_idx != -1 else (0.6 + 2.5 * (flow / max_sf),
                                                                                            f"rgba(88,166,255,{0.15 + 0.65 * (flow / max_sf):.2f})")
            lat_i, lng_i, lat_j, lng_j = core_meta.loc[si, "lat"], core_meta.loc[si, "lng"], core_meta.loc[sj, "lat"], \
            core_meta.loc[sj, "lng"]

            map_fig.add_trace(go.Scattermap(lat=[lat_i, lat_j, None], lon=[lng_i, lng_j, None], mode="lines",
                                            line=dict(width=line_width, color=line_color), hoverinfo="skip",
                                            showlegend=False))
            map_fig.add_trace(go.Scattermap(lat=[(lat_i + lat_j) / 2], lon=[(lng_i + lng_j) / 2], mode="markers",
                                            marker=dict(size=12, color="rgba(0,0,0,0)", opacity=0.0),
                                            customdata=[f"{si}||{sj}"], hoverinfo="none", showlegend=False))

    if not core_meta.empty:
        map_fig.add_trace(go.Scattermap(lat=core_meta["lat"].tolist(), lon=core_meta["lng"].tolist(), mode="markers",
                                        marker=dict(size=9, color=GREEN_BRT, opacity=0.92),
                                        customdata=core_meta.index.tolist(),
                                        hovertemplate="<b>%{text}</b><extra></extra>", text=core_meta["name"].tolist(),
                                        showlegend=False))

    if active_flows:
        for idx, f_str in enumerate(active_flows):
            for ep_sid in f_str.split("||"):
                if ep_sid in core_meta.index:
                    map_fig.add_trace(
                        go.Scattermap(lat=[core_meta.loc[ep_sid, "lat"]], lon=[core_meta.loc[ep_sid, "lng"]],
                                      mode="markers",
                                      marker=dict(size=20, color=COMP_PALETTE[idx], opacity=0.45, allowoverlap=True),
                                      hoverinfo="skip", showlegend=False))

    if clicked_station and clicked_station in core_meta.index:
        map_fig.add_trace(
            go.Scattermap(lat=[core_meta.loc[clicked_station, "lat"]], lon=[core_meta.loc[clicked_station, "lng"]],
                          mode="markers", marker=dict(size=18, color=ACCENT, opacity=0.55, allowoverlap=True),
                          hoverinfo="skip", showlegend=False))

    map_fig.update_layout(
        map=dict(style=MAP_STYLE, center=dict(lat=core_meta["lat"].mean(), lon=core_meta["lng"].mean()), zoom=13.0),
        margin=dict(l=0, r=0, t=0, b=0), **BASE_LAYOUT)

    n_items = len(show_ids)
    gap_sz = 2 if n_items <= 12 else (1 if n_items <= 25 else 0)
    mat_fig = go.Figure(go.Heatmap(
        z=sub_mat.values.astype(float), x=show_ids, y=show_ids,
        customdata=[[f"{si}||{sj}" for sj in show_ids] for si in show_ids], text=[[
                                                                                      f"Origin: {core_meta.loc[si, 'name'] if si in core_meta.index else si}<br>Dest: {core_meta.loc[sj, 'name'] if sj in core_meta.index else sj}"
                                                                                      for sj in show_ids] for si in
                                                                                  show_ids],
        hovertemplate="%{text}<br>Trips: %{z}<extra></extra>",
        colorscale=COLORSCALE, showscale=True, colorbar=dict(thickness=8, len=0.95, bgcolor=BG, bordercolor=BORDER,
                                                             title=dict(text="trips", font=dict(color=TEXT_SEC, size=8),
                                                                        side="right"),
                                                             tickfont=dict(color=TEXT_SEC, size=8), x=1.01,
                                                             xanchor="left"), xgap=gap_sz, ygap=gap_sz,
    ))

    if clicked_station and clicked_station in show_ids:
        s_idx = show_ids.index(clicked_station)
        mat_fig.add_shape(type="rect", x0=-0.5, x1=n_items - 0.5, y0=s_idx - 0.5, y1=s_idx + 0.5,
                          line=dict(color=GREEN_BRT, width=1.5), fillcolor="rgba(0,0,0,0)", xref="x", yref="y")
        mat_fig.add_shape(type="rect", x0=s_idx - 0.5, x1=s_idx + 0.5, y0=-0.5, y1=n_items - 0.5,
                          line=dict(color=GREEN_BRT, width=1.5), fillcolor="rgba(0,0,0,0)", xref="x", yref="y")

    if active_flows:
        for idx, f_str in enumerate(active_flows):
            si_h, sj_h = f_str.split("||")
            if si_h in show_ids and sj_h in show_ids:
                xi, yi = show_ids.index(sj_h), show_ids.index(si_h)
                mat_fig.add_shape(type="rect", x0=xi - 0.5, x1=xi + 0.5, y0=yi - 0.5, y1=yi + 0.5,
                                  line=dict(color=COMP_PALETTE[idx], width=2.5), fillcolor=COMP_BG[idx], xref="x",
                                  yref="y")
                xi_r, yi_r = show_ids.index(si_h), show_ids.index(sj_h)
                mat_fig.add_shape(type="rect", x0=xi_r - 0.5, x1=xi_r + 0.5, y0=yi_r - 0.5, y1=yi_r + 0.5,
                                  line=dict(color=COMP_PALETTE[idx], width=1.5, dash="dot"), fillcolor="rgba(0,0,0,0)",
                                  xref="x", yref="y")

    mat_fig.update_layout(**BASE_LAYOUT, margin=dict(l=10, r=50, t=5, b=10),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, autorange="reversed", showticklabels=False))

    return f"Zone {sel_zone} — {n_items} Stations — {fmt_date(pd.Timestamp(sel_date))}. Active: {len(active_flows)}/4. Click to toggle.", map_fig, mat_fig


@app.callback(Output("v4-od-timeseries", "figure"),
              Input("store-active-flows", "data"),
              Input("store-date", "data"))
def render_v4(selected_flows, sel_date):
    if not selected_flows: return empty_fig(
        "Select up to four distinct OD pairs by clicking map arcs or matrix cells in V3 above.")

    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], shared_xaxes=True, vertical_spacing=0.06)

    for idx, flow_str in enumerate(selected_flows):
        origin_id, dest_id = flow_str.split("||")
        s1, s2, is_canonical = canonical_pair(origin_id, dest_id)
        pair_num = od_pair_daily[(od_pair_daily["station_1"] == s1) & (od_pair_daily["station_2"] == s2)][
            ["date", "forward", "reverse"]].set_index("date").reindex(ALL_DATES, fill_value=0)
        fwd = pair_num["forward" if is_canonical else "reverse"].to_numpy(dtype=float)
        rev = pair_num["reverse" if is_canonical else "forward"].to_numpy(dtype=float)

        o_name, d_name = sid_to_name.get(origin_id, origin_id), sid_to_name.get(dest_id, dest_id)

        fig.add_trace(go.Scatter(x=ALL_DATES, y=roll7(fwd), mode="lines", line=dict(color=COMP_PALETTE[idx], width=2.5),
                                 name=f"Vol: {o_name} → {d_name}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ALL_DATES, y=roll7(fwd - rev), mode="lines",
                                 line=dict(color=COMP_PALETTE[idx], width=1.8, dash="dash"),
                                 name=f"Net: {o_name} → {d_name}", fill="tozeroy", fillcolor=COMP_BG[idx]),
                      row=2, col=1)

    if sel_date:
        fig.add_vline(x=pd.Timestamp(sel_date), line_width=1.5, line_dash="dash", line_color="#ffffff", opacity=0.8)

    fig.add_hline(y=0, line_width=1.5, line_color=TEXT_SEC, line_dash="dot", row=2, col=1)
    fig.update_layout(
        **BASE_LAYOUT, margin=dict(l=55, r=20, t=15, b=35), hovermode="x unified",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.04, yanchor="bottom", bgcolor=SURFACE,
                    bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT_SEC, size=9)),
        xaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False, tickfont=dict(color=TEXT_SEC, size=9)),
        xaxis2=dict(showgrid=True, gridcolor=BORDER, zeroline=False, tickfont=dict(color=TEXT_SEC, size=9)),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False, tickfont=dict(color=TEXT_SEC, size=9),
                   title=dict(text="Trips / Day (7d Avg)", font=dict(color=TEXT_SEC, size=9))),
        yaxis2=dict(showgrid=True, gridcolor=BORDER, zeroline=False, tickfont=dict(color=TEXT_SEC, size=9),
                    title=dict(text="Net Flow (7d Avg)", font=dict(color=TEXT_SEC, size=9))))

    return fig


if __name__ == "__main__":
    app.run(debug=False)
