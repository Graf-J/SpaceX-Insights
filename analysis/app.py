"""
app.py

A focused Streamlit dashboard for EDA and visualizations based on starlink.db.
- Loads starlink.db (default ../data/starlink.db)
- Computes satellite positions using sgp4 (compute_satellite_position)
- Shows interactive 3D visualizations
- Includes an animated time-slider 3D view
"""
import io
import sqlite3
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from sgp4.api import Satrec, jday
from sklearn.decomposition import PCA
from scipy.optimize import least_squares

import streamlit as st
import streamlit.components.v1 as components

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="SpaceX Insights", layout="wide")
st.title("🚀 SpaceX Insights")
st.markdown(
    "Exploratory data analysis and interactive visualizations for Starlink satellites. "
    "Data is loaded from a local `starlink.db` produced by the Prefect pipeline."
)

# -------------------------
# Helpers: compute positions
# -------------------------
@st.cache_data(show_spinner=False)
def compute_satellite_position_cached(tle1: str, tle2: str, propagate_time_iso: str):
    try:
        satrec = Satrec.twoline2rv(tle1, tle2)
    except Exception as e:
        return {"x": np.nan, "y": np.nan, "z": np.nan, "vx": np.nan, "vy": np.nan, "vz": np.nan}

    # parse propagate_time_iso to datetime with UTC
    try:
        propagate_time = datetime.fromisoformat(propagate_time_iso)
        if propagate_time.tzinfo is None:
            propagate_time = propagate_time.replace(tzinfo=timezone.utc)
        else:
            propagate_time = propagate_time.astimezone(timezone.utc)
    except Exception:
        propagate_time = datetime.now(timezone.utc)

    jd, fr = jday(
        propagate_time.year,
        propagate_time.month,
        propagate_time.day,
        propagate_time.hour,
        propagate_time.minute,
        propagate_time.second + propagate_time.microsecond * 1e-6,
    )
    error_code, r, v = satrec.sgp4(jd, fr)
    if error_code != 0:
        return pd.Series(
            {
                "x": np.nan,
                "y": np.nan,
                "z": np.nan,
                "vx": np.nan,
                "vy": np.nan,
                "vz": np.nan,
            }
        )
    return pd.Series(
        {"x": r[0], "y": r[1], "z": r[2], "vx": v[0], "vy": v[1], "vz": v[2]}
    )


def compute_satellite_position(row, propagate_time=None):
    """
    Compatibility wrapper that the user-provided code expects (returns pandas Series).
    Uses compute_satellite_position_cached internally.
    """
    if propagate_time is None:
        propagate_time = datetime.now(timezone.utc)
    iso = propagate_time.isoformat()
    res = compute_satellite_position_cached(row["tle_line1"], row["tle_line2"], iso)
    return pd.Series(res)


# -------------------------
# Helpers: load DB & data
# -------------------------
@st.cache_data(show_spinner=True)
def load_starlink_db(db_path: str):
    """
    Load the essential starlink & launch join into a DataFrame.
    """
    con = sqlite3.connect(db_path)
    df = pd.read_sql(
        """
        SELECT 
            s.id AS starlink_id,
            s.height_km,
            s.object_name AS starlink_name, 
            l.id AS launch_id,
            l.date AS launch_date,
            l.name AS launch_name,
            s.tle_line1, 
            s.tle_line2 
        FROM starlink s
        JOIN launch l ON s.launch_id = l.id;
        """,
        con,
    )
    con.close()

    # parse dates
    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce", utc=True)
    return df


# -------------------------
# Helpers: earth texture caching
# -------------------------
@st.cache_data(show_spinner=False)
def load_earth_texture(url: str, size=(512, 256)):
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


# -------------------------
# Plot satellites
# -------------------------
def plot_satellites_3d(df, cone_size=0.1, cone_scale=0.05, texture_size=(512, 256)):
    """
    Plot satellites + velocity cones on textured Earth using Plotly.
    df must have x,y,z,vx,vy,vz and starlink_name.
    """
    R_earth = 6371.0

    # Load texture (returns floats in 0-1)
    texture = load_earth_texture(
        "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57730/land_ocean_ice_2048.png",
        size=texture_size,
    )

    # Sphere coordinates (meshgrid consistent with texture shape)
    h, w, _ = texture.shape
    u = np.linspace(0, 2 * np.pi, w)
    v = np.linspace(0, np.pi, h)
    u, v = np.meshgrid(u, v)
    xe = R_earth * np.cos(u) * np.sin(v)
    ye = R_earth * np.sin(u) * np.sin(v)
    ze = R_earth * np.cos(v)

    # Convert texture to grayscale for surfacecolor (already 0-1 from loader)
    surfacecolor = 0.299 * texture[:, :, 0] + 0.587 * texture[:, :, 1] + 0.114 * texture[:, :, 2]

    earth_surface = go.Surface(
        x=xe,
        y=ye,
        z=ze,
        surfacecolor=surfacecolor,
        colorscale="gray",
        cmin=0,
        cmax=1,
        showscale=False,
        hoverinfo="skip",
        lighting=dict(ambient=0.9, diffuse=0.5, roughness=0.9, specular=0.2),
        name="Earth",
        showlegend=False,
    )

    fig = go.Figure()
    fig.add_trace(earth_surface)

    if df is not None:
        # Prepare palette for launches
        palette = [
            "red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown", "olive", "teal"
        ]
        
        # Group by launch_name (fall back to launch_id if not present)
        if "launch_name" in df.columns:
            group_key = "launch_name"
        elif "launch_id" in df.columns:
            group_key = "launch_id"
        else:
            group_key = None

        

        if group_key is None:
            # Single group: use all data in one color
            x = np.asarray(df["x"])
            y = np.asarray(df["y"])
            z = np.asarray(df["z"])
            uvec = np.asarray(df["vx"]) * cone_scale
            vvec = np.asarray(df["vy"]) * cone_scale
            wvec = np.asarray(df["vz"]) * cone_scale
            names = df.get("starlink_name", None)

            satellite_cones = go.Cone(
                x=x, y=y, z=z, u=uvec, v=vvec, w=wvec,
                colorscale=[[0, "red"], [1, "red"]],
                sizemode="absolute", sizeref=cone_size, anchor="tail",
                showscale=False, name="Starlink", text=names, hoverinfo="text"
            )
            fig.add_trace(satellite_cones)
        else:
            groups = df.groupby(group_key, sort=False)
            for i, (gname, gdf) in enumerate(groups):
                color = palette[i % len(palette)]
                x = np.asarray(gdf["x"])
                y = np.asarray(gdf["y"])
                z = np.asarray(gdf["z"])
                uvec = np.asarray(gdf["vx"]) * cone_scale
                vvec = np.asarray(gdf["vy"]) * cone_scale
                wvec = np.asarray(gdf["vz"]) * cone_scale
                names = gdf.get("starlink_name", None)

                # Cone trace per-launch (colorscale with same color at both ends)
                satellite_cones = go.Cone(
                    x=x, y=y, z=z, u=uvec, v=vvec, w=wvec,
                    colorscale=[[0, color], [1, color]],
                    sizemode="absolute", sizeref=cone_size, anchor="tail",
                    showscale=False, name=f"{gname}", text=names, hoverinfo="text",
                    showlegend=True
                )
                fig.add_trace(satellite_cones)

    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            aspectmode="data",
        ),
        title="Satellites and Velocities Close to Earth",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1)
    )

    return fig


def plot_satellites_3d_time_slider(
    df,
    minutes_range=60,
    step_minutes=1,
    cone_size=0.1,
    cone_scale=0.05,
    texture_size=(256, 128), 
):
    """
    Build an animated Plotly Figure with frames for each minute in range.
    Note: this can be slow for many satellites and many frames.
    """
    R_earth = 6371.0
    # Earth mesh (low-res)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x_earth = R_earth * np.outer(np.cos(u), np.sin(v))
    y_earth = R_earth * np.outer(np.sin(u), np.sin(v))
    z_earth = R_earth * np.outer(np.ones_like(u), np.cos(v))
    earth_surface = go.Surface(x=x_earth, y=y_earth, z=z_earth, colorscale=[[0, "blue"], [1, "blue"]], opacity=0.3, showscale=False)

    times = np.arange(0, minutes_range + 1, step_minutes)
    frames = None

    if df is not None:
        # Choose palette
        palette = [
            "red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown", "olive", "teal"
        ]

        # Determine grouping key
        if "launch_name" in df.columns:
            group_key = "launch_name"
        elif "launch_id" in df.columns:
            group_key = "launch_id"
        else:
            group_key = None

        # Precompute frames
        frames = []
        # axis ranges
        x_min, x_max = x_earth.min(), x_earth.max()
        y_min, y_max = y_earth.min(), y_earth.max()
        z_min, z_max = z_earth.min(), z_earth.max()

        # compute positions for each frame (may be heavy)
        for t in times:
            # propagate each satellite by launch_date + t minutes
            records = []
            for _, row in df.iterrows():
                base_launch = row["launch_date"]
                if pd.isna(base_launch):
                    continue
                pt = base_launch + pd.Timedelta(minutes=int(t))
                res = compute_satellite_position_cached(row["tle_line1"], row["tle_line2"], pt.isoformat())
                if not np.isnan(res["x"]):
                    rec = {**res, "starlink_name": row.get("starlink_name", None)}
                    # preserve the launch identifier if present
                    if group_key is not None:
                        rec[group_key] = row.get(group_key)
                    records.append(rec)
            
            if len(records) == 0:
                # frame with just earth
                frame_traces = [earth_surface]
            else:
                step_df = pd.DataFrame(records)
                x_min = min(x_min, step_df["x"].min()); x_max = max(x_max, step_df["x"].max())
                y_min = min(y_min, step_df["y"].min()); y_max = max(y_max, step_df["y"].max())
                z_min = min(z_min, step_df["z"].min()); z_max = max(z_max, step_df["z"].max())

                frame_traces = [earth_surface]
                if group_key is None:
                    # single group
                    cones = go.Cone(
                        x=step_df["x"], y=step_df["y"], z=step_df["z"],
                        u=step_df["vx"] * cone_scale, v=step_df["vy"] * cone_scale, w=step_df["vz"] * cone_scale,
                        colorscale=[[0, "red"], [1, "red"]], sizemode="absolute", sizeref=cone_size, anchor="tail",
                        showscale=False, text=step_df.get("starlink_name", None), hoverinfo="text", name="Starlink",
                        showlegend=False
                    )
                    frame_traces.append(cones)
                else:
                    # build per-launch cones/markers
                    for i, (gname, gdf) in enumerate(step_df.groupby(group_key, sort=False)):
                        color = palette[i % len(palette)]
                        cones = go.Cone(
                            x=gdf["x"], y=gdf["y"], z=gdf["z"],
                            u=gdf["vx"] * cone_scale, v=gdf["vy"] * cone_scale, w=gdf["vz"] * cone_scale,
                            colorscale=[[0, color], [1, color]], sizemode="absolute", sizeref=cone_size, anchor="tail",
                            showscale=False, text=gdf.get("starlink_name", None), hoverinfo="text", name=f"{gname}",
                            showlegend=True
                        )
                        frame_traces.extend([cones])

            frames.append(go.Frame(data=frame_traces, name=str(int(t))))

    # initial figure
    init_data = frames[0].data if frames else [earth_surface]
    fig = go.Figure(data=init_data, frames=frames)

    # slider steps
    steps = [
        dict(
            method="animate",
            args=[[str(int(t))], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
            label=f"{int(t)} min",
        )
        for t in times
    ]
    sliders = [dict(steps=steps, currentvalue=dict(prefix="Minutes: "), len=1.0)]

    fig.update_layout(
        sliders=sliders,
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
            aspectmode="data",
        ),
        title="Starlink Satellites Over Time",
        legend=dict(x=0.02, y=0.98)
    )

    return fig

# -------------------------
# Plot estimated orbits
# -------------------------

def add_orbit_overlay(fig, df_now, color_by_launch=True):
    """
    Adds PCA-fitted orbit ellipse(s) on top of an existing static figure.
    If color_by_launch is True and launch grouping exists, fits one ellipse per launch in the same color.
    Otherwise fits a single ellipse using all points and adds it in yellow.
    """
    from sklearn.decomposition import PCA
    import numpy as np

    palette = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", "brown", "olive", "teal"]

    if "launch_name" in df_now.columns:
        group_key = "launch_name"
    elif "launch_id" in df_now.columns:
        group_key = "launch_id"
    else:
        group_key = None

    def fit_and_add(points_3d, color, name_suffix):
        if points_3d is None or points_3d.shape[0] == 0:
            return
        if points_3d.shape[0] < 5:
            # Too few points — just plot markers
            fig.add_trace(go.Scatter3d(
                x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2],
                mode="markers", marker=dict(size=3, color=color), name=f"{name_suffix} (points)"
            ))
            return
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points_3d)
        try:
            x0, y0, a, b, theta_rot = fit_ellipse_geometric(points_2d, regularization=True)
        except Exception:
            fig.add_trace(go.Scatter3d(
                x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2],
                mode="markers", marker=dict(size=3, color=color), name=f"{name_suffix} (points)"
            ))
            return
        phi = np.linspace(0, 2 * np.pi, 300)
        ellipse_2d_x = x0 + a * np.cos(phi) * np.cos(theta_rot) - b * np.sin(phi) * np.sin(theta_rot)
        ellipse_2d_y = y0 + a * np.cos(phi) * np.sin(theta_rot) + b * np.sin(phi) * np.cos(theta_rot)
        ellipse_2d = np.column_stack([ellipse_2d_x, ellipse_2d_y])
        ellipse_3d = pca.inverse_transform(ellipse_2d)
        fig.add_trace(go.Scatter3d(
            x=ellipse_3d[:, 0], y=ellipse_3d[:, 1], z=ellipse_3d[:, 2],
            mode="lines", line=dict(color=color, width=4), name=f"{name_suffix} (fitted orbit)"
        ))

    if color_by_launch and group_key is not None:
        groups = list(df_now.groupby(group_key, sort=False))
        for i, (gname, gdf) in enumerate(groups):
            color = palette[i % len(palette)]
            pts = gdf[["x","y","z"]].values
            fit_and_add(pts, color, gname)
    else:
        pts_all = df_now[["x","y","z"]].values
        fit_and_add(pts_all, "yellow", "Estimated Orbit")

    return fig


def fit_ellipse_geometric(points_2d, regularization):
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    # Initial guess: center at mean, axes as std dev, angle 0
    x0_init, y0_init = x.mean(), y.mean()
    a_init, b_init = np.std(x), np.std(y)
    theta_init = 0.0
    params_init = [x0_init, y0_init, a_init, b_init, theta_init]

    def residuals(p):
        x0, y0, a, b, theta = p
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Project points to rotated ellipse frame
        dx = x - x0
        dy = y - y0
        xp = cos_t * dx + sin_t * dy
        yp = -sin_t * dx + cos_t * dy
        # Distance to ellipse
        return np.sqrt((xp / a) ** 2 + (yp / b) ** 2) - 1

    if regularization:
        res = least_squares(residuals, params_init, loss="soft_l1", f_scale=0.1)
    else:
        res = least_squares(residuals, params_init)

    return res.x


def augment_df(df, value_space):
    dfs = []
    times = np.arange(0, 90 + 1, value_space)
    for t in times:
        df_step = df.assign(
            **df.apply(
                lambda row: compute_satellite_position(
                    row, propagate_time=row["launch_date"] + timedelta(minutes=int(t))
                ),
                axis=1,
            )
        ).dropna(subset=["x", "y", "z"])

        dfs.append(df_step)

    return pd.concat(dfs, axis=0)


# -------------------------
# Data loading
# -------------------------
db_default = Path(".") / "data" / "starlink.db"

try:
    df_raw = load_starlink_db(str(db_default))
except Exception as e:
    st.error(f"Failed to load DB: {e}")
    st.stop()


# -------------------------
# EDA Panel (left column) + Visuals (right column)
# -------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data preview & filters")
    st.dataframe(df_raw.head(10))

    # filters
    st.markdown("**Filters**")
    # unique launches
    launch_options = df_raw["launch_name"].astype(str).unique().tolist()
    launch_selected = st.multiselect("Filter by launch_name (multi)", options=launch_options, default="Starlink 4-35 (v1.5)")
    if len(launch_selected) > 10:
        st.info("You selected more than 10 launches — using only the first 10.")
        launch_selected = launch_selected[:10]

    # apply filters to df
    if not launch_selected or len(launch_selected) == 0:
        df = None
    else:
        df = df_raw.copy()
        if launch_selected:
            df = df[df["launch_name"].astype(str).isin(launch_selected)]

    if df is not None:
        record_num = len(df)
    else:
        record_num = 0
        st.markdown(f"**Filtered records:** {record_num}")
    

    # Track filter changes: if filters change, reset df_positions
    prev_launch = st.session_state.get("prev_launch_selected", None)

    # Compare with current filters
    filters_changed = (
        prev_launch != launch_selected
    )

    # If changed, reset computed positions
    if filters_changed:
        st.session_state["df_positions"] = None

    # Update stored filter values for next rerun
    st.session_state["prev_launch_selected"] = launch_selected

    # Button to compute instantaneous positions (launch_date)
    if st.button("Compute positions for filtered set"):
        if df is None:
            st.info("No filtered data available. Try widening filters.")
            df_positions = None
        else:
            with st.spinner("Computing positions (may take a while for many satellites)..."):
                # compute position at launch_date per row
                df_positions = df.copy()
                # apply compute_satellite_position with caching helper
                df_positions = df_positions.assign(
                    **df_positions.apply(
                        lambda row: compute_satellite_position(row, propagate_time=row["launch_date"]),
                        axis=1,
                    )
                )
                df_positions = df_positions.dropna(subset=["x", "y", "z", "vx", "vy", "vz"])
                # Drop Satelites that are too far away
                R_earth = 6371.0
                df_positions["r"] = np.sqrt(df_positions["x"] ** 2 + df_positions["y"] ** 2 + df_positions["z"] ** 2)
                df_positions = df_positions[(df_positions["r"] - R_earth > 100) & (df_positions["r"] - R_earth < 5000)]
                st.session_state["df_positions"] = df_positions
                st.success(f"Computed positions for {len(df_positions)} satellites.")
    else:
        df_positions = st.session_state.get("df_positions", None)

    # quick table of computed positions
    if df_positions is not None:
        st.markdown("### Example computed positions")
        st.dataframe(df_positions[["starlink_name", "launch_name", "x", "y", "z"]].head(10))

with col2:
    cone_size = 0.2
    cone_scale = 0.05
    texture_size = (512, 256)

    st.header("Interactive Plots")

    # Choice: static 3D or animated time-slider
    vis_mode = st.radio("Visualization mode", ["Static (single time)", "Animate over time"])

    # If positions not computed at any time, compute at now
    if df is None:
        df_now = None
    elif df_positions is None:
        with st.spinner("Computing current positions..."):
            df_now = df.copy()
            df_now = df_now.assign(
                **df_now.apply(
                    lambda row: compute_satellite_position(row, propagate_time=row["launch_date"]),
                    axis=1,
                )
            )
            df_now = df_now.dropna(subset=["x", "y", "z", "vx", "vy", "vz"])
            R_earth = 6371.0
            df_now["r"] = np.sqrt(df_now["x"]**2 + df_now["y"]**2 + df_now["z"]**2)
            df_now = df_now[(df_now["r"] - R_earth > 100) & (df_now["r"] - R_earth < 5000)]
            df_positions = df_now
    else:
        df_now = df_positions

    if vis_mode == "Static (single time)":
        if df_now is None or len(df_now) == 0:
            st.info("No computed positions available to show. Try widening filters.")
            add_orbit = False
        else:
            # Controls: orbit overlay
            add_orbit = st.checkbox("Add estimated orbit overlay", value=False, key="add_orbit")

        # Build base static figure (grouping/coloring handled by plot_satellites_3d)
        fig3 = plot_satellites_3d(df_now, cone_size=cone_size, cone_scale=cone_scale, texture_size=texture_size)

        # Overlay orbit(s) if requested and button clicked
        if add_orbit:
            try:
                fig3 = add_orbit_overlay(fig3, df_now, color_by_launch=True)
            except Exception as e:
                st.error(f"Failed to compute orbit overlay: {e}")

        # Render the figure
        html = fig3.to_html(include_plotlyjs="cdn", full_html=False)
        components.html(html, height=700, scrolling=True)
    else:
        # Animated mode requires launch_date to be present
        st.info("Animation will compute positions for each time step. This may be slow for many satellites.")
        with st.spinner("Building animation frames..."):
            fig_anim = plot_satellites_3d_time_slider(
                df_now,
                cone_size=cone_size,
                cone_scale=cone_scale
            )
            html = fig_anim.to_html(include_plotlyjs="cdn", full_html=False)
            components.html(html, height=500, scrolling=True)

        