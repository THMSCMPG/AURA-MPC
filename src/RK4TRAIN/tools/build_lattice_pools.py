#!/usr/bin/env python3
"""build_lattice_pools.py

One-time (re-run only if the lattice design changes) prep step for the
RK4TRAN full-factorial lattice generator. Computes the four lattice axis
pools -- location, time, weather, orientation -- and emits them as a
compiled-in Fortran module (`lattice_pools_generated.f90`) that `main.f90`
`use`s directly, matching the existing LOC_POOL/TIME_POOL compiled-array
style rather than doing file I/O at Fortran runtime.

Requires network access to the Open-Elevation API for real elevation
lookups (blocked in sandboxed/offline environments -- use --offline-stub
to substitute placeholder elevations for smoke-testing the pipeline only;
never use --offline-stub for an actual training-data generation run).

Usage:
    pip install global-land-mask requests
    python3 build_lattice_pools.py [--offline-stub]
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

try:
    from global_land_mask import globe
except ImportError:
    sys.exit("Missing dependency: pip install global-land-mask")

OUT_PATH = Path(__file__).parent.parent / "lattice_pools_generated.f90"

# ---------------------------------------------------------------------------
# 1. Location pool: ~1000 land points -- a structured grid PLUS scattered
#    area-uniform random points, combined. REDESIGNED 2026-08-20 to support
#    randomized per-sprint training-location subsets (see
#    tools/select_training_locations.py) rather than the original fixed
#    24x12 grid.
#
#    Why not just make the grid finer? A pure grid (however fine) risks the
#    model learning "this specific finite set of coordinate combinations"
#    rather than the smooth underlying function of latitude/longitude --
#    Tommy's own stated goal was theoretical continuity (in the limit of
#    infinite time, every valid land lon/lat/elevation combination should be
#    reachable), which a purely structured grid can never approach no matter
#    how fine. Scattering genuinely random points (not just denser grid
#    points) breaks that grid-alignment risk directly. Practical compromise,
#    agreed explicitly: NOT literal continuous randomness (that would mean
#    fetching real-time elevation per sprint, which live compute nodes on
#    Loki likely can't do at all -- HPC compute nodes commonly have no
#    outbound internet access, only the login node does) -- instead, fetch
#    elevation ONCE for a large (~1000) pool, and draw a random SUBSET of
#    pool indices each sprint. At ~100 training locations/sprint (matching
#    the original per-sprint scale), a 1000-point pool gives ~10 sprints
#    before any repeat -- explicitly noted as a real limitation Tommy chose
#    to accept given he doesn't expect this campaign to run past ~10 weeks.
#
#    RESERVED VALIDATION INDICES: the first N_VAL_RESERVED points in the
#    returned pool (structured-grid points, for a stable, well-distributed,
#    reproducible held-out set) are reserved for validation and must NEVER
#    be drawn into a random training subset -- see
#    tools/select_training_locations.py, which enforces this by construction
#    (always excludes indices < N_VAL_RESERVED from its random draw).
# ---------------------------------------------------------------------------
N_VAL_RESERVED = 12           # first 12 pool points reserved for validation, permanently
N_STRUCTURED_TARGET = 500     # target land points from the structured grid component
N_SCATTERED_TARGET = 500      # target land points from the scattered-random component
RNG_SEED = 20260820           # fixed seed for the pool's scattered-point selection -- the
                               # POOL itself is reproducible; per-sprint SUBSET selection
                               # uses its own separate seed (see select_training_locations.py)


def _structured_grid_points(n_target: int) -> list[tuple[float, float]]:
    """Denser structured grid than the original 24x12, sized to land-filter
    down to roughly n_target points. Same pole-offset trick as before."""
    # empirically, ~35% of a lon/lat grid lands on land (matches the
    # original 100/288 ~= 34.7% hit rate) -- oversample the raw grid a bit
    # to comfortably clear n_target after filtering.
    raw_target = int(n_target / 0.35 * 1.15)
    # pick a lat/lon step that gives roughly raw_target raw grid points,
    # keeping the same "step evenly divides into a lon:lat ~2:1 aspect
    # ratio" shape as before (24 lon : 12 lat)
    step = (360.0 * 180.0 / raw_target) ** 0.5
    lons = np.arange(-180.0, 180.0, step)
    n_lat = max(2, round(180.0 / step))
    lat_step = 180.0 / n_lat
    lats = np.arange(-90.0 + lat_step / 2.0, 90.0, lat_step)  # offset off both poles, as before
    return [(float(lon), float(lat)) for lon in lons for lat in lats]


def _scattered_area_uniform_points(n_target_land: int, rng: np.random.Generator) -> list[tuple[float, float]]:
    """Genuinely area-uniform random (lon,lat) points over the sphere's
    surface -- NOT uniform in latitude (which would oversample the poles).
    Standard spherical sampling: lon ~ Uniform(-180,180), lat = arcsin(u)
    for u ~ Uniform(-1,1), converted to degrees.
    Oversamples and filters to land, looping until n_target_land land
    points are found (land is ~29% of Earth's surface, so this typically
    needs ~3-4x n_target_land raw draws)."""
    land_points: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    while len(land_points) < n_target_land:
        batch = max(200, (n_target_land - len(land_points)) * 4)
        lons = rng.uniform(-180.0, 180.0, batch)
        lats = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, batch)))
        for lon, lat in zip(lons, lats):
            key = (round(float(lon), 4), round(float(lat), 4))
            if key in seen:
                continue
            if globe.is_land(lat, lon):
                seen.add(key)
                land_points.append((float(lon), float(lat)))
                if len(land_points) >= n_target_land:
                    break
    return land_points


def build_location_pool(offline_stub: bool) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(RNG_SEED)

    structured_raw = _structured_grid_points(N_STRUCTURED_TARGET)
    structured_land = [(lon, lat) for lon, lat in structured_raw if globe.is_land(lat, lon)]
    # Shuffle before reserving validation indices -- unshuffled, the list is
    # ordered by longitude-then-latitude traversal, so "the first N" would
    # just be whatever land survives filtering at the first couple of
    # longitude columns tried (confirmed: gave a validation set clustered
    # at two narrow longitude bands near the poles, not a representative
    # geographic sample). Shuffling with a fixed seed keeps it reproducible
    # while actually spreading the reserved validation points globally.
    rng.shuffle(structured_land)
    print(f"[location] structured grid: {len(structured_raw)} raw -> {len(structured_land)} land points")

    scattered_land = _scattered_area_uniform_points(N_SCATTERED_TARGET, rng)
    print(f"[location] scattered area-uniform: {len(scattered_land)} land points")

    # dedupe (a scattered point landing very close to a grid point) and
    # combine -- reserved validation indices come from the STRUCTURED
    # component (first N_VAL_RESERVED), for a stable, well-distributed,
    # reproducible held-out set; everything after that (rest of structured
    # + all scattered) is eligible for random per-sprint training draws.
    combined: list[tuple[float, float]] = []
    seen = set()
    for lon, lat in structured_land + scattered_land:
        key = (round(lon, 3), round(lat, 3))
        if key not in seen:
            seen.add(key)
            combined.append((lon, lat))

    n_total = len(combined)
    print(f"[location] combined pool: {n_total} land points "
          f"(first {N_VAL_RESERVED} reserved for validation, {n_total - N_VAL_RESERVED} eligible for training)")

    if offline_stub:
        print("[location] WARNING: --offline-stub active, elevations are PLACEHOLDER ZEROS. "
              "Do not use this output for real training data.")
        return [(lon, lat, 0.0) for lon, lat in combined]

    import requests
    # Open-Elevation's bulk endpoint has a practical request-size limit --
    # chunk the pool into batches rather than one huge POST.
    elevations: list[float] = []
    CHUNK = 200
    for i in range(0, n_total, CHUNK):
        batch = combined[i:i + CHUNK]
        payload = {"locations": [{"latitude": lat, "longitude": lon} for lon, lat in batch]}
        resp = requests.post("https://api.open-elevation.com/api/v1/lookup", json=payload, timeout=120)
        resp.raise_for_status()
        elevations.extend(float(r["elevation"]) for r in resp.json()["results"])
        print(f"[location] elevation fetched: {min(i+CHUNK, n_total)}/{n_total}")

    return [(lon, lat, elev) for (lon, lat), elev in zip(combined, elevations)]


# ---------------------------------------------------------------------------
# 2. Time lattice: every Tuesday and Friday of all 52 ISO weeks (104 dates),
#    one fixed representative hour per date (solar noon, 12:00) -- matches
#    the 104-point count in the target lattice size (12x24x104x21x144),
#    NOT 104 dates x multiple times/day.
# ---------------------------------------------------------------------------
REFERENCE_YEAR = 2024  # arbitrary fixed reference year; matches prior TIME_POOL convention
FIXED_HOUR = 12.0      # solar noon


def build_time_pool() -> list[tuple[float, float, float, float, float]]:
    """Returns (minute, hour, day_of_year, month, year) tuples."""
    d = datetime.date(REFERENCE_YEAR, 1, 1)
    # walk forward to the first Tuesday (weekday()==1)
    while d.weekday() != 1:
        d += datetime.timedelta(days=1)
    dates = []
    cur = d
    while cur.year == REFERENCE_YEAR:
        dates.append(cur)
        cur += datetime.timedelta(days=3 if cur.weekday() == 1 else 4)  # Tue->Fri (+3), Fri->Tue (+4)
    dates = dates[:104]
    assert len(dates) == 104, f"expected 104 Tue/Fri dates, got {len(dates)}"
    pool = []
    for dt in dates:
        day_of_year = dt.timetuple().tm_yday
        pool.append((0.0, FIXED_HOUR, float(day_of_year), float(dt.month), float(REFERENCE_YEAR)))
    print(f"[time] {len(pool)} dates (Tue/Fri x 52 weeks, {REFERENCE_YEAR}, fixed {FIXED_HOUR:.0f}:00)")
    return pool


# ---------------------------------------------------------------------------
# 3. Weather: 21 real METAR sky-condition / present-weather category codes,
#    mapped to physical inputs (cloud_cover, humidity, wind_speed, pressure).
#    Temperature is deliberately NOT part of this pool -- ambient temperature
#    is a separately measured/modeled quantity (see main.f90's parametric
#    diurnal T_amb model), not bundled into a weather "archetype" the way an
#    earlier version of this script incorrectly did.
#    Standard METAR sky-cover codes: SKC/FEW/SCT/BKN/OVC/VV (oktas-based).
#    Present-weather codes (precip/obscuration/intensity) fill out the rest,
#    each carrying an implied sky condition. Mappings to cloud/humidity/wind/
#    pressure are representative engineering estimates (roughly monotonic
#    with severity), not sourced from a specific station -- flagged as such.
# ---------------------------------------------------------------------------
# code, cloud_cover, humidity, wind_speed(m/s), pressure(Pa)
METAR_CATEGORIES = [
    ("SKC",   0.000, 0.20,  2.0, 102000.0),  # sky clear
    ("FEW",   0.1875,0.30,  3.0, 101500.0),  # few clouds, 1-2 oktas
    ("SCT",   0.4375,0.40,  4.0, 101000.0),  # scattered, 3-4 oktas
    ("BKN",   0.750, 0.55,  6.0, 100000.0),  # broken, 5-7 oktas
    ("OVC",   1.000, 0.70,  7.0,  99000.0),  # overcast, 8 oktas
    ("VV",    1.000, 0.95,  1.0,  99500.0),  # indefinite ceiling / obscured
    ("BR",    0.500, 0.85,  2.0, 100500.0),  # mist
    ("FG",    0.900, 0.98,  0.5, 100000.0),  # fog
    ("HZ",    0.300, 0.55,  1.5, 101000.0),  # haze
    ("DZ",    0.850, 0.90,  3.0,  99800.0),  # drizzle
    ("-RA",   0.750, 0.80,  5.0,  99500.0),  # light rain
    ("RA",    0.900, 0.85,  8.0,  98800.0),  # moderate rain
    ("+RA",   1.000, 0.92, 12.0,  98000.0),  # heavy rain
    ("SH",    0.600, 0.75,  7.0,  99700.0),  # showers
    ("-SN",   0.800, 0.75,  4.0, 100200.0),  # light snow
    ("SN",    0.900, 0.80,  7.0,  99500.0),  # moderate snow
    ("+SN",   1.000, 0.85, 12.0,  98500.0),  # heavy snow
    ("FZRA",  0.950, 0.88,  6.0,  99000.0),  # freezing rain
    ("TS",    0.950, 0.80, 15.0,  98200.0),  # thunderstorm
    ("+TSRA", 1.000, 0.90, 20.0,  97500.0),  # thunderstorm, heavy rain
    ("GR",    1.000, 0.85, 22.0,  97000.0),  # hail / severe thunderstorm
]


def build_weather_pool() -> list[tuple[float, float, float, float]]:
    """Returns (wind, winddir, humidity, cloud, pressure) tuples -- NO temperature."""
    assert len(METAR_CATEGORIES) == 21, f"expected 21 METAR categories, got {len(METAR_CATEGORIES)}"
    pool = []
    for i, (code, cloud, humid, wind, press) in enumerate(METAR_CATEGORIES):
        winddir = float((i * 17) % 360)  # arbitrary variety, unused by physics (see main.f90)
        pool.append((wind, winddir, humid, cloud, press))
    codes = ", ".join(c for c, *_ in METAR_CATEGORIES)
    print(f"[weather] {len(pool)} METAR categories: {codes}")
    return pool


# ---------------------------------------------------------------------------
# 4. Orientation lattice: 12 pitch x 12 yaw = 144, roll FIXED at 0.
#    Tommy's own sizing formula (12x24x104x21x144) only multiplies by 144,
#    not by a 3rd roll axis -- roll is not swept. Physically defensible too:
#    in this model roll only ever attenuates capture (cos_aoi *= cos(roll)),
#    so roll=0 is also the electrically-optimal fixed choice.
#    Range: [0,90] for both pitch and yaw, exploiting the panel's 90-degree
#    mounting symmetry (same reduction referenced for the live RL action
#    space) rather than sweeping the full [-90,90]/[-180,180] range.
# ---------------------------------------------------------------------------
def build_orientation_pool() -> list[tuple[float, float, float]]:
    """Returns (pitch, roll, yaw) tuples, roll always 0.0."""
    pitches = np.linspace(0.0, 90.0, 12)
    yaws = np.linspace(0.0, 90.0, 12)
    pool = [(float(p), 0.0, float(y)) for p in pitches for y in yaws]
    assert len(pool) == 144
    print(f"[orientation] {len(pool)} points (12 pitch x 12 yaw, roll fixed 0)")
    return pool


# ---------------------------------------------------------------------------
# Fortran module emission
# ---------------------------------------------------------------------------
def emit_fortran_module(locations, times, weather, orientation) -> str:
    def real_array(name: str, values: list[float], per_line: int = 5) -> str:
        lines = [f"    real(c_double), parameter :: {name}({len(values)}) = [ &"]
        for i in range(0, len(values), per_line):
            chunk = values[i:i + per_line]
            body = ", ".join(f"{v:.6f}_c_double" for v in chunk)
            terminator = " ]" if i + per_line >= len(values) else ", &"
            lines.append(f"        {body}{terminator}")
        return "\n".join(lines)

    lon = [p[0] for p in locations]
    lat = [p[1] for p in locations]
    alt = [p[2] for p in locations]
    t_min = [t[0] for t in times]
    t_hour = [t[1] for t in times]
    t_day = [t[2] for t in times]
    t_month = [t[3] for t in times]
    t_year = [t[4] for t in times]
    w_wind = [w[0] for w in weather]
    w_winddir = [w[1] for w in weather]
    w_humid = [w[2] for w in weather]
    w_cloud = [w[3] for w in weather]
    w_press = [w[4] for w in weather]
    o_pitch = [o[0] for o in orientation]
    o_roll = [o[1] for o in orientation]
    o_yaw = [o[2] for o in orientation]

    src = f"""! AUTO-GENERATED by tools/build_lattice_pools.py -- do not hand-edit.
! Regenerate by re-running the script if the lattice design changes.
module lattice_pools
    use, intrinsic :: ISO_C_BINDING
    implicit none

    integer, parameter :: N_LOCATIONS = {len(locations)}
    integer, parameter :: N_TIMES = {len(times)}
    integer, parameter :: N_WEATHER = {len(weather)}
    integer, parameter :: N_ORIENTATIONS = {len(orientation)}

    ! --- location pool: lon, lat, elevation (m) ---
{real_array("LOC_LON", lon)}
{real_array("LOC_LAT", lat)}
{real_array("LOC_ALT", alt)}

    ! --- time pool: minute, hour, day-of-year, month, year ---
{real_array("TIME_MIN", t_min)}
{real_array("TIME_HOUR", t_hour)}
{real_array("TIME_DAY", t_day)}
{real_array("TIME_MONTH", t_month)}
{real_array("TIME_YEAR", t_year)}

    ! --- weather pool (21 real METAR categories): wind, winddir, humidity, cloud, pressure ---
    ! Temperature is NOT here -- see main.f90's parametric diurnal T_amb model.
{real_array("WX_WIND", w_wind)}
{real_array("WX_WINDDIR", w_winddir)}
{real_array("WX_HUMID", w_humid)}
{real_array("WX_CLOUD", w_cloud)}
{real_array("WX_PRESS", w_press)}

    ! --- orientation pool: pitch, roll (always 0), yaw ---
{real_array("ORI_PITCH", o_pitch)}
{real_array("ORI_ROLL", o_roll)}
{real_array("ORI_YAW", o_yaw)}

end module lattice_pools
"""
    return src


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-stub", action="store_true",
                         help="Use placeholder zero elevations instead of live Open-Elevation lookups "
                              "(pipeline smoke-testing ONLY, never for real data generation)")
    args = parser.parse_args()

    locations = build_location_pool(args.offline_stub)
    times = build_time_pool()
    weather = build_weather_pool()
    orientation = build_orientation_pool()

    total = len(locations) * len(times) * len(weather) * len(orientation)
    print(f"\nBase pool size: {len(locations)} x {len(times)} x {len(weather)} x {len(orientation)} = {total:,}")
    print("NOTE: final row count is larger than this -- T_amb (variable per date, computed at Fortran")
    print("runtime from a parametric diurnal model) and T_panel_initial (5 fixed points) both multiply")
    print("in inside main.f90, not here. See main.f90's header comment for the real total.")

    src = emit_fortran_module(locations, times, weather, orientation)
    OUT_PATH.write_text(src)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
