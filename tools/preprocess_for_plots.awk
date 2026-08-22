#!/usr/bin/awk -f
#
# preprocess_for_plots.awk
#
# Single-pass, memory-bounded aggregation over RK4TRAN's lattice CSV output,
# producing small binned-summary files for plot_synthetic_data.gp.
#
# WHY THIS EXISTS: the source data is ~3.8 billion rows / ~2TB as CSV text --
# far too large to load into any plotting tool's memory, and even gnuplot's
# native streaming can't usefully render billions of points (the image would
# be meaningless and the file enormous). This script computes binned
# aggregates (mean, stddev, count per bin) for 8 useful relationships in ONE
# pass through the data, using fixed-size associative arrays keyed by bin
# index -- memory usage is bounded by (number of bins x number of tracked
# statistics), completely independent of row count. Whether the input is
# 1GB or 2TB, this script's memory footprint stays a few hundred KB.
#
# Usage:
#   awk -f preprocess_for_plots.awk lattice_batches/*.csv
#   (or a single combined file -- multiple files are handled correctly,
#   each file's own header row is skipped via FNR==1)
#
# Output: writes 8 small .dat files (a few KB to a few hundred KB each,
# regardless of input size) to the current directory, consumed by
# plot_synthetic_data.gp.
#
# Runtime: I/O-bound, dominated by reading the full file(s) once. On a
# multi-TB dataset expect this to take hours depending on disk speed --
# that's expected and fine (optimized for memory, not time, per request).
# Progress is printed to stderr every 50M rows so you can confirm it's
# still alive during a long run.

BEGIN {
    FS = ","
    row_count = 0
    PROGRESS_EVERY = 50000000

    # bin widths / ranges -- see column comments below for what each covers
    N_HOUR_BINS = 48        # half-hour bins, 0-24h
    TAMB_BIN_WIDTH_C = 2.0  # T_amb bins, -50 to 70C -> 60 bins
    N_TAMB_BINS = 60
    N_ORIENT_ERR_BINS = 140 # 1-degree bins, orientation_error up to ~127 max possible
    N_CLOUD_BINS = 21       # 0.05-wide bins, cloud_cover 0-1
    RELAX_BIN_WIDTH_C = 5.0 # thermal relaxation bins, -150 to +150C -> 60 bins
    N_RELAX_BINS = 60
    ALT_BIN_WIDTH_M = 200.0 # elevation bins, 0-6000m -> 30 bins
    N_ALT_BINS = 30
}

FNR == 1 { next }  # skip header row of EACH input file

{
    row_count++
    if (row_count % PROGRESS_EVERY == 0) {
        printf("  ...processed %d rows\n", row_count) > "/dev/stderr"
    }

    # --- column extraction (1-indexed, matches RK4TRAN's CSV header exactly) ---
    lon = $1 + 0; lat = $2 + 0; alt = $3 + 0
    hour = $5 + 0
    T_amb_K = $9 + 0; wind_speed = $10 + 0; cloud_cover = $14 + 0
    pitch = $17 + 0
    T_operating_K = $20 + 0; T_operating_sigma = $21 + 0
    eta = $22 + 0
    optimal_pitch = $24 + 0
    orientation_error = $30 + 0
    T_panel_initial_K = $31 + 0
    T_after_15min_K = $32 + 0

    T_amb_C = T_amb_K - 273.15
    T_operating_C = T_operating_K - 273.15

    # --- Plot 1: diurnal cycle -- T_operating & eta vs hour-of-day ---
    hb = int(hour * 2)
    if (hb < 0) hb = 0
    if (hb >= N_HOUR_BINS) hb = N_HOUR_BINS - 1
    hour_n[hb]++
    hour_T_sum[hb] += T_operating_C; hour_T_sumsq[hb] += T_operating_C * T_operating_C
    hour_eta_sum[hb] += eta; hour_eta_sumsq[hb] += eta * eta

    # --- Plot 2: efficiency vs ambient temperature (BETA_T derating) ---
    tb = int((T_amb_C + 50.0) / TAMB_BIN_WIDTH_C)
    if (tb < 0) tb = 0
    if (tb >= N_TAMB_BINS) tb = N_TAMB_BINS - 1
    tamb_n[tb]++
    tamb_eta_sum[tb] += eta

    # --- Plot 3: optimal pitch vs latitude (exact-value keying -- only ~12
    #     distinct real lattice latitudes exist, no need to bin) ---
    lat_key = sprintf("%.1f", lat)
    lat_n[lat_key]++
    lat_pitch_sum[lat_key] += optimal_pitch
    if (!(lat_key in lat_seen)) { lat_seen[lat_key] = 1; lat_keys[++n_lat_keys] = lat_key }

    # --- Plot 4: efficiency vs orientation error (ties to optimal-vs-actual
    #     comparison analysis) ---
    ob = int(orientation_error)
    if (ob < 0) ob = 0
    if (ob >= N_ORIENT_ERR_BINS) ob = N_ORIENT_ERR_BINS - 1
    orient_n[ob]++
    orient_eta_sum[ob] += eta; orient_eta_sumsq[ob] += eta * eta

    # --- Plot 5: efficiency & uncertainty vs cloud cover ---
    cb = int(cloud_cover * 20.0)
    if (cb < 0) cb = 0
    if (cb >= N_CLOUD_BINS) cb = N_CLOUD_BINS - 1
    cloud_n[cb]++
    cloud_eta_sum[cb] += eta
    cloud_sigma_sum[cb] += T_operating_sigma

    # --- Plot 6: thermal relaxation -- how much T changes in 15min as a
    #     function of how far the starting temp was from steady state ---
    delta_from_ss_C = (T_panel_initial_K - T_operating_K)  # K diff == C diff
    rb = int((delta_from_ss_C + 150.0) / RELAX_BIN_WIDTH_C)
    if (rb < 0) rb = 0
    if (rb >= N_RELAX_BINS) rb = N_RELAX_BINS - 1
    relax_n[rb]++
    relax_delta_sum[rb] += (T_after_15min_K - T_panel_initial_K)

    # --- Plot 7: T_operating & eta vs elevation ---
    ab = int(alt / ALT_BIN_WIDTH_M)
    if (ab < 0) ab = 0
    if (ab >= N_ALT_BINS) ab = N_ALT_BINS - 1
    alt_n[ab]++
    alt_T_sum[ab] += T_operating_C
    alt_eta_sum[ab] += eta

    # --- Plot 8: uncertainty (T_operating_sigma) vs wind speed (exact-value
    #     keying -- at most 21 distinct wind speeds, one per METAR category) ---
    wind_key = sprintf("%.2f", wind_speed)
    wind_n[wind_key]++
    wind_sigma_sum[wind_key] += T_operating_sigma
    if (!(wind_key in wind_seen)) { wind_seen[wind_key] = 1; wind_keys[++n_wind_keys] = wind_key }
}

END {
    printf("Total rows processed: %d\n", row_count) > "/dev/stderr"

    # --- Plot 1 output: hour, mean_T_operating_C, stddev_T, mean_eta, stddev_eta, n ---
    out1 = "plot1_diurnal.dat"
    print "# hour mean_T_operating_C stddev_T mean_eta stddev_eta n" > out1
    for (i = 0; i < N_HOUR_BINS; i++) {
        if (hour_n[i] == 0) continue
        n = hour_n[i]
        mT = hour_T_sum[i] / n
        sdT = sqrt(hour_T_sumsq[i]/n - mT*mT < 0 ? 0 : hour_T_sumsq[i]/n - mT*mT)
        meta = hour_eta_sum[i] / n
        sdeta = sqrt(hour_eta_sumsq[i]/n - meta*meta < 0 ? 0 : hour_eta_sumsq[i]/n - meta*meta)
        printf("%.2f %.4f %.4f %.6f %.6f %d\n", i/2.0, mT, sdT, meta, sdeta, n) >> out1
    }
    close(out1)

    # --- Plot 2 output: T_amb_C, mean_eta, n ---
    out2 = "plot2_eta_vs_tamb.dat"
    print "# T_amb_C mean_eta n" > out2
    for (i = 0; i < N_TAMB_BINS; i++) {
        if (tamb_n[i] == 0) continue
        printf("%.2f %.6f %d\n", -50.0 + i*TAMB_BIN_WIDTH_C + TAMB_BIN_WIDTH_C/2.0, tamb_eta_sum[i]/tamb_n[i], tamb_n[i]) >> out2
    }
    close(out2)

    # --- Plot 3 output: latitude, mean_optimal_pitch, n ---
    out3 = "plot3_optimal_pitch_vs_lat.dat"
    print "# latitude mean_optimal_pitch n" > out3
    for (k = 1; k <= n_lat_keys; k++) {
        key = lat_keys[k]
        printf("%s %.4f %d\n", key, lat_pitch_sum[key]/lat_n[key], lat_n[key]) >> out3
    }
    close(out3)

    # --- Plot 4 output: orientation_error_deg, mean_eta, stddev_eta, n ---
    out4 = "plot4_eta_vs_orientation_error.dat"
    print "# orientation_error_deg mean_eta stddev_eta n" > out4
    for (i = 0; i < N_ORIENT_ERR_BINS; i++) {
        if (orient_n[i] == 0) continue
        n = orient_n[i]
        me = orient_eta_sum[i] / n
        sde = sqrt(orient_eta_sumsq[i]/n - me*me < 0 ? 0 : orient_eta_sumsq[i]/n - me*me)
        printf("%.1f %.6f %.6f %d\n", i + 0.5, me, sde, n) >> out4
    }
    close(out4)

    # --- Plot 5 output: cloud_cover, mean_eta, mean_T_operating_sigma, n ---
    out5 = "plot5_eta_vs_cloud.dat"
    print "# cloud_cover mean_eta mean_T_operating_sigma n" > out5
    for (i = 0; i < N_CLOUD_BINS; i++) {
        if (cloud_n[i] == 0) continue
        printf("%.3f %.6f %.4f %d\n", i*0.05 + 0.025, cloud_eta_sum[i]/cloud_n[i], cloud_sigma_sum[i]/cloud_n[i], cloud_n[i]) >> out5
    }
    close(out5)

    # --- Plot 6 output: delta_from_steady_state_C, mean_15min_delta_C, n ---
    out6 = "plot6_thermal_relaxation.dat"
    print "# delta_from_ss_C mean_15min_change_C n" > out6
    for (i = 0; i < N_RELAX_BINS; i++) {
        if (relax_n[i] == 0) continue
        printf("%.2f %.4f %d\n", -150.0 + i*RELAX_BIN_WIDTH_C + RELAX_BIN_WIDTH_C/2.0, relax_delta_sum[i]/relax_n[i], relax_n[i]) >> out6
    }
    close(out6)

    # --- Plot 7 output: elevation_m, mean_T_operating_C, mean_eta, n ---
    out7 = "plot7_vs_elevation.dat"
    print "# elevation_m mean_T_operating_C mean_eta n" > out7
    for (i = 0; i < N_ALT_BINS; i++) {
        if (alt_n[i] == 0) continue
        printf("%.1f %.4f %.6f %d\n", i*ALT_BIN_WIDTH_M + ALT_BIN_WIDTH_M/2.0, alt_T_sum[i]/alt_n[i], alt_eta_sum[i]/alt_n[i], alt_n[i]) >> out7
    }
    close(out7)

    # --- Plot 8 output: wind_speed, mean_T_operating_sigma, n ---
    out8 = "plot8_sigma_vs_wind.dat"
    print "# wind_speed mean_T_operating_sigma n" > out8
    for (k = 1; k <= n_wind_keys; k++) {
        key = wind_keys[k]
        printf("%s %.4f %d\n", key, wind_sigma_sum[key]/wind_n[key], wind_n[key]) >> out8
    }
    close(out8)

    printf("Wrote 8 aggregate files: plot1_diurnal.dat ... plot8_sigma_vs_wind.dat\n") > "/dev/stderr"
}
