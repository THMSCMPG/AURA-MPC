#!/usr/bin/awk -f
#
# preprocess_for_plots.awk
#
# Single-pass, memory-bounded aggregation over RK4TRAN lattice CSV output,
# producing small binned RAW-ACCUMULATOR files (sum, sum-of-squares, count
# per bin) for merge_chunk_aggregates.py to combine across multiple chunks,
# and ultimately for plot_synthetic_data.gp to render.
#
# CHUNKED PIPELINE NOTE: this script emits RAW sums/sumsq/counts per bin,
# NOT final means/stddevs -- this is deliberate so that running it separately
# on each location-chunk (see main.f90's --loc-range) and then summing the
# raw accumulators across chunks gives the mathematically EXACT same result
# as running it once on the full concatenated dataset. Averaging already-
# computed per-chunk means would NOT be correct (a mean of means isn't the
# true mean unless every chunk has equal weight, and never for stddev).
# merge_chunk_aggregates.py does that final sum + mean/stddev computation
# after all chunks have been processed and their aggregate files collected.
#
# WHY THIS EXISTS: the source data would be many TB as a single CSV -- far
# too large to load into any plotting tool's memory, and even gnuplot's
# native streaming can't usefully render billions of points. This script
# computes binned aggregates for 8 useful relationships in ONE pass through
# each chunk, using fixed-size associative arrays keyed by bin index --
# memory usage is bounded by (number of bins x number of tracked
# statistics), completely independent of row count, whether run on one
# chunk or (if you skip chunking) the whole dataset at once.
#
# Usage (per chunk, in the streaming pipeline):
#   awk -f preprocess_for_plots.awk -v out_prefix=chunk_003_ lattice_loc*.csv
# Usage (single-file / non-chunked, e.g. on the held-out validation set):
#   awk -f preprocess_for_plots.awk lattice_batches/*.csv
#
# Output: writes 8 small .dat files (out_prefix + plotN_*.dat, a few KB to a
# few hundred KB each, regardless of input size) to the current directory.

BEGIN {
    FS = ","
    row_count = 0
    PROGRESS_EVERY = 50000000
    if (out_prefix == "") out_prefix = ""

    N_HOUR_BINS = 48
    TAMB_BIN_WIDTH_C = 2.0
    N_TAMB_BINS = 60
    N_ORIENT_ERR_BINS = 140
    N_CLOUD_BINS = 21
    RELAX_BIN_WIDTH_C = 5.0
    N_RELAX_BINS = 60
    ALT_BIN_WIDTH_M = 200.0
    N_ALT_BINS = 30
}

FNR == 1 { next }  # skip header row of EACH input file

{
    row_count++
    if (row_count % PROGRESS_EVERY == 0) {
        printf("  ...processed %d rows\n", row_count) > "/dev/stderr"
    }

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

    hb = int(hour * 2)
    if (hb < 0) hb = 0
    if (hb >= N_HOUR_BINS) hb = N_HOUR_BINS - 1
    hour_n[hb]++
    hour_T_sum[hb] += T_operating_C; hour_T_sumsq[hb] += T_operating_C * T_operating_C
    hour_eta_sum[hb] += eta; hour_eta_sumsq[hb] += eta * eta

    tb = int((T_amb_C + 50.0) / TAMB_BIN_WIDTH_C)
    if (tb < 0) tb = 0
    if (tb >= N_TAMB_BINS) tb = N_TAMB_BINS - 1
    tamb_n[tb]++
    tamb_eta_sum[tb] += eta

    lat_key = sprintf("%.1f", lat)
    lat_n[lat_key]++
    lat_pitch_sum[lat_key] += optimal_pitch
    if (!(lat_key in lat_seen)) { lat_seen[lat_key] = 1; lat_keys[++n_lat_keys] = lat_key }

    ob = int(orientation_error)
    if (ob < 0) ob = 0
    if (ob >= N_ORIENT_ERR_BINS) ob = N_ORIENT_ERR_BINS - 1
    orient_n[ob]++
    orient_eta_sum[ob] += eta; orient_eta_sumsq[ob] += eta * eta

    cb = int(cloud_cover * 20.0)
    if (cb < 0) cb = 0
    if (cb >= N_CLOUD_BINS) cb = N_CLOUD_BINS - 1
    cloud_n[cb]++
    cloud_eta_sum[cb] += eta
    cloud_sigma_sum[cb] += T_operating_sigma

    delta_from_ss_C = (T_panel_initial_K - T_operating_K)
    rb = int((delta_from_ss_C + 150.0) / RELAX_BIN_WIDTH_C)
    if (rb < 0) rb = 0
    if (rb >= N_RELAX_BINS) rb = N_RELAX_BINS - 1
    relax_n[rb]++
    relax_delta_sum[rb] += (T_after_15min_K - T_panel_initial_K)

    ab = int(alt / ALT_BIN_WIDTH_M)
    if (ab < 0) ab = 0
    if (ab >= N_ALT_BINS) ab = N_ALT_BINS - 1
    alt_n[ab]++
    alt_T_sum[ab] += T_operating_C
    alt_eta_sum[ab] += eta

    wind_key = sprintf("%.2f", wind_speed)
    wind_n[wind_key]++
    wind_sigma_sum[wind_key] += T_operating_sigma
    if (!(wind_key in wind_seen)) { wind_seen[wind_key] = 1; wind_keys[++n_wind_keys] = wind_key }
}

END {
    printf("Total rows processed: %d\n", row_count) > "/dev/stderr"

    # --- Plot 1 raw accumulators: hour_bin sum_T sumsq_T sum_eta sumsq_eta n ---
    out1 = out_prefix "plot1_diurnal.raw"
    print "# hour_bin sum_T_C sumsq_T_C sum_eta sumsq_eta n" > out1
    for (i = 0; i < N_HOUR_BINS; i++) {
        if (hour_n[i] == 0) continue
        printf("%d %.6f %.6f %.6f %.6f %d\n", i, hour_T_sum[i], hour_T_sumsq[i], hour_eta_sum[i], hour_eta_sumsq[i], hour_n[i]) >> out1
    }
    close(out1)

    out2 = out_prefix "plot2_eta_vs_tamb.raw"
    print "# tamb_bin sum_eta n" > out2
    for (i = 0; i < N_TAMB_BINS; i++) {
        if (tamb_n[i] == 0) continue
        printf("%d %.6f %d\n", i, tamb_eta_sum[i], tamb_n[i]) >> out2
    }
    close(out2)

    out3 = out_prefix "plot3_optimal_pitch_vs_lat.raw"
    print "# latitude sum_pitch n" > out3
    for (k = 1; k <= n_lat_keys; k++) {
        key = lat_keys[k]
        printf("%s %.6f %d\n", key, lat_pitch_sum[key], lat_n[key]) >> out3
    }
    close(out3)

    out4 = out_prefix "plot4_eta_vs_orientation_error.raw"
    print "# orient_bin sum_eta sumsq_eta n" > out4
    for (i = 0; i < N_ORIENT_ERR_BINS; i++) {
        if (orient_n[i] == 0) continue
        printf("%d %.6f %.6f %d\n", i, orient_eta_sum[i], orient_eta_sumsq[i], orient_n[i]) >> out4
    }
    close(out4)

    out5 = out_prefix "plot5_eta_vs_cloud.raw"
    print "# cloud_bin sum_eta sum_sigma n" > out5
    for (i = 0; i < N_CLOUD_BINS; i++) {
        if (cloud_n[i] == 0) continue
        printf("%d %.6f %.6f %d\n", i, cloud_eta_sum[i], cloud_sigma_sum[i], cloud_n[i]) >> out5
    }
    close(out5)

    out6 = out_prefix "plot6_thermal_relaxation.raw"
    print "# relax_bin sum_delta n" > out6
    for (i = 0; i < N_RELAX_BINS; i++) {
        if (relax_n[i] == 0) continue
        printf("%d %.6f %d\n", i, relax_delta_sum[i], relax_n[i]) >> out6
    }
    close(out6)

    out7 = out_prefix "plot7_vs_elevation.raw"
    print "# alt_bin sum_T sum_eta n" > out7
    for (i = 0; i < N_ALT_BINS; i++) {
        if (alt_n[i] == 0) continue
        printf("%d %.6f %.6f %d\n", i, alt_T_sum[i], alt_eta_sum[i], alt_n[i]) >> out7
    }
    close(out7)

    out8 = out_prefix "plot8_sigma_vs_wind.raw"
    print "# wind_speed sum_sigma n" > out8
    for (k = 1; k <= n_wind_keys; k++) {
        key = wind_keys[k]
        printf("%s %.6f %d\n", key, wind_sigma_sum[key], wind_n[key]) >> out8
    }
    close(out8)

    printf("Wrote 8 raw-accumulator files (%splot1..8*.raw)\n", out_prefix) > "/dev/stderr"
}

