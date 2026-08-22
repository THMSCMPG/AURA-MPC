! program using just RK4 and RK45 to produce synthetic data for training AURA_MPC PINN
! will generate data for:
        ! independent: (loosely coupled)
                ! longitude, latitude, altitude (long, lat, alt)
                ! time of year, time of day (minute, hour, day, month, year)
                ! ambient temperature
                ! wind speed
                ! wind direction
                ! humidity
                ! irradiance
                ! cloud coverage
                ! barometric pressure
                ! height of PV
                ! pitch of PV
                ! roll of PV
                ! yaw of PV
        ! dependent: (should be fully derived using all independent variables)
                ! operating temperature
                ! electrical efficiency
! first we generate lazy data for all independent variables using strict boundary conditions
! then we use that lazy data to generate working data
! working data will be like a map or look up table that says:
        ! when independent variable x1 is this, boundary conditions of x(2:13) change
        ! and dependent variables y(1:2) become this
! working data will be produced to simulate all possible scenarios at 3 different grids
        ! spaceous -> plenty of gaps, very easy going, covers broad strokes
        ! comfortable -> tighter middle of the road, covers a bit more detail
        ! cramped -> covers the most detail possible
! output will be 3 main csv files (@ /) full of working data and 13 lazy csv files (@ /lazy/)
!=======================================================================
! BLOCK 1: shared env state + PV thermal ODE
!=======================================================================
module pv_env_state
    use, intrinsic :: ISO_C_BINDING
    implicit none

    ! parsed from location_str / time_str
    real(c_double) :: env_lon, env_lat, env_alt
    real(c_double) :: env_min, env_hour, env_day, env_month, env_year

    ! independent real(c_double) vars (ordered loosely→tightly coupled)
    real(c_double) :: env_T_amb    ! K
    real(c_double) :: env_wind     ! m/s
    real(c_double) :: env_winddir  ! deg from north
    real(c_double) :: env_humid    ! fraction [0,1]
    real(c_double) :: env_irr      ! W/m² (extraterrestrial base)
    real(c_double) :: env_cloud    ! fraction [0,1]
    real(c_double) :: env_press    ! Pa
    real(c_double) :: env_pv_h     ! m above ground
    real(c_double) :: env_pitch    ! deg tilt from horizontal
    real(c_double) :: env_roll     ! deg
    real(c_double) :: env_yaw      ! deg from south

    ! derived before each ODE call — do not set directly
    real(c_double) :: env_G_eff    ! effective irradiance [W/m²]

end module pv_env_state

!=======================================================================
module pv_ode_module
    use, intrinsic :: ISO_C_BINDING
    use pv_env_state
    implicit none

    real(c_double), parameter :: PI       = 3.14159265358979_c_double
    real(c_double), parameter :: D2R      = PI / 180.0_c_double
    real(c_double), parameter :: ETA_REF  = 0.20_c_double    ! c-Si STC efficiency
    real(c_double), parameter :: BETA_T   = 0.004_c_double   ! /K temperature coeff
    real(c_double), parameter :: T_STC    = 298.15_c_double  ! K
    real(c_double), parameter :: ALPHA    = 0.90_c_double    ! absorptivity
    real(c_double), parameter :: C_TH     = 10000.0_c_double ! J/m²/K panel heat cap

contains

    ! Compute env_G_eff from location+time+panel geometry+cloud.
    ! Call this once per sample before calling RK45.
    subroutine compute_G_eff()
        real(c_double) :: decl, H, sin_elev, cos_aoi
        real(c_double) :: az_sun, az_panel

        ! Solar declination (deg)
        decl = 23.45_c_double * sin(D2R * (284.0_c_double + env_day) &
               * 360.0_c_double / 365.0_c_double)

        ! Hour angle (deg): 15°/hr from solar noon
        H = 15.0_c_double * (env_hour + env_min / 60.0_c_double - 12.0_c_double)

        ! Solar elevation above horizon
        sin_elev = sin(D2R*env_lat)*sin(D2R*decl) &
                 + cos(D2R*env_lat)*cos(D2R*decl)*cos(D2R*H)

        ! BUGFIX (found during Section 1 sprint smoke-test): the original code
        ! floored sin_elev to 0 here, then used the floored value in the AOI
        ! formula below -- but the second AOI term (sqrt(1-sin_elev**2)*cos(...)
        ! *sin(pitch)) does NOT vanish when sin_elev is floored to exactly 0,
        ! so a tilted panel could pick up spurious nonzero G_eff purely from
        ! azimuth geometry even when the true sun elevation is negative (sun
        ! below horizon -- night, or polar night at high latitude). Fix: check
        ! the TRUE (unclamped) sin_elev and zero G_eff outright when the sun
        ! is below the horizon, instead of flooring and continuing.
        if (sin_elev <= 0.0_c_double) then
            env_G_eff = 0.0_c_double
            return
        end if

        ! Panel AOI: simplified dot product of sun vector with panel normal
        ! panel normal tilted by pitch toward south (yaw offset)
        az_sun   = atan2(sin(D2R*H), cos(D2R*H)*sin(D2R*env_lat) &
                   - tan(D2R*decl)*cos(D2R*env_lat)) / D2R
        az_panel = env_yaw  ! yaw is deviation from south

        cos_aoi = sin_elev * cos(D2R*env_pitch) &
                + sqrt(max(0.0_c_double, 1.0_c_double - sin_elev**2)) &
                * cos(D2R*(az_sun - az_panel)) * sin(D2R*env_pitch)
        cos_aoi = max(0.0_c_double, cos_aoi)

        ! roll: attenuates at extremes
        cos_aoi = cos_aoi * cos(D2R * env_roll)

        env_G_eff = env_irr * cos_aoi * (1.0_c_double - env_cloud)
    end subroutine compute_G_eff

    ! ODE: y(1) = T_cell [K],  y(2) = eta [-]
    function pv_ode(t, y) result(dydt)
        real(c_double), intent(in) :: t
        real(c_double), dimension(:), intent(in) :: y
        real(c_double), dimension(size(y)) :: dydt
        real(c_double) :: h_conv, dTdt

        ! McAdams + height exposure factor
        h_conv = (5.7_c_double + 3.8_c_double * env_wind) &
               * min(1.2_c_double, 0.5_c_double + env_pv_h / 4.0_c_double)

        dTdt    = (env_G_eff * ALPHA - env_G_eff * y(2) &
                  - h_conv * (y(1) - env_T_amb)) / C_TH

        dydt(1) = dTdt
        dydt(2) = -ETA_REF * BETA_T * dTdt
    end function pv_ode

end module pv_ode_module
program AURA_MFP
    use lattice_pools
    use pv_env_state
    use pv_ode_module
    use MC_UQ_Library, only: box_muller_sample
    use, intrinsic :: ISO_C_BINDING
    implicit none

    ! --- fixed assumptions (not part of the lattice; see checklist Section 1) ---
    real(c_double), parameter :: PV_HEIGHT_DEFAULT = 1.5_c_double   ! m; not swept
    real(c_double), parameter :: SOLAR_CONSTANT = 1361.0_c_double   ! W/m^2 TOA; weather no longer carries irradiance
    integer,        parameter :: N_MC = 100                        ! MC replicates/point
    real(c_double), parameter :: MC_SIGMA_FRAC = 0.05_c_double      ! 5% input perturbation
    real(c_double), parameter :: TRANSIENT_T = 900.0_c_double       ! 15 min, seconds

    ! --- T_panel_initial: independent axis, fixed physical range (per Tommy: not
    !     ambient/steady-state-coupled -- a real panel's temperature reflects
    !     thermal history/lag, so the PINN needs to see it swept independently) ---
    integer, parameter :: N_T_PANEL = 5
    real(c_double), parameter :: T_PANEL_LO_C = -40.0_c_double
    real(c_double), parameter :: T_PANEL_HI_C = 90.0_c_double

    ! --- T_amb: NOT a fixed pool -- computed at runtime per (location,date) from
    !     a parametric diurnal/seasonal model (per Tommy: use a simpler parametric
    !     model instead of pulling a real climate dataset), sampled at a number of
    !     daylight hours proportional to day length (fewer points on short days). ---
    integer, parameter :: N_AMB_MAX = 48   ! cap on samples/date, up to ~30min spacing on a 24hr polar-day

    integer :: i_loc, i_time, i_amb, i_wx, i_ori, i_panel, mc, best_idx, u
    integer :: n_amb_points, i_amb_hour
    real(c_double) :: h_conv, power, best_power
    real(c_double) :: G_eff_cache(N_ORIENTATIONS), T_ss_cache(N_ORIENTATIONS), eta_ss_cache(N_ORIENTATIONS)
    real(c_double) :: opt_pitch, opt_yaw
    real(c_double) :: T_ss, eta_ss, T_sigma, eta_sigma
    real(c_double) :: T_panel_init, T_15, eta_15, T_15_sigma, eta_15_sigma
    real(c_double) :: pitch_err, roll_err, yaw_err, orient_err
    real(c_double) :: G_pert, Tamb_pert, T_mc_ss, eta_mc_ss, T_mc_15, eta_mc_15
    real(c_double) :: sum_T, sum_T2, sum_eta, sum_eta2, sum_T15, sum_T15_2, sum_eta15, sum_eta15_2
    real(c_double) :: sunrise_hr, sunset_hr, day_length_hr, amb_hour, T_amb_K
    character(len=256) :: out_path, timestamp
    character(len=16) :: dt_vals(3)
    integer :: dt(8)
    logical :: smoke_test
    integer :: n_loc_use, n_time_use, n_wx_use
    integer :: loc_start, loc_end
    integer(8) :: row_count
    logical :: use_explicit_locs
    integer, allocatable :: loc_indices_to_process(:)
    character(len=512) :: loc_indices_file_path
    integer :: n_explicit, i_read, ios, idx_unit, i_pos

    smoke_test = .false.
    use_explicit_locs = .false.
    n_loc_use = N_LOCATIONS; n_time_use = N_TIMES; n_wx_use = N_WEATHER
    loc_start = 1; loc_end = N_LOCATIONS

    if (command_argument_count() >= 1) then
        block
            character(len=32) :: arg
            call get_command_argument(1, arg)
            if (trim(arg) == "--smoke") smoke_test = .true.
            if (trim(arg) == "--loc-range") then
                ! Chunked generation: process only locations [start,end] (1-indexed,
                ! inclusive). Locations are the outermost loop and fully independent
                ! (each row only ever depends on ONE location's lon/lat/alt), so this
                ! is a clean, embarrassingly-parallel chunking axis -- run many
                ! invocations with disjoint ranges across cluster workers, or process
                ! chunks sequentially on one machine to bound peak disk usage (see
                ! tools/streaming_pipeline.py for the generate->train->plot->delete
                ! orchestration this enables).
                if (command_argument_count() < 3) then
                    print *, "ERROR: --loc-range requires START END, e.g. --loc-range 1 10"
                    stop 1
                end if
                block
                    character(len=32) :: arg_start, arg_end
                    call get_command_argument(2, arg_start)
                    call get_command_argument(3, arg_end)
                    read(arg_start, *) loc_start
                    read(arg_end, *) loc_end
                end block
                if (loc_start < 1 .or. loc_end > N_LOCATIONS .or. loc_start > loc_end) then
                    print *, "ERROR: invalid --loc-range: ", loc_start, loc_end, &
                             " (valid: 1 to ", N_LOCATIONS, ")"
                    stop 1
                end if
                print *, "LOCATION CHUNK: processing locations ", loc_start, " to ", loc_end, &
                         " of ", N_LOCATIONS
            end if
            if (trim(arg) == "--loc-indices-file") then
                ! Randomized (non-contiguous) location subset, for the expanded
                ! ~1000-point pool (see tools/build_lattice_pools.py) -- reads a
                ! text file with one 1-indexed location index per line, generated
                ! by tools/select_training_locations.py. Unlike --loc-range, the
                ! indices need not be contiguous or sorted.
                if (command_argument_count() < 2) then
                    print *, "ERROR: --loc-indices-file requires a file path"
                    stop 1
                end if
                call get_command_argument(2, loc_indices_file_path)
                use_explicit_locs = .true.

                ! first pass: count lines
                n_explicit = 0
                open(newunit=idx_unit, file=trim(loc_indices_file_path), status='old', action='read', iostat=ios)
                if (ios /= 0) then
                    print *, "ERROR: could not open --loc-indices-file: ", trim(loc_indices_file_path)
                    stop 1
                end if
                do
                    read(idx_unit, *, iostat=ios) i_read
                    if (ios /= 0) exit
                    n_explicit = n_explicit + 1
                end do
                close(idx_unit)

                if (n_explicit < 1) then
                    print *, "ERROR: --loc-indices-file is empty: ", trim(loc_indices_file_path)
                    stop 1
                end if

                ! second pass: actually read the indices
                allocate(loc_indices_to_process(n_explicit))
                open(newunit=idx_unit, file=trim(loc_indices_file_path), status='old', action='read')
                do i_pos = 1, n_explicit
                    read(idx_unit, *) loc_indices_to_process(i_pos)
                end do
                close(idx_unit)

                do i_pos = 1, n_explicit
                    if (loc_indices_to_process(i_pos) < 1 .or. loc_indices_to_process(i_pos) > N_LOCATIONS) then
                        print *, "ERROR: index out of range in --loc-indices-file: ", &
                                 loc_indices_to_process(i_pos), " (valid: 1 to ", N_LOCATIONS, ")"
                        stop 1
                    end if
                end do

                print *, "LOCATION SUBSET: processing ", n_explicit, " explicit (non-contiguous) indices from ", &
                         trim(loc_indices_file_path)
            end if
        end block
    end if
    if (smoke_test) then
        n_loc_use = min(3, N_LOCATIONS)
        n_time_use = min(3, N_TIMES)
        n_wx_use = min(3, N_WEATHER)
        loc_start = 1; loc_end = n_loc_use
        if (command_argument_count() >= 2) then
            block
                character(len=32) :: arg2
                integer :: n
                call get_command_argument(2, arg2)
                read(arg2, *) n
                n_loc_use = min(n, N_LOCATIONS)
                n_time_use = min(n, N_TIMES)
                n_wx_use = min(n, N_WEATHER)
                loc_end = n_loc_use
            end block
        end if
        print *, "SMOKE TEST: ", n_loc_use, "loc x", n_time_use, "time x", n_wx_use, "wx"
    end if

    ! Build the unified list of location indices to process -- either the
    ! explicit (possibly non-contiguous) list just read above, or the
    ! contiguous [loc_start,loc_end] range (default / --loc-range / --smoke).
    ! The main loop below iterates this array either way, so both modes
    ! share the exact same body -- no duplicated logic.
    if (.not. use_explicit_locs) then
        allocate(loc_indices_to_process(loc_end - loc_start + 1))
        do i_pos = 1, size(loc_indices_to_process)
            loc_indices_to_process(i_pos) = loc_start + i_pos - 1
        end do
    end if

    call execute_command_line("mkdir -p lattice_batches", wait=.true.)
    call date_and_time(dt_vals(1), dt_vals(2), dt_vals(3), dt)
    write(timestamp, '(I4.4,I2.2,I2.2,"_",I2.2,I2.2,I2.2)') dt(1),dt(2),dt(3),dt(5),dt(6),dt(7)
    if (smoke_test) then
        out_path = "lattice_batches/smoke_" // trim(timestamp) // ".csv"
    else if (use_explicit_locs) then
        block
            character(len=32) :: idx_tag
            write(idx_tag, '("idx",I0,"n_")') size(loc_indices_to_process)
            out_path = "lattice_batches/lattice_" // trim(idx_tag) // trim(timestamp) // ".csv"
        end block
    else if (loc_start /= 1 .or. loc_end /= N_LOCATIONS) then
        block
            character(len=32) :: range_tag
            write(range_tag, '("loc",I0,"-",I0,"_")') loc_start, loc_end
            out_path = "lattice_batches/lattice_" // trim(range_tag) // trim(timestamp) // ".csv"
        end block
    else
        out_path = "lattice_batches/lattice_" // trim(timestamp) // ".csv"
    end if

    open(newunit=u, file=trim(out_path), status='replace', action='write')
    write(u, '(A)') "lon,lat,alt,minute,hour,day_of_year,month,year," // &
                     "T_amb,wind_speed,wind_dir,humidity,irradiance,cloud_cover,pressure,pv_height," // &
                     "pitch,roll,yaw,T_operating,T_operating_sigma,eta,eta_sigma," // &
                     "optimal_pitch,optimal_roll,optimal_yaw,pitch_error,roll_error,yaw_error,orientation_error," // &
                     "T_panel_initial,T_after_15min,T_after_15min_sigma,eta_after_15min,eta_after_15min_sigma"

    call random_seed()
    row_count = 0_8

    do i_pos = 1, size(loc_indices_to_process)
        i_loc = loc_indices_to_process(i_pos)
        env_lon = LOC_LON(i_loc)
        env_lat = LOC_LAT(i_loc)
        env_alt = LOC_ALT(i_loc)

        do i_time = 1, n_time_use
            env_min   = TIME_MIN(i_time)
            env_day   = TIME_DAY(i_time)
            env_month = TIME_MONTH(i_time)
            env_year  = TIME_YEAR(i_time)

            call solar_day_info(env_lat, env_day, sunrise_hr, sunset_hr, day_length_hr)
            ! ~30min hour-of-day spacing (per Tommy: cover more times than just
            ! noon). N_AMB_MAX=48 covers a 24hr polar-day at this spacing.
            n_amb_points = max(1, min(N_AMB_MAX, nint(day_length_hr / 0.5_c_double)))

            do i_amb = 1, n_amb_points
                ! UNIFIED (was previously decoupled): this same sampled hour now
                ! drives BOTH the parametric T_amb model AND env_hour, which
                ! feeds sun-angle geometry in compute_G_eff. Previously env_hour
                ! stayed fixed at solar noon while T_amb sampled across the full
                ! daylight window independently -- meaning a row's panel
                ! orientation was always optimized for noon sun position even
                ! when paired with a dawn/dusk temperature. Unifying removes
                ! that mismatch and is real, not just cosmetic: G_eff (hence
                ! steady-state T/eta) now genuinely varies across the day.
                if (n_amb_points == 1) then
                    amb_hour = 12.0_c_double
                else
                    amb_hour = sunrise_hr + (sunset_hr - sunrise_hr) &
                             * real(i_amb - 1, c_double) / real(n_amb_points - 1, c_double)
                end if
                env_hour = amb_hour
                call parametric_T_amb(env_lat, env_alt, env_day, amb_hour, sunrise_hr, sunset_hr, day_length_hr, T_amb_K)
                env_T_amb = T_amb_K

                do i_wx = 1, n_wx_use
                    env_wind    = WX_WIND(i_wx)
                    env_winddir = WX_WINDDIR(i_wx)
                    env_humid   = WX_HUMID(i_wx)
                    env_irr     = SOLAR_CONSTANT
                    env_cloud   = WX_CLOUD(i_wx)
                    env_press   = WX_PRESS(i_wx)
                    env_pv_h    = PV_HEIGHT_DEFAULT

                    h_conv = (5.7_c_double + 3.8_c_double * env_wind) &
                           * min(1.2_c_double, 0.5_c_double + env_pv_h / 4.0_c_double)

                    ! --- Pass 1: steady state at all 144 orientations (depends on
                    !     T_amb/weather, NOT on T_panel_initial -- steady state is
                    !     independent of starting point by definition) ---
                    do i_ori = 1, N_ORIENTATIONS
                        env_pitch = ORI_PITCH(i_ori)
                        env_roll  = ORI_ROLL(i_ori)
                        env_yaw   = ORI_YAW(i_ori)
                        call compute_G_eff()
                        G_eff_cache(i_ori) = env_G_eff
                        call steady_state(env_G_eff, h_conv, env_T_amb, T_ss_cache(i_ori), eta_ss_cache(i_ori))
                    end do

                    best_idx = 1
                    best_power = G_eff_cache(1) * eta_ss_cache(1)
                    do i_ori = 2, N_ORIENTATIONS
                        power = G_eff_cache(i_ori) * eta_ss_cache(i_ori)
                        if (power > best_power) then
                            best_power = power
                            best_idx = i_ori
                        end if
                    end do
                    opt_pitch = ORI_PITCH(best_idx)
                    opt_yaw   = ORI_YAW(best_idx)

                    ! --- Pass 2: for each orientation AND each independent starting
                    !     panel temperature, write one row ---
                    do i_ori = 1, N_ORIENTATIONS
                        T_ss   = T_ss_cache(i_ori)
                        eta_ss = eta_ss_cache(i_ori)

                        pitch_err = ORI_PITCH(i_ori) - opt_pitch
                        roll_err  = 0.0_c_double
                        yaw_err   = ORI_YAW(i_ori) - opt_yaw
                        orient_err = sqrt(pitch_err**2 + roll_err**2 + yaw_err**2)

                        do i_panel = 1, N_T_PANEL
                            if (N_T_PANEL == 1) then
                                T_panel_init = (T_PANEL_LO_C + T_PANEL_HI_C) / 2.0_c_double + 273.15_c_double
                            else
                                T_panel_init = T_PANEL_LO_C + (T_PANEL_HI_C - T_PANEL_LO_C) &
                                             * real(i_panel - 1, c_double) / real(N_T_PANEL - 1, c_double) &
                                             + 273.15_c_double
                            end if
                            call transient_state(G_eff_cache(i_ori), h_conv, env_T_amb, T_panel_init, &
                                                  TRANSIENT_T, T_15, eta_15)

                            ! MC uncertainty: perturb G_eff and T_amb by 5% (clipped
                            ! Gaussian), recompute BOTH steady-state and transient
                            ! closed forms each draw, sharing the same N_MC loop.
                            sum_T = 0.0_c_double; sum_T2 = 0.0_c_double
                            sum_eta = 0.0_c_double; sum_eta2 = 0.0_c_double
                            sum_T15 = 0.0_c_double; sum_T15_2 = 0.0_c_double
                            sum_eta15 = 0.0_c_double; sum_eta15_2 = 0.0_c_double
                            do mc = 1, N_MC
                                G_pert = max(0.0_c_double, G_eff_cache(i_ori) * (1.0_c_double + MC_SIGMA_FRAC * clipped_normal(1)))
                                Tamb_pert = env_T_amb * (1.0_c_double + MC_SIGMA_FRAC * clipped_normal(2))
                                call steady_state(G_pert, h_conv, Tamb_pert, T_mc_ss, eta_mc_ss)
                                call transient_state(G_pert, h_conv, Tamb_pert, T_panel_init, TRANSIENT_T, T_mc_15, eta_mc_15)
                                sum_T = sum_T + T_mc_ss; sum_T2 = sum_T2 + T_mc_ss*T_mc_ss
                                sum_eta = sum_eta + eta_mc_ss; sum_eta2 = sum_eta2 + eta_mc_ss*eta_mc_ss
                                sum_T15 = sum_T15 + T_mc_15; sum_T15_2 = sum_T15_2 + T_mc_15*T_mc_15
                                sum_eta15 = sum_eta15 + eta_mc_15; sum_eta15_2 = sum_eta15_2 + eta_mc_15*eta_mc_15
                            end do
                            T_sigma = sqrt(max(0.0_c_double, sum_T2/N_MC - (sum_T/N_MC)**2))
                            eta_sigma = sqrt(max(0.0_c_double, sum_eta2/N_MC - (sum_eta/N_MC)**2))
                            T_15_sigma = sqrt(max(0.0_c_double, sum_T15_2/N_MC - (sum_T15/N_MC)**2))
                            eta_15_sigma = sqrt(max(0.0_c_double, sum_eta15_2/N_MC - (sum_eta15/N_MC)**2))

                            write(u, '(34(F14.6,","),F14.6)') &
                                env_lon, env_lat, env_alt, &
                                env_min, env_hour, env_day, env_month, env_year, &
                                env_T_amb, env_wind, env_winddir, env_humid, env_irr, env_cloud, env_press, env_pv_h, &
                                ORI_PITCH(i_ori), ORI_ROLL(i_ori), ORI_YAW(i_ori), &
                                T_ss, T_sigma, eta_ss, eta_sigma, &
                                opt_pitch, 0.0_c_double, opt_yaw, &
                                pitch_err, roll_err, yaw_err, orient_err, &
                                T_panel_init, T_15, T_15_sigma, eta_15, eta_15_sigma

                            row_count = row_count + 1_8
                        end do
                    end do
                end do
            end do
        end do
    end do

    close(u)
    print '(A,A,A,I0,A)', " wrote ", trim(out_path), " (", row_count, " rows)"

    contains

    ! Exact closed-form steady state -- see checklist for derivation/validation.
    subroutine steady_state(G_eff, h_conv, T_amb, T_ss, eta_ss)
        real(c_double), intent(in)  :: G_eff, h_conv, T_amb
        real(c_double), intent(out) :: T_ss, eta_ss
        real(c_double) :: num, den
        real(c_double), parameter :: DEN_FLOOR = 1.0e-3_c_double

        num = G_eff*ALPHA - G_eff*ETA_REF - G_eff*ETA_REF*BETA_T*T_STC + h_conv*T_amb
        den = h_conv - G_eff*ETA_REF*BETA_T
        if (abs(den) < DEN_FLOOR) den = sign(DEN_FLOOR, den)
        T_ss = num / den
        eta_ss = ETA_REF * (1.0_c_double - BETA_T * (T_ss - T_STC))
    end subroutine steady_state

    ! Exact closed-form transient: T(t) = T_ss + (T0-T_ss)*exp(-den*t/C_TH).
    ! Validated against scipy solve_ivp to ~1e-11 K across several (G,h,Tamb,T0)
    ! combinations at t=900s. T_panel_init (T0) is the new independent axis --
    ! steady_state's T_ss/den are recomputed here rather than passed in, since
    ! this is called both from the main loop (cached G_eff, nominal T_amb) and
    ! from the MC loop (perturbed G_eff/T_amb) with different T_ss each time.
    subroutine transient_state(G_eff, h_conv, T_amb, T0, t, T_t, eta_t)
        real(c_double), intent(in)  :: G_eff, h_conv, T_amb, T0, t
        real(c_double), intent(out) :: T_t, eta_t
        real(c_double) :: num, den, T_ss, a_rate
        real(c_double), parameter :: DEN_FLOOR = 1.0e-3_c_double

        num = G_eff*ALPHA - G_eff*ETA_REF - G_eff*ETA_REF*BETA_T*T_STC + h_conv*T_amb
        den = h_conv - G_eff*ETA_REF*BETA_T
        if (abs(den) < DEN_FLOOR) den = sign(DEN_FLOOR, den)
        T_ss = num / den
        a_rate = -den / C_TH
        T_t = T_ss + (T0 - T_ss) * exp(a_rate * t)
        eta_t = ETA_REF * (1.0_c_double - BETA_T * (T_t - T_STC))
    end subroutine transient_state

    ! Standard normal sample clipped to +-4 sigma -- see checklist robustness note.
    function clipped_normal(which) result(z)
        integer, intent(in) :: which
        real(c_double) :: z
        real(c_double), parameter :: CLIP_SIGMA = 4.0_c_double
        z = box_muller_sample(which)
        z = max(-CLIP_SIGMA, min(CLIP_SIGMA, z))
    end function clipped_normal

    ! Sunrise/sunset hour (solar time, hours) and day length from latitude and
    ! day-of-year, reusing the same declination formula as compute_G_eff.
    ! Handles polar day (sun never sets) and polar night (never rises).
    subroutine solar_day_info(lat, day_of_year, sunrise_hr, sunset_hr, day_length_hr)
        real(c_double), intent(in)  :: lat, day_of_year
        real(c_double), intent(out) :: sunrise_hr, sunset_hr, day_length_hr
        real(c_double) :: decl, cos_H0, H0

        decl = 23.45_c_double * sin(D2R * (284.0_c_double + day_of_year) * 360.0_c_double / 365.0_c_double)
        cos_H0 = -tan(D2R*lat) * tan(D2R*decl)

        if (cos_H0 <= -1.0_c_double) then
            ! polar day: sun never sets
            sunrise_hr = 0.0_c_double; sunset_hr = 24.0_c_double; day_length_hr = 24.0_c_double
        else if (cos_H0 >= 1.0_c_double) then
            ! polar night: sun never rises
            sunrise_hr = 12.0_c_double; sunset_hr = 12.0_c_double; day_length_hr = 0.0_c_double
        else
            H0 = acos(cos_H0) / D2R   ! degrees
            day_length_hr = 2.0_c_double * H0 / 15.0_c_double
            sunrise_hr = 12.0_c_double - H0 / 15.0_c_double
            sunset_hr  = 12.0_c_double + H0 / 15.0_c_double
        end if
    end subroutine solar_day_info

    ! Parametric diurnal/seasonal ambient temperature model -- deliberately NOT
    ! a real climate dataset lookup (per Tommy: simpler parametric model instead).
    ! Drivers: latitude (annual mean + seasonal amplitude, both first-order real
    ! effects), day-of-year (seasonal phase, 6-month offset by hemisphere),
    ! elevation (standard 6.5 C/km environmental lapse rate), and hour-of-day
    ! (diurnal swing, sinusoidal between sunrise and sunset). This is a rough
    ! synthetic approximation, not calibrated against any specific station.
    subroutine parametric_T_amb(lat, elevation, day_of_year, hour, sunrise_hr, sunset_hr, day_length_hr, T_amb_K)
        real(c_double), intent(in)  :: lat, elevation, day_of_year, hour, sunrise_hr, sunset_hr, day_length_hr
        real(c_double), intent(out) :: T_amb_K
        real(c_double) :: T_annual_mean_C, seasonal_amp_C, peak_day, T_seasonal_C
        real(c_double), parameter :: DIURNAL_AMP_C = 8.0_c_double
        real(c_double), parameter :: LAPSE_C_PER_KM = 6.5_c_double

        T_annual_mean_C = 30.0_c_double - 0.5_c_double * abs(lat)
        seasonal_amp_C  = 3.0_c_double + 0.4_c_double * abs(lat)
        if (lat >= 0.0_c_double) then
            peak_day = 172.0_c_double   ! ~June 21, N hemisphere summer
        else
            peak_day = 355.0_c_double   ! ~Dec 21, S hemisphere summer
        end if
        T_seasonal_C = T_annual_mean_C + seasonal_amp_C * cos(2.0_c_double*PI*(day_of_year-peak_day)/365.0_c_double)
        T_seasonal_C = T_seasonal_C - LAPSE_C_PER_KM * (elevation / 1000.0_c_double)

        if (day_length_hr <= 0.0_c_double) then
            ! polar night: no diurnal cycle, use seasonal mean minus half swing
            T_amb_K = T_seasonal_C - DIURNAL_AMP_C/2.0_c_double + 273.15_c_double
        else
            ! sinusoidal diurnal shape, trough at sunrise/sunset, peak at midday
            T_amb_K = T_seasonal_C - DIURNAL_AMP_C/2.0_c_double &
                    + DIURNAL_AMP_C * sin(PI * max(0.0_c_double, min(1.0_c_double, &
                        (hour-sunrise_hr)/max(1.0e-6_c_double, sunset_hr-sunrise_hr)))) &
                    + 273.15_c_double
        end if
    end subroutine parametric_T_amb

end program AURA_MFP
