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
        sin_elev = max(0.0_c_double, sin_elev)   ! below horizon → 0

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
    use RK4TRAN
    use pv_env_state
    use pv_ode_module
    use, intrinsic :: ISO_C_BINDING
    implicit none

    ! --- grid sample counts (scaled for diversity) ---
    integer, parameter :: N_LAZY       = 200
    integer, parameter :: N_SPACIOUS   = 2000
    integer, parameter :: N_COMFORTABLE= 10000
    integer, parameter :: N_CRAMPED    = 50000

    ! --- boundary conditions (expanded for better coverage) ---
    real(c_double), parameter :: BC_TAMB(2)    = [233.15_c_double, 333.15_c_double]  ! K (-40 to +60°C, expanded for hot climates)
    real(c_double), parameter :: BC_WIND(2)    = [0.0_c_double,    30.0_c_double  ]  ! m/s (include storm winds)
    real(c_double), parameter :: BC_WINDDIR(2) = [0.0_c_double,    360.0_c_double ]  ! deg
    real(c_double), parameter :: BC_HUMID(2)   = [0.0_c_double,    1.0_c_double   ]  ! fraction
    real(c_double), parameter :: BC_IRR(2)     = [0.0_c_double,    1367.0_c_double]  ! W/m² (solar constant)
    real(c_double), parameter :: BC_CLOUD(2)   = [0.0_c_double,    1.0_c_double   ]  ! fraction
    real(c_double), parameter :: BC_PRESS(2)   = [50000.0_c_double,105000.0_c_double]! Pa (expanded for high altitude)
    real(c_double), parameter :: BC_PVH(2)     = [0.1_c_double,    10.0_c_double  ]  ! m (expanded range)
    real(c_double), parameter :: BC_PITCH(2)   = [-90.0_c_double,  90.0_c_double  ]  ! deg (added negative for south-facing north)
    real(c_double), parameter :: BC_ROLL(2)    = [-60.0_c_double,  60.0_c_double  ]  ! deg (expanded)
    real(c_double), parameter :: BC_YAW(2)     = [-180.0_c_double, 180.0_c_double ]  ! deg

    ! location and time string pools for lazy sampling (expanded for geographic diversity)
    integer, parameter :: N_LOC = 43
    character(len=32), parameter :: LOC_POOL(N_LOC) = [ &
        ! North America
        "-122.00  37.50   25.0         ", &  ! San Jose, CA
        " -87.63  41.88  181.0         ", &  ! Chicago, IL
        " -74.01  40.71   10.0         ", &  ! New York, NY
        "-104.99  39.74 1609.0         ", &  ! Denver, CO
        "-118.24  34.05   71.0         ", &  ! Los Angeles, CA
        " -96.69  32.82  143.0         ", &  ! Dallas, TX
        " -87.62  30.27   -1.0         ", &  ! New Orleans, LA
        "-120.50  48.42  343.0         ", &  ! Spokane, WA
        ! South America
        " -46.63 -23.55  760.0         ", &  ! São Paulo, Brazil
        " -68.15 -16.40 3640.0         ", &  ! La Paz, Bolivia
        " -58.38 -34.60   25.0         ", &  ! Buenos Aires, Argentina
        ! Europe
        "  -0.13  51.51   11.0         ", &  ! London, UK
        "   2.35  48.85   35.0         ", &  ! Paris, France
        "  13.40  52.52   34.0         ", &  ! Berlin, Germany
        "  -3.70  40.42  646.0         ", &  ! Madrid, Spain
        "  21.01  52.23  100.0         ", &  ! Warsaw, Poland
        ! Asia
        " 139.69  35.69   40.0         ", &  ! Tokyo, Japan
        " 103.82   1.35   15.0         ", &  ! Singapore
        " 113.27  23.13   12.0         ", &  ! Hong Kong
        " 120.16  30.27 2313.0         ", &  ! Chengdu, China
        "  88.40  27.99 1353.0         ", &  ! Kathmandu, Nepal
        ! Middle East & Africa
        "  55.30  25.20    5.0         ", &  ! Dubai, UAE
        "  46.68  24.15   48.0         ", &  ! Riyadh, Saudi Arabia
        "  31.25  30.04   20.0         ", &  ! Cairo, Egypt
        "  28.05 -26.20 1753.0         ", &  ! Johannesburg, South Africa
        "  37.67   -1.28 1661.0        ", &  ! Nairobi, Kenya
        ! Oceania
        " 151.21 -33.87   39.0         ", &  ! Sydney, Australia
        " 115.86 -31.95   17.0         ", &  ! Perth, Australia
        " 144.96 -37.81   58.0         ", &  ! Melbourne, Australia
        " 174.89 -41.29   12.0         ", &  ! Auckland, New Zealand
        ! Southern Africa
        "  18.42 -34.03   44.0         ", &  ! Cape Town, South Africa
        ! Additional strategic locations
        " -43.17 -22.91    2.0         ", &  ! Rio de Janeiro, Brazil
        " -51.52 -25.43  276.0         ", &  ! Curitiba, Brazil
        "  79.88  12.97    7.0         ", &  ! Bangalore, India
        "  77.21  28.61  216.0         ", &  ! New Delhi, India
        " -74.87   4.71 2640.0         ", &  ! Bogotá, Colombia
        " 106.85 -6.21    8.0          ", &  ! Jakarta, Indonesia
        " -71.54 -12.05  505.0         ", &  ! Lima, Peru
        " 135.52  34.69    3.0         ", &  ! Kobe, Japan
        " 139.77  35.48   40.0         ", &  ! Tokyo extended
        " -79.53  -0.22 2850.0         ", &  ! Quito, Ecuador
        " 151.20 -33.86  100.0         ", &  ! Sydney extended
        "  25.74  -1.95 1650.0         "  ]  ! Kampala, Uganda

    integer, parameter :: N_TIME = 48
    character(len=24), parameter :: TIME_POOL(N_TIME) = [ &
        ! January (day 15)
        " 0  6  15  1 2024", &   ! 6 AM
        " 0  9  15  1 2024", &   ! 9 AM
        " 0 12  15  1 2024", &   ! noon
        " 0 15  15  1 2024", &   ! 3 PM
        ! February (day 46)
        " 0  6  46  2 2024", &
        " 0  9  46  2 2024", &
        " 0 12  46  2 2024", &
        " 0 15  46  2 2024", &
        ! March equinox (day 79)
        " 0  6  79  3 2024", &
        " 0  9  79  3 2024", &
        " 0 12  79  3 2024", &
        " 0 15  79  3 2024", &
        ! April (day 105)
        " 0  6 105  4 2024", &
        " 0  9 105  4 2024", &
        " 0 12 105  4 2024", &
        " 0 15 105  4 2024", &
        ! May (day 135)
        " 0  6 135  5 2024", &
        " 0  9 135  5 2024", &
        " 0 12 135  5 2024", &
        " 0 15 135  5 2024", &
        ! June solstice (day 172)
        " 0  6 172  6 2024", &
        " 0  9 172  6 2024", &
        " 0 12 172  6 2024", &
        " 0 15 172  6 2024", &
        ! July (day 202)
        " 0  6 202  7 2024", &
        " 0  9 202  7 2024", &
        " 0 12 202  7 2024", &
        " 0 15 202  7 2024", &
        ! August (day 228)
        " 0  6 228  8 2024", &
        " 0  9 228  8 2024", &
        " 0 12 228  8 2024", &
        " 0 15 228  8 2024", &
        ! September equinox (day 266)
        " 0  6 266  9 2024", &
        " 0  9 266  9 2024", &
        " 0 12 266  9 2024", &
        " 0 15 266  9 2024", &
        ! October (day 299)
        " 0  6 299 10 2024", &
        " 0  9 299 10 2024", &
        " 0 12 299 10 2024", &
        " 0 15 299 10 2024", &
        ! November (day 325)
        " 0  6 325 11 2024", &
        " 0  9 325 11 2024", &
        " 0 12 325 11 2024", &
        " 0 15 325 11 2024", &
        ! December solstice (day 355)
        " 0  6 355 12 2024", &
        " 0  9 355 12 2024", &
        " 0 12 355 12 2024", &
        " 0 15 355 12 2024"  ]

    ! working variables
    integer :: i, funit
    real(c_double) :: rval, ic(2), result_mat(1,1)
    character(len=32) :: loc_str, time_str
    real(c_double) :: T_ss, eta_ss
    real(c_double), allocatable :: work_out(:,:)
    ! MC parameters
    integer, parameter :: N_MC_SPACIOUS    = 100
    integer, parameter :: N_MC_COMFORTABLE = 500
    integer, parameter :: N_MC_CRAMPED     = 1000
    real(c_double), parameter :: MC_SIGMA_FRAC = 0.05_c_double  ! 5% perturbation

    !=======================================================================
    ! BLOCK 3: lazy data — one CSV per independent variable
    !=======================================================================
    call execute_command_line("mkdir -p lazy", wait=.true.)

    call write_lazy_str ("lazy/location.csv",   LOC_POOL,  N_LOC,  N_LAZY)
    call write_lazy_str ("lazy/time.csv",        TIME_POOL, N_TIME, N_LAZY)
    call write_lazy_real("lazy/T_amb.csv",       BC_TAMB,   N_LAZY)
    call write_lazy_real("lazy/wind_speed.csv",  BC_WIND,   N_LAZY)
    call write_lazy_real("lazy/wind_dir.csv",    BC_WINDDIR,N_LAZY)
    call write_lazy_real("lazy/humidity.csv",    BC_HUMID,  N_LAZY)
    call write_lazy_real("lazy/irradiance.csv",  BC_IRR,    N_LAZY)
    call write_lazy_real("lazy/cloud_cover.csv", BC_CLOUD,  N_LAZY)
    call write_lazy_real("lazy/pressure.csv",    BC_PRESS,  N_LAZY)
    call write_lazy_real("lazy/pv_height.csv",   BC_PVH,    N_LAZY)
    call write_lazy_real("lazy/pitch.csv",       BC_PITCH,  N_LAZY)
    call write_lazy_real("lazy/roll.csv",        BC_ROLL,   N_LAZY)
    call write_lazy_real("lazy/yaw.csv",         BC_YAW,    N_LAZY)

    print *, "lazy: 13 files written to lazy/"

    !=======================================================================
    ! BLOCK 4: working data — 3 grid densities with Monte Carlo UQ
    !=======================================================================
    call generate_working_data("work/spacious.csv",    N_SPACIOUS,    1.0e-4_c_double, N_MC_SPACIOUS)
    call generate_working_data("work/comfortable.csv", N_COMFORTABLE, 1.0e-5_c_double, N_MC_COMFORTABLE)
    call generate_working_data("work/cramped.csv",     N_CRAMPED,     1.0e-6_c_double, N_MC_CRAMPED)

    print *, "working: 3 files written to work/ with MC UQ"

    contains

    ! Write N_LAZY uniformly spaced samples of a real variable to CSV
    subroutine write_lazy_real(filename, bc, n)
        character(len=*), intent(in) :: filename
        real(c_double),   intent(in) :: bc(2)
        integer,          intent(in) :: n
        integer :: u, i
        real(c_double) :: v

        open(newunit=u, file=filename, status='replace', action='write')
        write(u, '(A)') "value"
        do i = 1, n
            v = bc(1) + (bc(2) - bc(1)) * real(i-1, c_double) / real(n-1, c_double)
            write(u, '(F14.6)') v
        end do
        close(u)
    end subroutine write_lazy_real

    ! Write N samples cycling through a string pool to CSV
    subroutine write_lazy_str(filename, pool, pool_sz, n)
        character(len=*), intent(in) :: filename
        character(len=*), intent(in) :: pool(:)
        integer,          intent(in) :: pool_sz, n
        integer :: u, i

        open(newunit=u, file=filename, status='replace', action='write')
        write(u, '(A)') "value"
        do i = 1, n
            write(u, '(A)') trim(pool(mod(i-1, pool_sz) + 1))
        end do
        close(u)
    end subroutine write_lazy_str

    ! Parse a location string "lon lat alt" into env state
    subroutine load_location(loc_str)
        character(len=*), intent(in) :: loc_str
        read(loc_str, *) env_lon, env_lat, env_alt
    end subroutine load_location

    ! Parse a time string "min hour day month year" into env state
    subroutine load_time(time_str)
        character(len=*), intent(in) :: time_str
        read(time_str, *) env_min, env_hour, env_day, env_month, env_year
    end subroutine load_time

    ! Sample a real(c_double) uniformly in [lo, hi]
    function rand_in(lo, hi) result(v)
        real(c_double), intent(in) :: lo, hi
        real(c_double) :: v, r
        call random_number(r)
        v = lo + r * (hi - lo)
    end function rand_in

    ! Generate N working data samples at a given ODE tolerance with MC UQ, write to filename
    subroutine generate_working_data(filename, n, tol, n_mc)
        character(len=*), intent(in) :: filename
        integer,          intent(in) :: n, n_mc
        real(c_double),   intent(in) :: tol

        integer          :: u, i, loc_idx, time_idx
        real(c_double)   :: ic(2), ic_sigma(2)
        real(c_double)   :: r
      
        open(newunit=u, file=filename, status='replace', action='write')
        write(u, '(A)') "location,time,T_amb,wind_speed,wind_dir,humidity," // &
                         "irradiance,cloud_cover,pressure,pv_height,pitch," // &
                         "roll,yaw,T_operating,T_operating_sigma,eta,eta_sigma"

        call random_seed()

        do i = 1, n
            ! --- sample string variables ---
            call random_number(r)
            loc_idx  = int(r * N_LOC)  + 1
            call random_number(r)
            time_idx = int(r * N_TIME) + 1

            call load_location(LOC_POOL(loc_idx))
            call load_time(TIME_POOL(time_idx))

            ! --- sample real variables ---
            env_T_amb   = rand_in(BC_TAMB(1),    BC_TAMB(2))
            env_wind    = rand_in(BC_WIND(1),    BC_WIND(2))
            env_winddir = rand_in(BC_WINDDIR(1), BC_WINDDIR(2))
            env_humid   = rand_in(BC_HUMID(1),   BC_HUMID(2))
            env_irr     = rand_in(BC_IRR(1),     BC_IRR(2))
            env_cloud   = rand_in(BC_CLOUD(1),   BC_CLOUD(2))
            env_press   = rand_in(BC_PRESS(1),   BC_PRESS(2))
            env_pv_h    = rand_in(BC_PVH(1),     BC_PVH(2))
            env_pitch   = rand_in(BC_PITCH(1),   BC_PITCH(2))
            env_roll    = rand_in(BC_ROLL(1),    BC_ROLL(2))
            env_yaw     = rand_in(BC_YAW(1),     BC_YAW(2))

            ! --- derive G_eff from location + time + panel geometry ---
            call compute_G_eff()

            ! --- run RK45 to quasi-steady-state (t=0 to t=600s) ---
            ic(1) = env_T_amb   ! start at ambient
            ic(2) = ETA_REF * (1.0_c_double - BETA_T * (env_T_amb - T_STC))

            call RK45(pv_ode, ic, 0.0_c_double, 600.0_c_double, tol, ".rk45_tmp.csv")

            ! read back final state from tmp file
            call read_last_row(".rk45_tmp.csv", ic)

            ! --- compute MC uncertainty bounds at final state ---
            call mc_sigma_bounds(pv_ode, 600.0_c_double, ic, MC_SIGMA_FRAC, n_mc, ic, ic_sigma)

            ! --- write row with uncertainty columns ---
            write(u, '(A,",",A,11(",",F12.4),4(",",F12.6))') &
                trim(LOC_POOL(loc_idx)), trim(TIME_POOL(time_idx)), &
                env_T_amb, env_wind, env_winddir, env_humid, &
                env_irr, env_cloud, env_press, env_pv_h, &
                env_pitch, env_roll, env_yaw, &
                ic(1), ic_sigma(1), ic(2), ic_sigma(2)   ! T_operating, T_sigma, eta, eta_sigma
        end do

        close(u)
        call execute_command_line("rm -f .rk45_tmp.csv", wait=.true.)
        print '(A,A,A,I6,A)', " wrote ", trim(filename), " (", n, " samples)"
    end subroutine generate_working_data

    ! Read the last row of an RK45 output CSV and return the state vector
    subroutine read_last_row(filename, y_final)
        character(len=*), intent(in)  :: filename
        real(c_double),   intent(out) :: y_final(:)
        integer :: u, io
        real(c_double) :: t_tmp, y1_tmp, y2_tmp
        real(c_double) :: t_last, y1_last, y2_last

        open(newunit=u, file=filename, status='old', action='read')
        do
            read(u, *, iostat=io) t_tmp, y1_tmp, y2_tmp
            if (io /= 0) exit
            t_last  = t_tmp
            y1_last = y1_tmp
            y2_last = y2_tmp
        end do
        close(u)
        y_final(1) = y1_last
        y_final(2) = y2_last
    end subroutine read_last_row
end program AURA_MFP
