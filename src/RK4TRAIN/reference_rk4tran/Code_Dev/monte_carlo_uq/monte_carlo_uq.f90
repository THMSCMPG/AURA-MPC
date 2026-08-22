!=============================================================================
!  monte_carlo_uq.f90  --  Monte Carlo Uncertainty Quantification
!
!  Propagates real-world measurement uncertainty through the nonlinear PV
!  thermal ODE using Improved Euler (Heun), building probability distributions
!  for the predicted temperature trajectory across N_MC independent runs.
!
!  Each run draws a fresh set of physical parameters from Gaussian distributions
!  whose widths reflect documented real-world variability:
!
!    h_conv : mean = 15.0 W/m^2K,  std = 4.5    (+-30%  wind-speed variation)
!    G      : mean = 1000 W/m^2,   std = 100.0  (+-10%  cloud cover / AOD)
!    eps    : mean = 0.85,          std = 0.0425 (+-5%   surface degradation)
!    T_amb  : mean = 298.15 K,      std = 2.0 K  (+-2K   weather variation)
!
!  Gaussian sampling uses the Box-Muller transform (no external library).
!  Physical bounds are enforced after sampling (h_conv > 1, G >= 0, etc.)
!
!  The nonlinear ODE RHS is:
!    dT/dt = (1/mcp) * [(alpha-eta)*G*A - h_conv*A*(T-T_amb) - eps*sigma*A*(T^4-T_sky^4)]
!
!  Output files:
!    mc_trajectories.dat  -- t | T_mean | T_std | T_min | T_max
!    mc_parameters.dat    -- run | h_conv | G | eps | T_amb  (first 100 runs)
!
!  Compile: gfortran -O2 -o monte_carlo_uq monte_carlo_uq.f90
!=============================================================================
program monte_carlo_uq
    implicit none

    integer,  parameter :: N_MC    = 1000         ! number of Monte Carlo runs
    real(8),  parameter :: x_start = 0.0d0
    real(8),  parameter :: x_end   = 3600.0d0     ! 1 hour
    real(8),  parameter :: h       = 100.0d0       ! step size (s) -- matches baseline
    real(8),  parameter :: y0      = 298.15d0     ! T_0 = T_amb (K)

    ! Parameter distribution means (match euler_ode.f90 nominal values)
    real(8),  parameter :: h_conv_mean = 15.0d0
    real(8),  parameter :: G_mean      = 1000.0d0
    real(8),  parameter :: eps_mean    = 0.85d0
    real(8),  parameter :: T_amb_mean  = 298.15d0

    ! Standard deviations (physical measurement uncertainty)
    real(8),  parameter :: h_conv_std  = 4.5d0    ! +- 30%
    real(8),  parameter :: G_std       = 100.0d0  ! +- 10%
    real(8),  parameter :: eps_std     = 0.0425d0 ! +- 5%
    real(8),  parameter :: T_amb_std   = 2.0d0    ! +- 2 K

    ! Fixed parameters (no uncertainty applied)
    real(8),  parameter :: alpha  = 0.9d0
    real(8),  parameter :: eta    = 0.18d0
    real(8),  parameter :: A      = 1.6d0
    real(8),  parameter :: m_pan  = 12.0d0
    real(8),  parameter :: cp     = 900.0d0
    real(8),  parameter :: sigma  = 5.670374419d-8  ! W/m^2/K^4  (CODATA 2018)
    real(8),  parameter :: T_sky  = 278.15d0

    ! Box-Muller constant
    real(8),  parameter :: TWO_PI = 6.283185307179586d0

    integer  :: n_steps, i, mc
    real(8)  :: x, y, k1, k2, dT
    real(8)  :: h_conv_s, G_s, eps_s, T_amb_s   ! per-run sampled parameters
    real(8)  :: r1, r2, z0, z1                   ! Box-Muller working vars
    real(8)  :: T_val, T_mean, T_std

    ! Ensemble statistics  (indexed 0:n_steps for consistency with loop)
    real(8), allocatable :: T_sum(:), T_sum2(:), T_min(:), T_max(:)
    real(8), allocatable :: params(:,:)           ! [N_MC x 4] sampled values
    integer :: alloc_stat

    n_steps = nint((x_end - x_start) / h)

    allocate(T_sum(0:n_steps),  STAT=alloc_stat)
    allocate(T_sum2(0:n_steps), STAT=alloc_stat)
    allocate(T_min(0:n_steps),  STAT=alloc_stat)
    allocate(T_max(0:n_steps),  STAT=alloc_stat)
    allocate(params(N_MC, 4),   STAT=alloc_stat)
    if (alloc_stat /= 0) then
        write(*,'(A)') 'ERROR: allocation failed. Stopping.'
        stop 1
    end if

    T_sum  = 0.0d0
    T_sum2 = 0.0d0
    T_min  =  1.0d15
    T_max  = -1.0d15

    call random_seed()

    ! -----------------------------------------------------------------------
    !  Monte Carlo ensemble loop
    ! -----------------------------------------------------------------------
    do mc = 1, N_MC

        ! --- Sample parameters via Box-Muller transform ---
        call random_number(r1)
        call random_number(r2)
        r1 = max(r1, 2.22d-16)   ! guard against log(0)
        r2 = max(r2, 2.22d-16)
        z0 = sqrt(-2.0d0 * log(r1)) * cos(TWO_PI * r2)
        z1 = sqrt(-2.0d0 * log(r1)) * sin(TWO_PI * r2)
        h_conv_s = max(1.0d0, h_conv_mean + h_conv_std * z0)   ! physical lower bound
        G_s      = max(0.0d0, G_mean      + G_std      * z1)

        call random_number(r1)
        call random_number(r2)
        r1 = max(r1, 2.22d-16)
        r2 = max(r2, 2.22d-16)
        z0 = sqrt(-2.0d0 * log(r1)) * cos(TWO_PI * r2)
        z1 = sqrt(-2.0d0 * log(r1)) * sin(TWO_PI * r2)
        eps_s   = max(0.0d0, min(1.0d0, eps_mean   + eps_std   * z0))
        T_amb_s = T_amb_mean + T_amb_std * z1

        params(mc, 1) = h_conv_s
        params(mc, 2) = G_s
        params(mc, 3) = eps_s
        params(mc, 4) = T_amb_s

        ! --- Integrate using Improved Euler (best method from baseline study) ---
        x = x_start
        y = y0

        do i = 0, n_steps

            ! Accumulate statistics at the start of each step (before advancing)
            T_val = y
            T_sum(i)  = T_sum(i)  + T_val
            T_sum2(i) = T_sum2(i) + T_val * T_val
            if (T_val < T_min(i)) T_min(i) = T_val
            if (T_val > T_max(i)) T_max(i) = T_val

            ! Improved Euler (Heun) step with sampled parameters
            !   k1 = f(t,   y)
            !   k2 = f(t+h, y + h*k1)
            !   y_new = y + (h/2)*(k1 + k2)

            k1 = (1.0d0 / (m_pan * cp)) * (                                 &
                   (alpha - eta) * G_s * A                                   &
                 - h_conv_s * A * (y - T_amb_s)                              &
                 - eps_s * sigma * A * (y**4 - T_sky**4) )

            dT = y + h * k1
            k2 = (1.0d0 / (m_pan * cp)) * (                                 &
                   (alpha - eta) * G_s * A                                   &
                 - h_conv_s * A * (dT - T_amb_s)                             &
                 - eps_s * sigma * A * (dT**4 - T_sky**4) )

            y = y + (h / 2.0d0) * (k1 + k2)
            x = x + h

        end do

    end do  ! mc loop

    ! -----------------------------------------------------------------------
    !  Write ensemble statistics
    ! -----------------------------------------------------------------------
    open(unit=20, file='mc_trajectories.dat', status='replace', action='write')
    write(20,'(A)') '# t(s)          T_mean(K)      T_std(K)' // &
                    '       T_min(K)       T_max(K)'

    x = x_start
    do i = 0, n_steps
        T_mean = T_sum(i) / real(N_MC, 8)
        T_std  = sqrt(max(0.0d0, T_sum2(i) / real(N_MC, 8) - T_mean**2))
        write(20,'(5F15.6)') x, T_mean, T_std, T_min(i), T_max(i)
        x = x + h
    end do
    close(20)

    ! Write sampled parameters (first 100 runs for diagnostic / gnuplot use)
    open(unit=21, file='mc_parameters.dat', status='replace', action='write')
    write(21,'(A)') '# run     h_conv(W/m2K)   G(W/m2)        eps' // &
                    '            T_amb(K)'
    do mc = 1, min(N_MC, 100)
        write(21,'(I6,4F15.6)') mc, params(mc,1), params(mc,2), &
                                     params(mc,3), params(mc,4)
    end do
    close(21)

    ! --- Summary to stdout ---
    T_mean = T_sum(n_steps) / real(N_MC, 8)
    T_std  = sqrt(max(0.0d0, T_sum2(n_steps) / real(N_MC, 8) - T_mean**2))

    write(*,'(A,I0,A)') 'Monte Carlo UQ complete.  N_MC = ', N_MC, ' runs.'
    write(*,'(A,F8.3,A,F6.3,A)') '  T(t=1hr): mean = ', T_mean, ' K,  1-sigma = ', T_std, ' K'
    write(*,'(A,F8.3,A,F8.3,A)') '  T(t=1hr): range [', T_min(n_steps), &
                                  ',  ', T_max(n_steps), '] K'
    write(*,'(A,F6.2,A)') '  95% CI half-width: +/-', 1.96d0 * T_std, ' K'
    write(*,'(A)') '  -> mc_trajectories.dat  (mean | 1-sigma | min | max at each t)'
    write(*,'(A)') '  -> mc_parameters.dat    (first 100 sampled parameter sets)'

    deallocate(T_sum, T_sum2, T_min, T_max, params)

end program monte_carlo_uq
