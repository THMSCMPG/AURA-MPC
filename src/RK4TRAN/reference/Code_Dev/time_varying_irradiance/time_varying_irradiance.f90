!=============================================================================
!  time_varying_irradiance.f90  --  Diurnal Irradiance and Temperature Modeling
!
!  Replaces the constant G = 1000 W/m^2 from euler_ode.f90 with a realistic
!  time-dependent function that approximates solar irradiance over a full day:
!
!    G(t) = G_peak * sin^2(pi * t / T_day)   for 0 <= t <= T_day  (daytime)
!    G(t) = 0                                 for T_day < t <= 86400 (night)
!
!  where:
!    G_peak = 1000 W/m^2  (peak irradiance at solar noon -- matches baseline)
!    T_day  = 43200 s     (12-hour photoperiod; t=0 is sunrise, t=T_day/2 is noon)
!    t is elapsed time from sunrise (the panel starts at T_amb at dawn)
!
!  This allows simulation of:
!    - The morning warm-up ramp
!    - Peak temperature around solar noon (lagged behind peak irradiance)
!    - Evening cool-down and overnight radiative cooling toward T_amb
!    - Daily energy yield integrated over the full diurnal cycle
!
!  Both linear and nonlinear ODEs are solved with RK4 (O(h^4)) since the
!  time-varying G(t) makes the linear ODE non-autonomous (no exact solution).
!
!  Daily energy yield (Wh) is accumulated using the rectangle rule:
!    E_day = sum( eta * G(t_i) * A * h )   over all i
!
!  Output file:
!    diurnal_output.dat  -- t(s) | G(t)(W/m^2) | T_nonlin(K) | T_lin(K) | P_elec(W)
!
!  Compile: gfortran -O2 -o time_varying time_varying_irradiance.f90
!=============================================================================
program time_varying_irradiance
    implicit none

    real(8), parameter :: x_start  = 0.0d0
    real(8), parameter :: x_end    = 86400.0d0   ! 24 hours = 1 full diurnal cycle
    real(8), parameter :: h        = 50.0d0      ! step size (s)
    real(8), parameter :: y0       = 298.15d0    ! T at sunrise = T_amb (K)

    ! Diurnal irradiance model parameters
    real(8), parameter :: G_peak   = 1000.0d0    ! W/m^2  -- peak at solar noon
    real(8), parameter :: T_day    = 43200.0d0   ! s      -- 12-hour daylight window
    real(8), parameter :: PI       = 3.141592653589793d0

    ! Physical parameters (identical to euler_ode.f90)
    real(8), parameter :: alpha  = 0.9d0
    real(8), parameter :: eta    = 0.18d0
    real(8), parameter :: A      = 1.6d0
    real(8), parameter :: h_conv = 15.0d0
    real(8), parameter :: m_pan  = 12.0d0
    real(8), parameter :: cp     = 900.0d0
    real(8), parameter :: T_amb  = 298.15d0
    real(8), parameter :: eps    = 0.85d0
    real(8), parameter :: sigma  = 5.670374419d-8
    real(8), parameter :: T_sky  = 278.15d0

    integer  :: n_steps, i
    real(8)  :: x, G_t
    real(8)  :: y_nlin, y_lin         ! temperature trajectories
    real(8)  :: kn1, kn2, kn3, kn4   ! RK4 stages for nonlinear
    real(8)  :: kl1, kl2, kl3, kl4   ! RK4 stages for linear
    real(8)  :: P_elec                ! electrical output [W]
    real(8)  :: E_total               ! daily energy yield [J]
    real(8)  :: T_max_nlin, T_max_lin ! peak panel temperature [K]
    real(8)  :: G_t2, yn_tmp, yl_tmp  ! intermediate temps for RK4 stages

    n_steps    = nint((x_end - x_start) / h)
    E_total    = 0.0d0
    T_max_nlin = y0
    T_max_lin  = y0

    open(unit=40, file='diurnal_output.dat', status='replace', action='write')
    write(40,'(A)') '# t(s)          G(t)(W/m2)     T_nonlin(K)' // &
                    '    T_lin(K)       P_elec(W)'

    x      = x_start
    y_nlin = y0
    y_lin  = y0

    do i = 0, n_steps

        ! G(t): sin^2 profile over daytime, zero at night
        if (x >= 0.0d0 .and. x <= T_day) then
            G_t = G_peak * (sin(PI * x / T_day))**2
        else
            G_t = 0.0d0
        end if

        P_elec  = eta * G_t * A
        E_total = E_total + P_elec * h   ! rectangle-rule energy accumulation

        if (y_nlin > T_max_nlin) T_max_nlin = y_nlin
        if (y_lin  > T_max_lin)  T_max_lin  = y_lin

        write(40,'(5F15.6)') x, G_t, y_nlin, y_lin, P_elec

        ! -------------------------------------------------------------------
        !  NONLINEAR ODE: RK4 with G(t) re-evaluated at each stage
        ! -------------------------------------------------------------------

        ! k1 at (x, y_nlin)
        G_t2 = G_t
        kn1 = (1.0d0 / (m_pan * cp)) * (                                    &
                (alpha - eta)*G_t2*A                                         &
              - h_conv*A*(y_nlin - T_amb)                                    &
              - eps*sigma*A*(y_nlin**4 - T_sky**4) )

        ! k2 at (x + h/2, y + h/2*k1)
        yn_tmp = y_nlin + (h / 2.0d0) * kn1
        if ((x + h/2.0d0) >= 0.0d0 .and. (x + h/2.0d0) <= T_day) then
            G_t2 = G_peak * (sin(PI * (x + h/2.0d0) / T_day))**2
        else
            G_t2 = 0.0d0
        end if
        kn2 = (1.0d0 / (m_pan * cp)) * (                                    &
                (alpha - eta)*G_t2*A                                         &
              - h_conv*A*(yn_tmp - T_amb)                                    &
              - eps*sigma*A*(yn_tmp**4 - T_sky**4) )

        ! k3 at (x + h/2, y + h/2*k2)
        yn_tmp = y_nlin + (h / 2.0d0) * kn2
        kn3 = (1.0d0 / (m_pan * cp)) * (                                    &
                (alpha - eta)*G_t2*A                                         &
              - h_conv*A*(yn_tmp - T_amb)                                    &
              - eps*sigma*A*(yn_tmp**4 - T_sky**4) )

        ! k4 at (x + h, y + h*k3)
        yn_tmp = y_nlin + h * kn3
        if ((x + h) >= 0.0d0 .and. (x + h) <= T_day) then
            G_t2 = G_peak * (sin(PI * (x + h) / T_day))**2
        else
            G_t2 = 0.0d0
        end if
        kn4 = (1.0d0 / (m_pan * cp)) * (                                    &
                (alpha - eta)*G_t2*A                                         &
              - h_conv*A*(yn_tmp - T_amb)                                    &
              - eps*sigma*A*(yn_tmp**4 - T_sky**4) )

        y_nlin = y_nlin + (h / 6.0d0) * (kn1 + 2.0d0*kn2 + 2.0d0*kn3 + kn4)

        ! -------------------------------------------------------------------
        !  LINEAR ODE: RK4 with G(t) re-evaluated at each stage
        !  (non-autonomous: G(t) makes the linear ODE non-separable too)
        ! -------------------------------------------------------------------

        ! k1 at (x, y_lin)
        G_t2 = G_t    ! already computed above for this x
        kl1 = (1.0d0 / (m_pan * cp)) * (                                    &
                (alpha - eta)*G_t2*A - h_conv*A*(y_lin - T_amb) )

        ! k2 at (x + h/2, y_lin + h/2*k1)
        yl_tmp = y_lin + (h / 2.0d0) * kl1
        if ((x + h/2.0d0) >= 0.0d0 .and. (x + h/2.0d0) <= T_day) then
            G_t2 = G_peak * (sin(PI * (x + h/2.0d0) / T_day))**2
        else
            G_t2 = 0.0d0
        end if
        kl2 = (1.0d0 / (m_pan * cp)) * (                                    &
                (alpha - eta)*G_t2*A - h_conv*A*(yl_tmp - T_amb) )

        ! k3 at (x + h/2, y_lin + h/2*k2)
        yl_tmp = y_lin + (h / 2.0d0) * kl2
        kl3 = (1.0d0 / (m_pan * cp)) * (                                    &
                (alpha - eta)*G_t2*A - h_conv*A*(yl_tmp - T_amb) )

        ! k4 at (x + h, y_lin + h*k3)
        yl_tmp = y_lin + h * kl3
        if ((x + h) >= 0.0d0 .and. (x + h) <= T_day) then
            G_t2 = G_peak * (sin(PI * (x + h) / T_day))**2
        else
            G_t2 = 0.0d0
        end if
        kl4 = (1.0d0 / (m_pan * cp)) * (                                    &
                (alpha - eta)*G_t2*A - h_conv*A*(yl_tmp - T_amb) )

        y_lin = y_lin + (h / 6.0d0) * (kl1 + 2.0d0*kl2 + 2.0d0*kl3 + kl4)

        x = x + h
    end do

    close(40)

    write(*,'(A)') 'Time-varying irradiance simulation complete (24-hour diurnal).'
    write(*,'(A,F8.3,A)') '  Peak T_nonlin               : ', T_max_nlin,    ' K'
    write(*,'(A,F8.3,A)') '  Peak T_lin                  : ', T_max_lin,     ' K'
    write(*,'(A,F8.3,A)') '  Final T_nonlin (midnight)   : ', y_nlin,        ' K'
    write(*,'(A,F8.3,A)') '  Final T_lin    (midnight)   : ', y_lin,         ' K'
    write(*,'(A,F9.2,A)') '  Daily energy yield (total)  : ', E_total / 3600.0d0, ' Wh'
    write(*,'(A,F9.2,A)') '  Daily energy yield (per m2) : ', &
                            E_total / (3600.0d0 * A), ' Wh/m2'
    write(*,'(A)') '  -> diurnal_output.dat'

end program time_varying_irradiance
