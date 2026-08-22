!=============================================================================
!  atmospheric_properties.f90  --  Atmospheric Effects on PV Thermal Model
!
!  Extends euler_ode.f90 by computing physically-motivated values for G and
!  h_conv instead of using the fixed G=1000 W/m^2, h_conv=15 W/m^2K baseline.
!
!  Two atmospheric sub-models are coupled to the nonlinear ODE:
!
!  1. Beer-Lambert Attenuation  (AURA-MFP: BTE free-path / Beer-Lambert direct)
!     Models how aerosols, Rayleigh scattering, and water vapour attenuate
!     incoming solar radiation before it reaches the panel surface:
!
!       G_eff(AOD, theta_z) = G_toa * exp(-(tau_ray + tau_wv + AOD) / cos(theta_z))
!
!     where:
!       G_toa   = 1361.0 W/m^2  (AURA-MFP SOLAR_CONSTANT; top-of-atmosphere)
!       tau_ray = 0.09           (Rayleigh scattering optical depth)
!       tau_wv  = 0.03           (water vapour absorption)
!       AOD     = aerosol optical depth (scenario-dependent; see below)
!       theta_z = solar zenith angle [radians]
!
!  2. McAdams Forced-Convection Correlation  (AURA-MFP: lofi_solver_module)
!     Replaces the fixed h_conv = 15 W/m^2K with the wind-speed-dependent
!     McAdams correlation widely used in PV thermal modelling:
!
!       h_conv(WS) = 5.7 + 3.8 * WS   [W/m^2 K]
!
!     where WS is the wind speed in m/s.
!
!  Five scenarios are simulated simultaneously for direct comparison:
!    (a) Baseline   : G=1000 W/m^2, h_conv=15 W/m^2K  (matches euler_ode.f90)
!    (b) Clear+Calm : AOD=0.05, WS=1.0 m/s
!    (c) Clear+Wind : AOD=0.05, WS=8.0 m/s
!    (d) Hazy+Calm  : AOD=0.40, WS=1.0 m/s  (urban/industrial pollution)
!    (e) Hazy+Wind  : AOD=0.40, WS=8.0 m/s
!
!  Solar zenith angle is fixed at theta_z=30 deg (mid-morning / afternoon).
!  All scenarios use RK4 integration.
!
!  Output file:
!    atm_output.dat  -- t | T_base | T_clear_calm | T_clear_wind |
!                       T_hazy_calm | T_hazy_wind | G_clear | G_hazy
!
!  Compile: gfortran -O2 -o atmospheric_properties atmospheric_properties.f90
!=============================================================================
program atmospheric_properties
    implicit none

    real(8), parameter :: x_start = 0.0d0
    real(8), parameter :: x_end   = 3600.0d0    ! 1 hour
    real(8), parameter :: h       = 100.0d0      ! step size (s) -- matches baseline
    real(8), parameter :: y0      = 298.15d0    ! T_0 = T_amb (K)

    ! Top-of-atmosphere irradiance (AURA-MFP constants_module: SOLAR_CONSTANT)
    real(8), parameter :: G_toa   = 1361.0d0    ! W/m^2

    ! Atmospheric optical depths (1-layer column model)
    real(8), parameter :: tau_ray  = 0.09d0     ! Rayleigh scattering
    real(8), parameter :: tau_wv   = 0.03d0     ! water vapour absorption
    real(8), parameter :: AOD_clear = 0.05d0    ! clean marine/rural atmosphere
    real(8), parameter :: AOD_hazy  = 0.40d0    ! polluted/urban atmosphere

    ! Solar geometry
    real(8), parameter :: theta_z  = 0.5235987756d0  ! rad = 30 deg zenith angle
    real(8), parameter :: PI       = 3.141592653589793d0

    ! Wind speed scenarios
    real(8), parameter :: WS_calm  = 1.0d0   ! m/s
    real(8), parameter :: WS_windy = 8.0d0   ! m/s

    ! Fixed physical parameters (identical to euler_ode.f90)
    real(8), parameter :: alpha  = 0.9d0
    real(8), parameter :: eta    = 0.18d0
    real(8), parameter :: A      = 1.6d0
    real(8), parameter :: m_pan  = 12.0d0
    real(8), parameter :: cp     = 900.0d0
    real(8), parameter :: T_amb  = 298.15d0
    real(8), parameter :: eps    = 0.85d0
    real(8), parameter :: sigma  = 5.670374419d-8   ! W/m^2/K^4  (CODATA 2018)
    real(8), parameter :: T_sky  = 278.15d0

    ! Baseline (no atmospheric correction -- reproduces euler_ode.f90 nonlinear)
    real(8), parameter :: G_base     = 1000.0d0
    real(8), parameter :: h_conv_base = 15.0d0

    integer  :: n_steps, i
    real(8)  :: x
    real(8)  :: y_base, y_cc, y_cw, y_hc, y_hw   ! temperature per scenario
    real(8)  :: G_clear, G_hazy                   ! effective irradiances [W/m^2]
    real(8)  :: h_calm, h_windy                   ! McAdams h_conv [W/m^2 K]
    real(8)  :: k1, k2, k3, k4, ytmp, G_s, hc_s  ! RK4 working vars

    real(8), external :: nonlin_rhs

    n_steps = nint((x_end - x_start) / h)

    ! --- Pre-compute scenario-fixed quantities ---

    ! Beer-Lambert: G_eff = G_toa * exp(-tau_total / cos(theta_z))
    G_clear = G_toa * exp(-(tau_ray + tau_wv + AOD_clear) / cos(theta_z))
    G_hazy  = G_toa * exp(-(tau_ray + tau_wv + AOD_hazy)  / cos(theta_z))

    ! McAdams correlation: h = 5.7 + 3.8 * WS  (AURA-MFP lofi_solver_module)
    h_calm  = 5.7d0 + 3.8d0 * WS_calm
    h_windy = 5.7d0 + 3.8d0 * WS_windy

    open(unit=50, file='atm_output.dat', status='replace', action='write')
    write(50,'(A)') '# t(s)       T_base(K)    T_clr_clm    T_clr_wnd' // &
                    '    T_hzy_clm    T_hzy_wnd    G_eff_clr    G_eff_hzy'

    x     = x_start
    y_base = y0
    y_cc   = y0   ! clear + calm
    y_cw   = y0   ! clear + windy
    y_hc   = y0   ! hazy  + calm
    y_hw   = y0   ! hazy  + windy

    do i = 0, n_steps

        write(50,'(7F15.6)') x, y_base, y_cc, y_cw, y_hc, y_hw, G_clear, G_hazy

        ! -------------------------------------------------------------------
        !  Macro: RK4 step for the nonlinear ODE
        !  Given scenario (G_s, hc_s) and current y, compute the new y.
        !  The macro is inlined four times (one per scenario) to avoid
        !  external-function argument lists while keeping code readable.
        ! -------------------------------------------------------------------

        ! (a) BASELINE  G=1000, h_conv=15
        G_s = G_base ; hc_s = h_conv_base
        k1 = nonlin_rhs(y_base,        G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_base + (h/2.0d0)*k1
        k2 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_base + (h/2.0d0)*k2
        k3 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_base + h*k3
        k4 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        y_base = y_base + (h/6.0d0)*(k1 + 2.0d0*k2 + 2.0d0*k3 + k4)

        ! (b) CLEAR + CALM  G=G_clear, h_conv=h_calm
        G_s = G_clear ; hc_s = h_calm
        k1 = nonlin_rhs(y_cc,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_cc + (h/2.0d0)*k1
        k2 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_cc + (h/2.0d0)*k2
        k3 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_cc + h*k3
        k4 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        y_cc = y_cc + (h/6.0d0)*(k1 + 2.0d0*k2 + 2.0d0*k3 + k4)

        ! (c) CLEAR + WINDY  G=G_clear, h_conv=h_windy
        G_s = G_clear ; hc_s = h_windy
        k1 = nonlin_rhs(y_cw,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_cw + (h/2.0d0)*k1
        k2 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_cw + (h/2.0d0)*k2
        k3 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_cw + h*k3
        k4 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        y_cw = y_cw + (h/6.0d0)*(k1 + 2.0d0*k2 + 2.0d0*k3 + k4)

        ! (d) HAZY + CALM  G=G_hazy, h_conv=h_calm
        G_s = G_hazy ; hc_s = h_calm
        k1 = nonlin_rhs(y_hc,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_hc + (h/2.0d0)*k1
        k2 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_hc + (h/2.0d0)*k2
        k3 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_hc + h*k3
        k4 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        y_hc = y_hc + (h/6.0d0)*(k1 + 2.0d0*k2 + 2.0d0*k3 + k4)

        ! (e) HAZY + WINDY  G=G_hazy, h_conv=h_windy
        G_s = G_hazy ; hc_s = h_windy
        k1 = nonlin_rhs(y_hw,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_hw + (h/2.0d0)*k1
        k2 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_hw + (h/2.0d0)*k2
        k3 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        ytmp = y_hw + h*k3
        k4 = nonlin_rhs(ytmp,          G_s, hc_s, alpha, eta, A, m_pan, cp, T_amb, eps, sigma, T_sky)
        y_hw = y_hw + (h/6.0d0)*(k1 + 2.0d0*k2 + 2.0d0*k3 + k4)

        x = x + h
    end do

    close(50)

    write(*,'(A)') 'Atmospheric properties simulation complete.'
    write(*,'(A)') '  Effective irradiances (theta_z = 30 deg):'
    write(*,'(A,F8.2,A)') '    G_clear  (AOD=0.05) : ', G_clear, ' W/m^2'
    write(*,'(A,F8.2,A)') '    G_hazy   (AOD=0.40) : ', G_hazy,  ' W/m^2'
    write(*,'(A)') '  McAdams h_conv:'
    write(*,'(A,F6.2,A)') '    WS=1 m/s (calm)  : ', h_calm,  ' W/m^2 K'
    write(*,'(A,F6.2,A)') '    WS=8 m/s (windy) : ', h_windy, ' W/m^2 K'
    write(*,'(A)') '  T(t=1hr) per scenario:'
    write(*,'(A,F8.3,A)') '    (a) Baseline         : ', y_base, ' K'
    write(*,'(A,F8.3,A)') '    (b) Clear + calm     : ', y_cc,   ' K'
    write(*,'(A,F8.3,A)') '    (c) Clear + windy    : ', y_cw,   ' K'
    write(*,'(A,F8.3,A)') '    (d) Hazy  + calm     : ', y_hc,   ' K'
    write(*,'(A,F8.3,A)') '    (e) Hazy  + windy    : ', y_hw,   ' K'
    write(*,'(A)') '  -> atm_output.dat'

end program atmospheric_properties


!  nonlin_rhs(T, G, h_conv, ...)  --  generic nonlinear ODE RHS
!  dT/dt = (1/mcp) * [(alpha-eta)*G*A - h_conv*A*(T-T_amb) - eps*sigma*A*(T^4-T_sky^4)]
!  G and h_conv are passed in explicitly, allowing scenario-by-scenario variation.
real(8) function nonlin_rhs(T, G, h_conv, alpha, eta, A, m, cp, &
                             T_amb, eps, sigma, T_sky)
    real(8), intent(in) :: T, G, h_conv, alpha, eta, A, m, cp
    real(8), intent(in) :: T_amb, eps, sigma, T_sky

    nonlin_rhs = (1.0d0 / (m * cp)) * (  (alpha - eta)*G*A             &
                                        - h_conv*A*(T - T_amb)          &
                                        - eps*sigma*A*(T**4 - T_sky**4) )

end function nonlin_rhs
