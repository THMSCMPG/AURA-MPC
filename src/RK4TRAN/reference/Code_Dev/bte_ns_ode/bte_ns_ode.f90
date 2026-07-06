!=============================================================================
!  bte_ns_ode.f90  --  Tightly-Coupled BTE–NS Solar Panel ODE System
!
!  Synthesises every sub-model from the First-Order ODE Solar project into a
!  single, fully-coupled two-state ODE system solved with RK4.
!
!  ---- CONTEXT IN THE PROJECT SEQUENCE ----------------------------------------
!
!  Module             G(t)          h_conv             eta
!  -----------------  ------------  -----------------  -----------------
!  euler_ode          const 1000    const 15           const 0.18
!  rk4_ode            const 1000    const 15           const 0.18
!  atmospheric_prop   Beer-Lambert  McAdams(WS fixed)  const 0.18
!  time_varying_irr   sin^2 G(t)    const 15           const 0.18
!  coupled_thermal    const 1000    const 15           eta(T)
!  monte_carlo_uq     stochastic    stochastic         const 0.18
!  bte_ns_ode (HERE)  BTE diurnal   NS mixed+BL ODE    eta(T)   <-- all coupled
!
!  ---- PHYSICAL MODEL ----------------------------------------------------------
!
!  State vector  :  y = [T(t), WS(t)]
!    T   = solar panel temperature (K)
!    WS  = wind speed at panel surface (m/s)  [new dynamic state]
!
!  --- BTE Component (Boltzmann / Beer-Lambert photon transport) ---
!
!  Diurnal effective irradiance with full atmospheric attenuation:
!
!    cos_z(t) = sin(pi * t / T_day)          solar elevation over photoperiod
!                                             (t=0: sunrise, t=T_day/2: noon)
!
!    G_eff(t) = G_toa * cos_z(t) *           Beer-Lambert + cosine projection
!               exp(-tau_tot / cos_z(t))     (W/m^2); zero at night
!
!    tau_tot  = tau_ray + tau_wv + AOD       total one-way optical depth
!             = 0.09   + 0.03   + 0.20 = 0.32
!
!  Stefan-Boltzmann longwave emission (BTE thermal radiation closure):
!    Q_rad = eps * sigma * A * (T^4 - T_sky^4)
!
!  Temperature-dependent efficiency (BTE-thermal-electrical coupling):
!    eta(T) = eta_stc * [1 - beta_T * (T - T_stc)]    clamped to [0, 1]
!
!  --- NS Component (Navier-Stokes convective energy transport) ---
!
!  McAdams forced convection  (AURA-MFP lofi_solver_module):
!    h_forced = 5.7 + 3.8 * WS              (W/m^2 K)
!
!  Turbulent free (natural) convection  [Churchill flat-plate correlation]:
!    h_nat    = C_nc * |T - T_amb|^(1/3)    C_nc = 1.31 W/(m^2 K^{4/3})
!
!  Mixed-convection combination rule  (Churchill & Usagi 1972):
!    h_eff    = (h_forced^3 + h_nat^3)^(1/3)    (W/m^2 K)
!
!  NS surface-layer boundary-layer ODE for wind speed:
!    dWS/dt = -lambda*(WS - WS_geo) + gamma*(T - T_amb)
!
!    lambda = 1/tau_BL = 1/600 s^-1   boundary-layer relaxation
!    WS_geo = 3.0 m/s                  geostrophic (background) wind
!    gamma  = 3.0e-3 m/s/K             thermal buoyancy coupling coefficient
!
!  ---- COUPLED ODE SYSTEM -------------------------------------------------------
!
!    dT/dt  = (1/mcp)*[(alpha - eta(T))*G_eff(t)*A          <- BTE absorption
!                     - h_eff(T,WS)*A*(T - T_amb)           <- NS convection
!                     - eps*sigma*A*(T^4 - T_sky^4)]        <- BTE emission
!
!    dWS/dt = -lambda*(WS - WS_geo) + gamma*(T - T_amb)     <- NS BL dynamics
!
!  The tight coupling is enforced by evaluating both G_eff(t), eta(T), h_eff,
!  and dWS/dt simultaneously at every RK4 stage k1..k4.
!
!  ---- DECOUPLED BASELINE (for comparison) --------------------------------------
!
!  Single-state dT/dt with no dynamic feedbacks:
!    G = 1000 W/m^2 (constant),  h_conv = 15.0 W/m^2 K (constant)
!    eta = eta_stc = 0.18        (constant)
!    WS  = not solved (fixed)
!
!  This reproduces the nonlinear ODE from rk4_ode.f90 exactly.
!
!  ---- SIMULATION PARAMETERS ----------------------------------------------------
!    t in [0, 86400 s]   (24-hour diurnal cycle; t=0 is sunrise)
!    h = 50 s            (RK4 step size; matches all previous modules)
!    T(0)  = T_amb = 298.15 K    panel at ambient temperature at sunrise
!    WS(0) = 1.5 m/s             calm morning wind condition
!
!  ---- OUTPUT FILE --------------------------------------------------------------
!    bte_ns_output.dat  (9 columns):
!      Col 1 : t          (s)
!      Col 2 : T_dec      (K)       decoupled baseline: const G, h_conv, eta
!      Col 3 : T_coup     (K)       tightly coupled BTE–NS model
!      Col 4 : WS         (m/s)     NS BL dynamic wind speed
!      Col 5 : G_eff      (W/m^2)   BTE Beer-Lambert diurnal irradiance
!      Col 6 : h_eff      (W/m^2 K) NS mixed-convection coefficient
!      Col 7 : eta_T      (-)       temperature-dependent efficiency
!      Col 8 : P_dec      (W)       electrical power, decoupled model
!      Col 9 : P_coup     (W)       electrical power, BTE–NS coupled model
!
!  Compile: gfortran -O2 -o bte_ns_ode bte_ns_ode.f90
!  Run:     ./bte_ns_ode
!  Plot:    gnuplot plot_bte_ns.gp
!=============================================================================
program bte_ns_ode
    implicit none

    !--------------------------------------------------------------------------
    !  Integration parameters
    !--------------------------------------------------------------------------
    real(8), parameter :: x_start  = 0.0d0          ! sunrise (s)
    real(8), parameter :: x_end    = 86400.0d0       ! 24-hour diurnal cycle (s)
    real(8), parameter :: h_step   = 100.0d0          ! RK4 step size (s)

    !--------------------------------------------------------------------------
    !  Initial conditions
    !--------------------------------------------------------------------------
    real(8), parameter :: T_0      = 298.15d0        ! panel temperature at t=0 (K)
    real(8), parameter :: WS_0     = 1.5d0           ! wind speed at t=0 (m/s)

    !--------------------------------------------------------------------------
    !  BTE parameters
    !--------------------------------------------------------------------------
    real(8), parameter :: G_toa    = 1361.0d0        ! top-of-atmosphere irradiance (W/m^2)
                                                      ! AURA-MFP: SOLAR_CONSTANT
    real(8), parameter :: tau_ray  = 0.09d0           ! Rayleigh scattering optical depth
    real(8), parameter :: tau_wv   = 0.03d0           ! water vapour absorption
    real(8), parameter :: AOD      = 0.20d0           ! aerosol optical depth (moderate)
    real(8), parameter :: tau_tot  = tau_ray + tau_wv + AOD   ! = 0.32 total
    real(8), parameter :: T_day    = 43200.0d0        ! photoperiod = 12 hours (s)
    real(8), parameter :: PI       = 3.141592653589793d0

    !--------------------------------------------------------------------------
    !  Thermal / optical parameters (from rk4_ode.f90 for direct comparability)
    !--------------------------------------------------------------------------
    real(8), parameter :: alpha    = 0.9d0            ! panel absorptivity
    real(8), parameter :: A        = 1.6d0            ! panel area (m^2)
    real(8), parameter :: m_pan    = 12.0d0           ! panel mass (kg)
    real(8), parameter :: cp       = 900.0d0          ! specific heat (J/kg/K)
    real(8), parameter :: T_amb    = 298.15d0         ! ambient temperature (K)
    real(8), parameter :: eps      = 0.85d0           ! surface emissivity
    real(8), parameter :: sigma    = 5.670374419d-8   ! W/m^2/K^4 (CODATA 2018)
    real(8), parameter :: T_sky    = 278.15d0         ! effective sky temperature (K)

    !--------------------------------------------------------------------------
    !  Thermal-electrical coupling  (from coupled_thermal_electrical.f90)
    !--------------------------------------------------------------------------
    real(8), parameter :: eta_stc  = 0.18d0           ! STC conversion efficiency
    real(8), parameter :: beta_T   = 0.0045d0         ! temperature coefficient /K  (c-Si)
    real(8), parameter :: T_stc    = 298.15d0         ! STC reference temperature (K)

    !--------------------------------------------------------------------------
    !  NS convection parameters  (from atmospheric_properties.f90 + Churchill)
    !--------------------------------------------------------------------------
    !  McAdams:      h_forced = 5.7 + 3.8 * WS          [W/m^2 K]
    !  Free convect: h_nat    = C_nc * |T - T_amb|^(1/3) [W/m^2 K]
    !    C_nc = 1.31 W/(m^2 K^{4/3}) for turbulent free convection on
    !    a flat horizontal plate in air (Churchill & Chu 1975 correlation)
    !  Mixed:        h_eff    = (h_forced^3 + h_nat^3)^(1/3)
    real(8), parameter :: C_nc     = 1.31d0            ! W/(m^2 K^{4/3})

    !--------------------------------------------------------------------------
    !  NS boundary-layer ODE parameters  (new in this module)
    !  dWS/dt = -lambda*(WS - WS_geo) + gamma*(T - T_amb)
    !--------------------------------------------------------------------------
    real(8), parameter :: WS_geo   = 3.0d0             ! geostrophic wind speed (m/s)
    real(8), parameter :: tau_BL   = 600.0d0           ! BL relaxation time scale (s)
    real(8), parameter :: lambda   = 1.0d0 / tau_BL    ! BL relaxation rate (s^-1)
    real(8), parameter :: gamma_ws = 3.0d-3             ! buoyancy coupling (m/s/K)

    !--------------------------------------------------------------------------
    !  Decoupled baseline parameters  (reproduces rk4_ode.f90 nonlinear ODE)
    !--------------------------------------------------------------------------
    real(8), parameter :: G_base   = 1000.0d0           ! constant irradiance (W/m^2)
    real(8), parameter :: h_base   = 15.0d0             ! constant h_conv (W/m^2 K)

    !--------------------------------------------------------------------------
    !  Working variables
    !--------------------------------------------------------------------------
    integer  :: n_steps, i
    real(8)  :: t                      ! current time (s)
    real(8)  :: T_dec                  ! decoupled baseline temperature (K)
    real(8)  :: y_c(2)                 ! coupled state:  y_c(1)=T, y_c(2)=WS
    real(8)  :: k1(2), k2(2), k3(2), k4(2)   ! RK4 stage vectors for coupled
    real(8)  :: dk1, dk2, dk3, dk4    ! RK4 stages for decoupled T
    real(8)  :: G_now                  ! G_eff at current t  (diagnostic)
    real(8)  :: h_now                  ! h_eff at current state  (diagnostic)
    real(8)  :: eta_now                ! eta(T) at current T_coup  (diagnostic)
    real(8)  :: P_dec, P_coup          ! electrical power outputs (W)
    real(8)  :: T_max_coup, T_max_dec  ! peak temperatures (K)
    real(8)  :: E_coup, E_dec          ! daily energy yield accumulators (J)

    n_steps    = nint((x_end - x_start) / h_step)
    T_max_coup = T_0
    T_max_dec  = T_0
    E_coup     = 0.0d0
    E_dec      = 0.0d0

    open(unit=70, file='bte_ns_output.dat', status='replace', action='write')
    write(70,'(A)') '# t(s)         T_dec(K)       T_coup(K)' // &
                    '      WS(m/s)        G_eff(W/m2)    h_eff(W/m2K)' // &
                    '   eta_T          P_dec(W)       P_coup(W)'

    !--------------------------------------------------------------------------
    !  Initialise states
    !--------------------------------------------------------------------------
    t      = x_start
    T_dec  = T_0
    y_c(1) = T_0        ! T_coupled
    y_c(2) = WS_0       ! WS initial condition

    !--------------------------------------------------------------------------
    !  Main time-stepping loop
    !--------------------------------------------------------------------------
    do i = 0, n_steps

        !----------------------------------------------------------------------
        !  Compute diagnostic quantities at current (t, state) for output
        !----------------------------------------------------------------------
        G_now   = g_eff(t)
        h_now   = h_eff(y_c(1), y_c(2))
        eta_now = eta_T(y_c(1))

        ! Electrical power: P = eta * G * A
        P_dec  = eta_stc  * G_base * A      ! decoupled: constant G, const eta
        P_coup = eta_now  * G_now  * A      ! coupled:   BTE G(t), eta(T)

        ! Accumulate daily energy yield (rectangle rule)
        E_coup = E_coup + P_coup * h_step
        E_dec  = E_dec  + P_dec  * h_step

        ! Track peak temperatures
        if (y_c(1) > T_max_coup) T_max_coup = y_c(1)
        if (T_dec  > T_max_dec)  T_max_dec  = T_dec

        write(70,'(9F15.6)') t,        T_dec,  y_c(1), y_c(2), &
                              G_now,    h_now,  eta_now,         &
                              P_dec,    P_coup

        !----------------------------------------------------------------------
        !  RK4 for COUPLED system:  d/dt[T, WS] = F_coupled(t, T, WS)
        !
        !  Evaluates G_eff(t+h/2), eta(T+dk/2), h_eff(T+dk/2, WS+dk/2)
        !  simultaneously at each stage -- this is the tight coupling.
        !----------------------------------------------------------------------
        call rhs_coupled(t,             y_c,                          k1)
        call rhs_coupled(t + h_step/2,  y_c + (h_step/2.0d0)*k1,     k2)
        call rhs_coupled(t + h_step/2,  y_c + (h_step/2.0d0)*k2,     k3)
        call rhs_coupled(t + h_step,    y_c + h_step*k3,              k4)
        y_c = y_c + (h_step/6.0d0)*(k1 + 2.0d0*k2 + 2.0d0*k3 + k4)

        !----------------------------------------------------------------------
        !  RK4 for DECOUPLED baseline:  dT/dt = f_decoupled(T)
        !  Const G=1000, h_conv=15, eta=0.18  (reproduces rk4_ode nonlinear)
        !----------------------------------------------------------------------
        dk1 = f_decoupled(T_dec)
        dk2 = f_decoupled(T_dec + (h_step/2.0d0)*dk1)
        dk3 = f_decoupled(T_dec + (h_step/2.0d0)*dk2)
        dk4 = f_decoupled(T_dec +  h_step*dk3)
        T_dec = T_dec + (h_step/6.0d0)*(dk1 + 2.0d0*dk2 + 2.0d0*dk3 + dk4)

        t = t + h_step

    end do

    close(70)

    !--------------------------------------------------------------------------
    !  Final diagnostic summary
    !--------------------------------------------------------------------------
    write(*,'(A)') '============================================================'
    write(*,'(A)') '  bte_ns_ode.f90  --  BTE-NS Tight Coupling  (24-hour run)'
    write(*,'(A)') '============================================================'
    write(*,'(A)') '  Physics coupled at every RK4 stage:'
    write(*,'(A)') '    BTE : Beer-Lambert G_eff(t) + S-B emission + eta(T)'
    write(*,'(A)') '    NS  : McAdams forced + Churchill free convection'
    write(*,'(A)') '          + boundary-layer dWS/dt ODE'
    write(*,'(A)') '------------------------------------------------------------'
    write(*,'(A,F8.3,A)') '  T_decoupled (t=24h)          : ', T_dec,       ' K'
    write(*,'(A,F8.3,A)') '  T_coupled   (t=24h)          : ', y_c(1),      ' K'
    write(*,'(A,F6.3,A)') '  WS          (t=24h)          : ', y_c(2),      ' m/s'
    write(*,'(A,F8.3,A)') '  Peak T_decoupled             : ', T_max_dec,   ' K'
    write(*,'(A,F8.3,A)') '  Peak T_coupled               : ', T_max_coup,  ' K'
    write(*,'(A,F8.4)')   '  eta(T_coupled, t=24h)        : ', eta_T(y_c(1))
    write(*,'(A,F9.2,A)') '  Daily energy yield (dec.)    : ', E_dec  / 3600.0d0, ' Wh'
    write(*,'(A,F9.2,A)') '  Daily energy yield (coup.)   : ', E_coup / 3600.0d0, ' Wh'
    write(*,'(A,I0,A)')   '  Steps written                : ', n_steps + 1, ' rows'
    write(*,'(A)') '------------------------------------------------------------'
    write(*,'(A)') '  -> bte_ns_output.dat'
    write(*,'(A)') '  -> gnuplot plot_bte_ns.gp  to visualise'
    write(*,'(A)') '============================================================'

contains

    !==========================================================================
    !  rhs_coupled  --  RHS of the tightly-coupled [T, WS] ODE system
    !
    !  All three BTE–NS coupling feedbacks are evaluated on every call:
    !    (1) G_eff(t_in)          BTE Beer-Lambert diurnal irradiance
    !    (2) eta_T(y(1))          BTE thermal-electrical efficiency feedback
    !    (3) h_eff(y(1), y(2))    NS mixed forced + natural convection
    !
    !  This simultaneous evaluation inside RK4 constitutes the tight coupling:
    !  each sub-step sees self-consistent values of G, eta, and h at the
    !  intermediate state (t + h/2, y + dk/2), not the values from t.
    !==========================================================================
    subroutine rhs_coupled(t_in, y, dydt)
        real(8), intent(in)  :: t_in
        real(8), intent(in)  :: y(2)
        real(8), intent(out) :: dydt(2)

        real(8) :: T_loc, WS_loc
        real(8) :: G_loc, eta_loc, heff_loc

        T_loc   = y(1)
        WS_loc  = y(2)

        ! --- BTE: irradiance and efficiency at this stage ---
        G_loc   = g_eff(t_in)
        eta_loc = eta_T(T_loc)

        ! --- NS: mixed-convection coefficient at this stage ---
        heff_loc = h_eff(T_loc, WS_loc)

        ! --- dT/dt : thermal energy equation (BTE absorbed + NS convected + BTE emitted) ---
        dydt(1) = (1.0d0 / (m_pan * cp)) * (                                  &
                   (alpha - eta_loc) * G_loc * A                               &  ! BTE: SW absorbed
                 - heff_loc * A * (T_loc - T_amb)                              &  ! NS:  convective loss
                 - eps * sigma * A * (T_loc**4 - T_sky**4) )                      ! BTE: LW emitted

        ! --- dWS/dt : NS surface-layer boundary-layer ODE ---
        !  First term:  exponential relaxation toward geostrophic wind  (BL spin-down)
        !  Second term: thermal buoyancy enhancement from warm panel    (local forcing)
        dydt(2) = -lambda * (WS_loc - WS_geo) + gamma_ws * (T_loc - T_amb)

    end subroutine rhs_coupled

    !==========================================================================
    !  f_decoupled  --  RHS of the single-state decoupled baseline
    !
    !  Reproduces the nonlinear ODE from rk4_ode.f90 exactly:
    !    G = 1000 W/m^2 (constant),  h_conv = 15 W/m^2 K,  eta = eta_stc = 0.18
    !  No BTE diurnal variation, no NS wind dynamics, no eta(T) feedback.
    !==========================================================================
    real(8) function f_decoupled(T_in)
        real(8), intent(in) :: T_in

        f_decoupled = (1.0d0 / (m_pan * cp)) * (                              &
                       (alpha - eta_stc) * G_base * A                         &  ! const SW absorbed
                     - h_base * A * (T_in - T_amb)                            &  ! const convective loss
                     - eps * sigma * A * (T_in**4 - T_sky**4) )                  ! LW emitted

    end function f_decoupled

    !==========================================================================
    !  g_eff  --  BTE Beer-Lambert diurnal effective irradiance  [W/m^2]
    !
    !  Solar elevation geometry:
    !    cos_z(t) = sin(pi * t / T_day)
    !    Maps sunrise (t=0) through solar noon (t=T_day/2) to sunset (t=T_day).
    !    cos_z is the cosine of the solar zenith angle AND the air-mass factor.
    !
    !  Beer-Lambert attenuation through atmosphere:
    !    G_eff = G_toa * cos_z * exp(-tau_tot / cos_z)
    !    The cos_z prefactor is the geometric projection onto the panel plane.
    !    The exp term is the Beer-Lambert transmittance along the slant path.
    !
    !  Source: atmospheric_properties.f90 (Beer-Lambert, theta_z model)
    !          time_varying_irradiance.f90 (diurnal G(t) framework)
    !==========================================================================
    real(8) function g_eff(t_in)
        real(8), intent(in) :: t_in
        real(8) :: cos_z

        if (t_in >= 0.0d0 .and. t_in <= T_day) then
            cos_z = sin(PI * t_in / T_day)     ! solar elevation (= cos of zenith)
            if (cos_z > 1.0d-6) then
                g_eff = G_toa * cos_z * exp(-tau_tot / cos_z)
            else
                g_eff = 0.0d0
            end if
        else
            g_eff = 0.0d0                       ! night: no solar irradiance
        end if

    end function g_eff

    !==========================================================================
    !  h_eff  --  NS mixed-convection coefficient  [W/m^2 K]
    !
    !  Combines forced and natural convection using the Churchill-Usagi
    !  blending rule with exponent n=3 (standard for mixed-convection flat
    !  plates when neither regime dominates):
    !
    !    h_eff = (h_forced^3 + h_nat^3)^(1/3)
    !
    !  h_forced = 5.7 + 3.8*WS  McAdams (1954) correlation
    !             from AURA-MFP lofi_solver_module (atmospheric_properties.f90)
    !
    !  h_nat = C_nc * |T-T_amb|^(1/3)
    !          Turbulent free-convection flat-plate correlation
    !          (Churchill & Chu 1975; C_nc = 1.31 W/(m^2 K^{4/3}) for air)
    !
    !  Note: WS is clamped to >= 0 to prevent unphysical negative values
    !  during RK4 sub-steps when WS is near zero.
    !==========================================================================
    real(8) function h_eff(T_in, WS_in)
        real(8), intent(in) :: T_in, WS_in
        real(8) :: h_frc, h_nat_val, WS_safe

        WS_safe   = max(0.0d0, WS_in)
        h_frc     = 5.7d0 + 3.8d0 * WS_safe
        h_nat_val = C_nc * abs(T_in - T_amb)**(1.0d0/3.0d0)

        h_eff     = (h_frc**3 + h_nat_val**3)**(1.0d0/3.0d0)

    end function h_eff

    !==========================================================================
    !  eta_T  --  temperature-dependent conversion efficiency  [-]
    !
    !  Skoplaki-Palyvos linear model for crystalline silicon (c-Si):
    !    eta(T) = eta_stc * [1 - beta_T * (T - T_stc)]
    !
    !  Physically clamped to [0, 1].  At STC (T = 298.15 K), eta = eta_stc.
    !  For every 1 K above STC the efficiency decreases by beta_T = 0.45 %.
    !
    !  Source: coupled_thermal_electrical.f90 (Skoplaki & Palyvos, SE 2009)
    !==========================================================================
    real(8) function eta_T(T_in)
        real(8), intent(in) :: T_in
        eta_T = max(0.0d0, min(1.0d0, eta_stc * (1.0d0 - beta_T * (T_in - T_stc))))
    end function eta_T

end program bte_ns_ode
