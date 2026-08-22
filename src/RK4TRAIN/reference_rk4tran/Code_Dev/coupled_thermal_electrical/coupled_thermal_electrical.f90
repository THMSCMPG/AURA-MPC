!=============================================================================
!  coupled_thermal_electrical.f90  --  Coupled Thermal-Electrical PV Model
!
!  Extends euler_ode.f90 by replacing the fixed efficiency eta = 0.18 with a
!  temperature-dependent efficiency eta(T). For crystalline silicon (c-Si):
!
!    eta(T) = eta_stc * [1 - beta_T * (T - T_stc)]
!
!  where:
!    eta_stc = 0.18       (18% efficiency at Standard Test Conditions)
!    beta_T  = 0.0045 /K  (0.45%/K temperature coefficient for c-Si)
!    T_stc   = 298.15 K   (STC reference temperature = 25 deg C)
!
!  Physical feedback loop:
!    Rising T  -->  lower eta(T)  -->  less electrical power extracted
!    -->  more net heat absorbed  -->  T rises further to a new equilibrium
!
!  The modified nonlinear ODE RHS is:
!    dT/dt = (1/mcp) * [(alpha - eta(T))*G*A
!                      - h_conv*A*(T - T_amb)
!                      - eps*sigma*A*(T^4 - T_sky^4)]
!
!  Note: eta(T) appears inside the nonlinear RHS, so the ODE is no longer
!  separable and has no closed-form solution. RK4 is used for both models.
!
!  Two trajectories are compared side-by-side:
!    T_const : constant eta = eta_stc (matches euler_ode.f90 nonlinear)
!    T_coupled: temperature-dependent eta(T)
!
!  Electrical power output is also tracked:
!    P(t) = eta(T) * G * A   [W]
!
!  Output file:
!    coupled_te_output.dat  -- t | T_const | T_coupled | eta(T) | P_const | P_coupled
!
!  Compile: gfortran -O2 -o coupled_te coupled_thermal_electrical.f90
!=============================================================================
program coupled_thermal_electrical
    implicit none

    real(8), parameter :: x_start = 0.0d0
    real(8), parameter :: x_end   = 3600.0d0    ! 1 hour
    real(8), parameter :: h       = 100.0d0      ! step size (s) -- matches baseline
    real(8), parameter :: y0      = 298.15d0    ! T_0 = T_amb (K)

    ! Electrical efficiency model (Skoplaki & Palyvos, Solar Energy 2009)
    real(8), parameter :: eta_stc  = 0.18d0      ! efficiency at STC
    real(8), parameter :: beta_T   = 0.0045d0    ! temperature coefficient /K
    real(8), parameter :: T_stc    = 298.15d0    ! STC reference temperature (K)

    ! Thermal / optical parameters (identical to euler_ode.f90)
    real(8), parameter :: alpha  = 0.9d0
    real(8), parameter :: G      = 1000.0d0     ! W/m^2
    real(8), parameter :: A      = 1.6d0        ! m^2
    real(8), parameter :: h_conv = 15.0d0       ! W/m^2/K
    real(8), parameter :: m_pan  = 12.0d0       ! kg
    real(8), parameter :: cp     = 900.0d0      ! J/kg/K
    real(8), parameter :: T_amb  = 298.15d0     ! K
    real(8), parameter :: eps    = 0.85d0
    real(8), parameter :: sigma  = 5.670374419d-8   ! W/m^2/K^4  (CODATA 2018)
    real(8), parameter :: T_sky  = 278.15d0

    integer  :: n_steps, i
    real(8)  :: x
    real(8)  :: y_const, y_coupled     ! temperature: constant-eta vs. coupled-eta
    real(8)  :: eta_val                ! eta(T) evaluated at current T_coupled
    real(8)  :: P_const, P_coupled    ! electrical power output [W]
    real(8)  :: k1, k2, k3, k4        ! RK4 stage slopes
    real(8)  :: yt, eta_t             ! temporary values for RK4 stages

    n_steps = nint((x_end - x_start) / h)

    open(unit=30, file='coupled_te_output.dat', status='replace', action='write')
    write(30,'(A)') '# t(s)          T_const(K)     T_coupled(K)' // &
                    '   eta(T)         P_const(W)     P_coupled(W)'

    x        = x_start
    y_const  = y0
    y_coupled = y0

    do i = 0, n_steps

        ! eta(T) for the coupled model at current temperature
        eta_val = eta_stc * (1.0d0 - beta_T * (y_coupled - T_stc))
        eta_val = max(0.0d0, min(1.0d0, eta_val))   ! physical clamp to [0, 1]

        ! Instantaneous electrical power output [W]
        P_const   = eta_stc  * G * A
        P_coupled = eta_val  * G * A

        write(30,'(6F15.6)') x, y_const, y_coupled, eta_val, P_const, P_coupled

        ! -------------------------------------------------------------------
        !  CONSTANT-ETA MODEL: Improved Euler (matches nonlinear in euler_ode.f90)
        !  Uses eta = eta_stc fixed throughout
        ! -------------------------------------------------------------------
        k1 = (1.0d0 / (m_pan * cp)) * (                                     &
               (alpha - eta_stc) * G * A                                     &
             - h_conv * A * (y_const - T_amb)                                &
             - eps * sigma * A * (y_const**4 - T_sky**4) )

        yt = y_const + h * k1
        k2 = (1.0d0 / (m_pan * cp)) * (                                     &
               (alpha - eta_stc) * G * A                                     &
             - h_conv * A * (yt - T_amb)                                     &
             - eps * sigma * A * (yt**4 - T_sky**4) )

        y_const = y_const + (h / 2.0d0) * (k1 + k2)

        ! -------------------------------------------------------------------
        !  COUPLED-ETA MODEL: RK4 (eta re-evaluated at each stage)
        !  eta(T) = eta_stc * [1 - beta_T * (T - T_stc)]
        ! -------------------------------------------------------------------

        ! Stage k1: eta at current y_coupled
        eta_t = max(0.0d0, min(1.0d0, eta_stc * (1.0d0 - beta_T * (y_coupled - T_stc))))
        k1 = (1.0d0 / (m_pan * cp)) * (                                     &
               (alpha - eta_t) * G * A                                       &
             - h_conv * A * (y_coupled - T_amb)                              &
             - eps * sigma * A * (y_coupled**4 - T_sky**4) )

        ! Stage k2: eta at y + h/2 * k1
        yt    = y_coupled + (h / 2.0d0) * k1
        eta_t = max(0.0d0, min(1.0d0, eta_stc * (1.0d0 - beta_T * (yt - T_stc))))
        k2 = (1.0d0 / (m_pan * cp)) * (                                     &
               (alpha - eta_t) * G * A                                       &
             - h_conv * A * (yt - T_amb)                                     &
             - eps * sigma * A * (yt**4 - T_sky**4) )

        ! Stage k3: eta at y + h/2 * k2
        yt    = y_coupled + (h / 2.0d0) * k2
        eta_t = max(0.0d0, min(1.0d0, eta_stc * (1.0d0 - beta_T * (yt - T_stc))))
        k3 = (1.0d0 / (m_pan * cp)) * (                                     &
               (alpha - eta_t) * G * A                                       &
             - h_conv * A * (yt - T_amb)                                     &
             - eps * sigma * A * (yt**4 - T_sky**4) )

        ! Stage k4: eta at y + h * k3
        yt    = y_coupled + h * k3
        eta_t = max(0.0d0, min(1.0d0, eta_stc * (1.0d0 - beta_T * (yt - T_stc))))
        k4 = (1.0d0 / (m_pan * cp)) * (                                     &
               (alpha - eta_t) * G * A                                       &
             - h_conv * A * (yt - T_amb)                                     &
             - eps * sigma * A * (yt**4 - T_sky**4) )

        y_coupled = y_coupled + (h / 6.0d0) * (k1 + 2.0d0*k2 + 2.0d0*k3 + k4)

        x = x + h
    end do

    close(30)

    ! Final efficiency and power comparison
    eta_val = eta_stc * (1.0d0 - beta_T * (y_coupled - T_stc))
    eta_val = max(0.0d0, min(1.0d0, eta_val))

    write(*,'(A)') 'Coupled Thermal-Electrical model complete.'
    write(*,'(A,F8.3,A)') '  T_const   (t=1hr, fixed eta)    = ', y_const,   ' K'
    write(*,'(A,F8.3,A)') '  T_coupled (t=1hr, eta(T))       = ', y_coupled, ' K'
    write(*,'(A,F8.3,A)') '  Delta T due to thermal feedback = ', &
                            y_coupled - y_const, ' K'
    write(*,'(A,F6.4)')   '  eta(T_coupled, t=1hr)           = ', eta_val
    write(*,'(A,F6.2,A)') '  Efficiency loss vs STC          = ', &
                            (eta_stc - eta_val) / eta_stc * 100.0d0, ' %'
    write(*,'(A,F7.2,A)') '  P_coupled (t=1hr)               = ', eta_val * G * A, ' W'
    write(*,'(A)') '  -> coupled_te_output.dat'

end program coupled_thermal_electrical
