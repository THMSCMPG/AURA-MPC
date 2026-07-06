!=============================================================================
!  rk4_ode.f90  --  Classical 4th-Order Runge-Kutta for Solar Panel ODE
!
!  Extends euler_ode.f90 by adding RK4 as a third integration method alongside
!  Forward Euler and Improved Euler (Heun). Because the nonlinear ODE has no
!  closed-form solution, RK4 (O(h^4) global error) serves as a near-exact
!  benchmark for that case, completing the three-way accuracy comparison
!  described in the Future Research section.
!
!  Physics (unchanged from euler_ode.f90):
!    Linear ODE:    dT/dt = (1/mcp) * [(alpha-eta)*G*A - h_conv*A*(T - T_amb)]
!    Nonlinear ODE: adds Stefan-Boltzmann term: -eps*sigma*A*(T^4 - T_sky^4)
!
!  RK4 formula (O(h^4) global truncation error):
!    k1 = f(t,       y)
!    k2 = f(t + h/2, y + h/2 * k1)
!    k3 = f(t + h/2, y + h/2 * k2)
!    k4 = f(t + h,   y + h   * k3)
!    y_new = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
!
!  Output files:
!    rk4_linear_output.dat    -- t | euler | imp_euler | rk4 | exact |
!                                err_euler | err_imp_euler | err_rk4
!    rk4_nonlinear_output.dat -- t | euler | imp_euler | rk4
!
!  Compile: gfortran -O2 -o rk4_ode rk4_ode.f90
!=============================================================================
program rk4_ode
    implicit none

    ! Integration parameters (identical to euler_ode.f90 for direct comparison)
    real(8), parameter :: x_start = 0.0d0       ! t = 0 s
    real(8), parameter :: x_end   = 3600.0d0    ! t = 1 hour
    real(8), parameter :: h       = 100.0d0      ! step size (s)
    real(8), parameter :: y0      = 298.15d0    ! T_0 = T_amb (K)

    integer  :: n_steps, i
    real(8)  :: x
    real(8)  :: y_lin_e,    y_lin_ie,    y_lin_rk4    ! linear:    3 methods
    real(8)  :: y_nlin_e,   y_nlin_ie,   y_nlin_rk4   ! nonlinear: 3 methods
    real(8)  :: y_exact_val
    real(8)  :: k1, k2, k3, k4                        ! RK4 stage slopes

    real(8), external :: f, w, exact_solution

    n_steps = nint((x_end - x_start) / h)

    ! -----------------------------------------------------------------------
    !  LINEAR ODE
    !  Columns: t | euler | imp_euler | rk4 | exact | err_e | err_ie | err_rk4
    ! -----------------------------------------------------------------------
    open(unit=10, file='rk4_linear_output.dat', status='replace', action='write')
    write(10,'(A)') '# t              y_euler        y_imp_euler' // &
                    '    y_rk4          y_exact        err_euler' // &
                    '      err_imp_euler  err_rk4'

    x          = x_start
    y_lin_e    = y0
    y_lin_ie   = y0
    y_lin_rk4  = y0

    do i = 0, n_steps
        y_exact_val = exact_solution(x, y0, x_start)

        write(10,'(8F15.6)') x,                             &
                              y_lin_e,                       &
                              y_lin_ie,                      &
                              y_lin_rk4,                     &
                              y_exact_val,                   &
                              abs(y_lin_e    - y_exact_val), &
                              abs(y_lin_ie   - y_exact_val), &
                              abs(y_lin_rk4  - y_exact_val)

        ! Forward Euler  (O(h^1) -- 1 function evaluation)
        y_lin_e = y_lin_e + h * f(x, y_lin_e)

        ! Improved Euler / Heun  (O(h^2) -- 2 function evaluations)
        k1 = f(x,     y_lin_ie)
        k2 = f(x + h, y_lin_ie + h * k1)
        y_lin_ie = y_lin_ie + (h / 2.0d0) * (k1 + k2)

        ! RK4  (O(h^4) -- 4 function evaluations)
        k1 = f(x,           y_lin_rk4)
        k2 = f(x + h/2.0d0, y_lin_rk4 + (h / 2.0d0) * k1)
        k3 = f(x + h/2.0d0, y_lin_rk4 + (h / 2.0d0) * k2)
        k4 = f(x + h,       y_lin_rk4 + h * k3)
        y_lin_rk4 = y_lin_rk4 + (h / 6.0d0) * (k1 + 2.0d0*k2 + 2.0d0*k3 + k4)

        x = x + h
    end do

    close(10)

    ! -----------------------------------------------------------------------
    !  NONLINEAR ODE  (convection + radiation -- no exact solution)
    !  Columns: t | euler | imp_euler | rk4
    ! -----------------------------------------------------------------------
    open(unit=11, file='rk4_nonlinear_output.dat', status='replace', action='write')
    write(11,'(A)') '# t              y_euler        y_imp_euler    y_rk4'

    x           = x_start
    y_nlin_e    = y0
    y_nlin_ie   = y0
    y_nlin_rk4  = y0

    do i = 0, n_steps
        write(11,'(4F15.6)') x, y_nlin_e, y_nlin_ie, y_nlin_rk4

        ! Forward Euler
        y_nlin_e = y_nlin_e + h * w(x, y_nlin_e)

        ! Improved Euler / Heun
        k1 = w(x,     y_nlin_ie)
        k2 = w(x + h, y_nlin_ie + h * k1)
        y_nlin_ie = y_nlin_ie + (h / 2.0d0) * (k1 + k2)

        ! RK4
        k1 = w(x,           y_nlin_rk4)
        k2 = w(x + h/2.0d0, y_nlin_rk4 + (h / 2.0d0) * k1)
        k3 = w(x + h/2.0d0, y_nlin_rk4 + (h / 2.0d0) * k2)
        k4 = w(x + h,       y_nlin_rk4 + h * k3)
        y_nlin_rk4 = y_nlin_rk4 + (h / 6.0d0) * (k1 + 2.0d0*k2 + 2.0d0*k3 + k4)

        x = x + h
    end do

    close(11)

    write(*,'(A,I0,A)') 'Done. Wrote ', n_steps + 1, ' points per method.'
    write(*,'(A)')       '  -> rk4_linear_output.dat    (Euler | Imp.Euler | RK4 | Exact)'
    write(*,'(A)')       '  -> rk4_nonlinear_output.dat (Euler | Imp.Euler | RK4)'
    write(*,'(A,F6.2)')  '  Step size h = ', h
    write(*,'(A,F6.1,A,F8.1,A)') '  Interval  [', x_start, ', ', x_end, '] s'

end program rk4_ode


!  f(x,y)  --  RHS of the LINEAR ODE (convection only)
!  dT/dt = (1/mcp) * [(alpha - eta)*G*A  -  h_conv*A*(T - T_amb)]
!  Identical to euler_ode.f90 -- included here for self-contained compilation.
real(8) function f(x, y)
    real(8), intent(in) :: x, y
    real(8), parameter :: alpha  = 0.9d0
    real(8), parameter :: eta    = 0.18d0
    real(8), parameter :: G      = 1000.0d0   ! W/m^2
    real(8), parameter :: A      = 1.6d0      ! m^2
    real(8), parameter :: h_conv = 15.0d0     ! W/m^2/K  (NOT the step size h)
    real(8), parameter :: m      = 12.0d0     ! kg
    real(8), parameter :: cp     = 900.0d0    ! J/kg/K
    real(8), parameter :: T_amb  = 298.15d0   ! K

    f = (1.0d0 / (m * cp)) * ( (alpha - eta)*G*A  -  h_conv*A*(y - T_amb) )

end function f


!  w(x,y)  --  RHS of the NONLINEAR ODE (convection + Stefan-Boltzmann radiation)
!  dT/dt = (1/mcp) * [(alpha-eta)*G*A
!                    - h_conv*A*(T - T_amb)
!                    - eps*sigma*A*(T^4 - T_sky^4)]
!  Uses AURA-MFP SIGMA_SB = 5.670374419e-8 (more accurate than original 5.67e-8)
real(8) function w(x, y)
    real(8), intent(in) :: x, y
    real(8), parameter :: alpha  = 0.9d0
    real(8), parameter :: eta    = 0.18d0
    real(8), parameter :: G      = 1000.0d0
    real(8), parameter :: A      = 1.6d0
    real(8), parameter :: h_conv = 15.0d0
    real(8), parameter :: m      = 12.0d0
    real(8), parameter :: cp     = 900.0d0
    real(8), parameter :: T_amb  = 298.15d0
    real(8), parameter :: eps    = 0.85d0
    real(8), parameter :: sigma  = 5.670374419d-8   ! W/m^2/K^4  (CODATA 2018)
    real(8), parameter :: T_sky  = 278.15d0          ! K

    w = (1.0d0 / (m * cp)) * (  (alpha - eta)*G*A             &
                               - h_conv*A*(y - T_amb)          &
                               - eps*sigma*A*(y**4 - T_sky**4) )

end function w


!  exact_solution(t, T0, b0)  --  closed-form for the LINEAR ODE
!  T(t) = T_ss + (T0 - T_ss) * exp(-(t - t0) / tau)
!    T_ss = T_amb + (alpha - eta)*G / h_conv    (steady-state temperature)
!    tau  = m*cp / (h_conv * A)                 (thermal time constant)
real(8) function exact_solution(t, T0, b0)
    real(8), intent(in) :: t, T0, b0
    real(8), parameter :: alpha  = 0.9d0
    real(8), parameter :: eta    = 0.18d0
    real(8), parameter :: G      = 1000.0d0
    real(8), parameter :: A      = 1.6d0
    real(8), parameter :: h_conv = 15.0d0
    real(8), parameter :: m      = 12.0d0
    real(8), parameter :: cp     = 900.0d0
    real(8), parameter :: T_amb  = 298.15d0
    real(8) :: T_ss, tau

    T_ss = T_amb + (alpha - eta)*G / h_conv
    tau  = (m * cp) / (h_conv * A)

    exact_solution = T_ss + (T0 - T_ss) * exp(-(t - b0) / tau)

end function exact_solution
