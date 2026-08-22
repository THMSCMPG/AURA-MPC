program euler_ode
    implicit none

    ! Integration parameters
    real(8), parameter :: x_start = 0.0d0      ! t = 0 s
    real(8), parameter :: x_end   = 3600.0d0   ! t = 1 hour
    real(8), parameter :: h       = 50.0d0     ! 10-second time steps
    real(8), parameter :: y0      = 298.15d0   ! initial temp = ambient (K)

    integer  :: n_steps, i
    real(8)  :: x
    real(8)  :: y_lin_e,  y_lin_ie           ! linear:    Euler, Improved Euler
    real(8)  :: y_nlin_e, y_nlin_ie          ! nonlinear: Euler, Improved Euler
    real(8)  :: y_exact_val
    real(8)  :: k1, k2                       ! Heun's method slopes

    real(8), external :: f, w, exact_solution

    n_steps = nint((x_end - x_start) / h)

    !  LINEAR ODE
    !  Columns: t | y_euler | y_imp_euler | y_exact | err_euler | err_imp_euler
    open(unit=10, file="linear_output.dat", status='replace', action='write')
    write(10,'(A)') "# t              y_euler        y_imp_euler    y_exact" // &
                    "        err_euler      err_imp_euler"

    x        = x_start
    y_lin_e  = y0
    y_lin_ie = y0

    do i = 0, n_steps
        y_exact_val = exact_solution(x, y0, x_start)

        write(10,'(6F15.6)') x, y_lin_e, y_lin_ie, y_exact_val, &
                              abs(y_lin_e  - y_exact_val),        &
                              abs(y_lin_ie - y_exact_val)

        ! Forward Euler (linear)
        y_lin_e = y_lin_e + h * f(x, y_lin_e)

        ! Improved Euler / Heun (linear)
        !    k1 = slope at (x, y)
        !    k2 = slope at predictor (x+h, y + h*k1)
        !    y_new = y + h/2 * (k1 + k2)
        k1 = f(x,     y_lin_ie)
        k2 = f(x + h, y_lin_ie + h * k1)
        y_lin_ie = y_lin_ie + (h / 2.0d0) * (k1 + k2)

        x = x + h
    end do

    close(10)

    !  NONLINEAR ODE  (convection + radiation — no exact solution)
    !  Columns: t | y_euler | y_imp_euler
    open(unit=11, file="nonlinear_output.dat", status='replace', action='write')
    write(11,'(A)') "# t              y_euler        y_imp_euler"

    x         = x_start
    y_nlin_e  = y0
    y_nlin_ie = y0

    do i = 0, n_steps
        write(11,'(3F15.6)') x, y_nlin_e, y_nlin_ie

        ! Forward Euler (nonlinear)
        y_nlin_e = y_nlin_e + h * w(x, y_nlin_e)

        ! Improved Euler / Heun (nonlinear)
        k1 = w(x,     y_nlin_ie)
        k2 = w(x + h, y_nlin_ie + h * k1)
        y_nlin_ie = y_nlin_ie + (h / 2.0d0) * (k1 + k2)

        x = x + h
    end do

    close(11)

    write(*,'(A,I0,A)') "Done. Wrote ", n_steps + 1, " points."
    write(*,'(A)')       "  -> linear_output.dat    (Euler | Improved Euler | Exact)"
    write(*,'(A)')       "  -> nonlinear_output.dat (Euler | Improved Euler)"
    write(*,'(A,F6.2)')  "  Step size h = ", h
    write(*,'(A,F6.1,A,F8.1,A)') "  Interval  [", x_start, ", ", x_end, "]"
end program euler_ode


!  f(x,y)  —  RHS of the LINEAR ODE (convection only)
!  dT/dt = (1 / m*cp) * [ (alpha - eta)*G*A  -  h_conv*A*(T - T_amb) ]
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


!  g(x,y)  —  RHS of the NONLINEAR ODE (convection + radiation)
!  dT/dt = (1 / m*cp) * [ (alpha-eta)*G*A
!                        - h_conv*A*(T - T_amb)
!                        - eps*sigma*A*(T^4 - T_sky^4) ]
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
    real(8), parameter :: sigma  = 5.67d-8    ! W/m^2/K^4
    real(8), parameter :: T_sky  = 278.15d0   ! K

    w = (1.0d0 / (m * cp)) * (  (alpha - eta)*G*A             &
                               - h_conv*A*(y - T_amb)          &
                               - eps*sigma*A*(y**4 - T_sky**4) )

end function w


!  exact_solution(t, T0, b0)  —  closed-form for the LINEAR ODE
!  T(t) = T_ss + (T0 - T_ss) * exp(-(t - t0) / tau)
!    T_ss = T_amb + (alpha - eta)*G / h_conv
!    tau  = m*cp / (h_conv * A)
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

    T_ss = T_amb + (alpha - eta)*G / h_conv   ! steady-state temperature (K)
    tau  = (m * cp) / (h_conv * A)            ! thermal time constant (s)

    exact_solution = T_ss + (T0 - T_ss) * exp(-(t - b0) / tau)

end function exact_solution
