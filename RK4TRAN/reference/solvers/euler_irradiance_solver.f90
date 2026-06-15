! euler_irradiance_solver.f90
! Forward-Euler integration of the Faiman PV thermal ODE with G_poa as
! a first-class parameter. Enables a single-call irradiance sweep without
! re-instantiating the integrator.

module euler_irradiance_solver
  use precision_module, only: WP
  use pv_ode_module,    only: faiman_rhs
  implicit none
  private

  public :: solve_euler_irradiance

contains

  !> Forward-Euler integration with G_poa as sweep parameter.
  !!
  !! @param[in]  T0       Initial panel temperature (°C)
  !! @param[in]  G_poa    Plane-of-array irradiance (W m⁻²)
  !! @param[in]  T_amb_c  Ambient temperature (°C)
  !! @param[in]  WS       Wind speed (m s⁻¹)
  !! @param[in]  t_end    Integration horizon (s)
  !! @param[in]  dt       Euler step size (s)
  !! @param[out] T_final  Panel temperature at t_end (°C)
  subroutine solve_euler_irradiance(T0, G_poa, T_amb_c, WS, t_end, dt, T_final)
    real(WP), intent(in)  :: T0, G_poa, T_amb_c, WS, t_end, dt
    real(WP), intent(out) :: T_final

    real(WP) :: T, tt
    integer  :: n, i

    T  = T0
    tt = 0.0_WP
    if (dt <= 0.0_WP .or. t_end <= 0.0_WP) then
      T_final = T
      return
    end if
    n = max(1, nint(t_end / dt))

    do i = 1, n
      T  = T + dt * faiman_rhs(T, G_poa, T_amb_c, WS)
      tt = tt + dt
    end do
    T_final = T
  end subroutine solve_euler_irradiance

end module euler_irradiance_solver
