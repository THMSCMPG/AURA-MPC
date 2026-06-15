! euler_power_solver.f90
! Forward-Euler integration of the Faiman PV thermal ODE with simultaneous
! P_mp (maximum power point) computation at every timestep.
!
! P_mp = eta(T) * G_poa * A,  eta(T) = eta_ref * (1 + beta*(T - T_ref))
! Uses the Sandia temperature derating from thermal_module.

module euler_power_solver
  use precision_module, only: WP
  use pv_ode_module,    only: faiman_rhs
  use thermal_module,   only: sandia_efficiency
  implicit none
  private

  public :: solve_euler_power

contains

  !> Forward-Euler integration with P_mp computed at every step.
  !!
  !! @param[in]  T0       Initial panel temperature (°C)
  !! @param[in]  G_poa    Plane-of-array irradiance (W m⁻²)
  !! @param[in]  T_amb_c  Ambient temperature (°C)
  !! @param[in]  WS       Wind speed (m s⁻¹)
  !! @param[in]  area     Module area (m²)
  !! @param[in]  t_end    Integration horizon (s)
  !! @param[in]  dt       Euler step size (s)
  !! @param[out] T_final  Panel temperature at t_end (°C)
  !! @param[out] P_final  P_mp at t_end (W)
  subroutine solve_euler_power(T0, G_poa, T_amb_c, WS, area, t_end, dt, T_final, P_final)
    real(WP), intent(in)  :: T0, G_poa, T_amb_c, WS, area, t_end, dt
    real(WP), intent(out) :: T_final, P_final

    real(WP) :: T, tt, eta
    integer  :: n, i

    T  = T0
    tt = 0.0_WP
    if (dt <= 0.0_WP .or. t_end <= 0.0_WP) then
      T_final = T
      P_final = sandia_efficiency(T) * G_poa * area
      return
    end if
    n = max(1, nint(t_end / dt))

    do i = 1, n
      T  = T + dt * faiman_rhs(T, G_poa, T_amb_c, WS)
      tt = tt + dt
    end do
    T_final = T
    eta     = sandia_efficiency(T_final)
    P_final = eta * G_poa * area
  end subroutine solve_euler_power

end module euler_power_solver
