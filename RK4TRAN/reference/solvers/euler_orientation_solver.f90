! euler_orientation_solver.f90
! Forward-Euler integration of the Faiman PV thermal ODE with tilt-adjusted
! effective irradiance. The effective G is computed as:
!
!   G_eff(tilt) = G_horiz * cos(tilt_rad) + G_diff
!
! where tilt_rad is the panel tilt angle in radians, G_horiz is the
! direct horizontal component, and G_diff is the isotropic diffuse component.

module euler_orientation_solver
  use precision_module, only: WP
  use pv_ode_module,    only: faiman_rhs
  use constants_module, only: PI
  implicit none
  private

  public :: solve_euler_orientation
  public :: tilt_adjusted_irradiance

contains

  !> Effective irradiance for a given panel tilt angle.
  !!
  !! @param[in] G_horiz  Direct horizontal irradiance (W m⁻²)
  !! @param[in] G_diff   Isotropic diffuse irradiance (W m⁻²)
  !! @param[in] tilt_deg Panel tilt angle from horizontal (degrees)
  !! @return             Effective plane-of-array irradiance (W m⁻²)
  pure function tilt_adjusted_irradiance(G_horiz, G_diff, tilt_deg) result(G_eff)
    real(WP), intent(in) :: G_horiz, G_diff, tilt_deg
    real(WP) :: G_eff, tilt_rad

    tilt_rad = tilt_deg * PI / 180.0_WP
    G_eff    = G_horiz * cos(tilt_rad) + G_diff
    if (G_eff < G_diff) G_eff = G_diff   ! prevent negative direct component
  end function tilt_adjusted_irradiance

  !> Forward-Euler integration with tilt-adjusted effective irradiance.
  !!
  !! @param[in]  T0       Initial panel temperature (°C)
  !! @param[in]  G_horiz  Direct horizontal irradiance (W m⁻²)
  !! @param[in]  G_diff   Diffuse irradiance (W m⁻²)
  !! @param[in]  tilt_deg Panel tilt angle (degrees)
  !! @param[in]  T_amb_c  Ambient temperature (°C)
  !! @param[in]  WS       Wind speed (m s⁻¹)
  !! @param[in]  t_end    Integration horizon (s)
  !! @param[in]  dt       Euler step size (s)
  !! @param[out] T_final  Panel temperature at t_end (°C)
  subroutine solve_euler_orientation(T0, G_horiz, G_diff, tilt_deg, T_amb_c, WS, t_end, dt, T_final)
    real(WP), intent(in)  :: T0, G_horiz, G_diff, tilt_deg, T_amb_c, WS, t_end, dt
    real(WP), intent(out) :: T_final

    real(WP) :: T, tt, G_eff
    integer  :: n, i

    G_eff = tilt_adjusted_irradiance(G_horiz, G_diff, tilt_deg)
    T     = T0
    tt    = 0.0_WP
    if (dt <= 0.0_WP .or. t_end <= 0.0_WP) then
      T_final = T
      return
    end if
    n = max(1, nint(t_end / dt))

    do i = 1, n
      T  = T + dt * faiman_rhs(T, G_eff, T_amb_c, WS)
      tt = tt + dt
    end do
    T_final = T
  end subroutine solve_euler_orientation

end module euler_orientation_solver
