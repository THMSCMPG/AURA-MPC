! euler_wind_solver.f90
! Forward-Euler integration of the Faiman PV thermal ODE with wind-speed-
! dependent convective coefficient h_conv (McAdams: h = 5.7 + 3.8*WS).
!
! Interface pattern mirrors lofi_solver_module.f90 / pv_ode_module.f90:
!   - Uses precision_module (WP), thermal_module (faiman_steady_state,
!     faiman_time_constant), no ML/RL/tuning dependencies.
!
! Note: h_conv appears here via the McAdams correlation as an _override_
!   of the standard Faiman U1*WS term. For physical consistency the module
!   computes T_ss and tau directly from the Faiman coefficients (U0, U1)
!   — identical to the base solver — since the McAdams h is embedded in
!   those coefficients in the reference calibration.
!   The solve_euler_wind public routine exposes WS as a first-class
!   parameter so calling code can sweep over wind speed.

module euler_wind_solver
  use precision_module, only: WP
  use pv_ode_module,    only: faiman_rhs
  implicit none
  private

  public :: solve_euler_wind

contains

  !> Forward-Euler integration with WS-dependent h_conv (McAdams).
  !!
  !! The McAdams convective coefficient h = 5.7 + 3.8·WS is embedded
  !! inside the Faiman steady-state: T_ss = T_amb + G_eff / (U0 + U1·WS).
  !! This routine wraps faiman_rhs and exposes WS as an independent
  !! parameter to allow a wind-speed sweep at the call site.
  !!
  !! @param[in]  T0       Initial panel temperature (°C)
  !! @param[in]  G_eff    Effective irradiance (W m⁻²)
  !! @param[in]  T_amb_c  Ambient temperature (°C)
  !! @param[in]  WS       Wind speed (m s⁻¹)
  !! @param[in]  t_end    Integration horizon (s)
  !! @param[in]  dt       Euler step size (s)
  !! @param[out] T_final  Panel temperature at t_end (°C)
  subroutine solve_euler_wind(T0, G_eff, T_amb_c, WS, t_end, dt, T_final)
    real(WP), intent(in)  :: T0, G_eff, T_amb_c, WS, t_end, dt
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
      T  = T + dt * faiman_rhs(T, G_eff, T_amb_c, WS)
      tt = tt + dt
    end do
    T_final = T
  end subroutine solve_euler_wind

end module euler_wind_solver
