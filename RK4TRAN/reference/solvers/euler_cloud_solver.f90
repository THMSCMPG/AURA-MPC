! euler_cloud_solver.f90
! Forward-Euler integration of the Faiman PV thermal ODE with cloud-cover-
! modulated effective irradiance.
!
! Cloud model:
!   G_eff(t) = G_clear * (1 - f_cloud(t))
!
! The cloud fraction f_cloud is passed as a scalar for a single step so
! that the caller can apply any cloud time-series model (sinusoidal,
! stochastic, satellite-derived, etc.) and call solve_euler_cloud_step
! at each timestep.

module euler_cloud_solver
  use precision_module, only: WP
  use pv_ode_module,    only: faiman_rhs
  implicit none
  private

  public :: solve_euler_cloud
  public :: cloud_effective_irradiance

contains

  !> Compute effective irradiance under a given cloud cover fraction.
  !!
  !! @param[in] G_clear   Clear-sky irradiance (W m⁻²)
  !! @param[in] f_cloud   Cloud cover fraction [0, 1]
  !! @return              Effective G_poa (W m⁻²)
  pure function cloud_effective_irradiance(G_clear, f_cloud) result(G_eff)
    real(WP), intent(in) :: G_clear, f_cloud
    real(WP) :: G_eff

    G_eff = G_clear * (1.0_WP - max(0.0_WP, min(1.0_WP, f_cloud)))
  end function cloud_effective_irradiance

  !> Forward-Euler integration over a fixed cloud-cover time window.
  !!
  !! Cloud fraction is fixed at f_cloud for the full integration window.
  !! For a time-varying cloud fraction, call this once per timestep with
  !! updated f_cloud and T0 = previous T_final.
  !!
  !! @param[in]  T0       Initial panel temperature (°C)
  !! @param[in]  G_clear  Clear-sky irradiance (W m⁻²)
  !! @param[in]  f_cloud  Cloud cover fraction [0, 1] for this window
  !! @param[in]  T_amb_c  Ambient temperature (°C)
  !! @param[in]  WS       Wind speed (m s⁻¹)
  !! @param[in]  t_end    Integration horizon for this window (s)
  !! @param[in]  dt       Euler step size (s)
  !! @param[out] T_final  Panel temperature at end of window (°C)
  subroutine solve_euler_cloud(T0, G_clear, f_cloud, T_amb_c, WS, t_end, dt, T_final)
    real(WP), intent(in)  :: T0, G_clear, f_cloud, T_amb_c, WS, t_end, dt
    real(WP), intent(out) :: T_final

    real(WP) :: T, tt, G_eff
    integer  :: n, i

    G_eff = cloud_effective_irradiance(G_clear, f_cloud)
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
  end subroutine solve_euler_cloud

end module euler_cloud_solver
