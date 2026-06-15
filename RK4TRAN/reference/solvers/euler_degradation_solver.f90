! euler_degradation_solver.f90
! Forward-Euler integration of the Faiman PV thermal ODE with an annual
! power-degradation factor applied to both effective G and the efficiency.
!
! Degradation model:
!   f_d(yr) = 1 - degrad_rate * yr        (linear annual degradation)
!   G_eff   = G_peak * f_d * fill_factor
!   eta     = eta_ref * f_d * (1 + beta*(T - T_ref))

module euler_degradation_solver
  use precision_module, only: WP
  use pv_ode_module,    only: faiman_rhs
  use thermal_module,   only: sandia_efficiency
  implicit none
  private

  public :: solve_euler_degradation

contains

  !> Forward-Euler integration with annual degradation factor.
  !!
  !! @param[in]  T0           Initial panel temperature (°C)
  !! @param[in]  G_peak       Clear-sky peak irradiance (W m⁻²)
  !! @param[in]  fill_factor  Fraction of day with significant irradiance [-]
  !! @param[in]  T_amb_c      Ambient temperature (°C)
  !! @param[in]  WS           Wind speed (m s⁻¹)
  !! @param[in]  degrad_rate  Annual degradation rate (/yr, e.g. 0.005 for 0.5%/yr)
  !! @param[in]  year         Current year (0-based)
  !! @param[in]  area         Module area (m²)
  !! @param[in]  t_end        Integration horizon (s)
  !! @param[in]  dt           Euler step size (s)
  !! @param[out] T_final      Panel temperature at t_end (°C)
  !! @param[out] P_final      Derated P_mp at t_end (W)
  !! @param[out] degrad_fac   Degradation factor applied this year [-]
  subroutine solve_euler_degradation(T0, G_peak, fill_factor, T_amb_c, WS, &
                                     degrad_rate, year, area, t_end, dt,     &
                                     T_final, P_final, degrad_fac)
    real(WP), intent(in)  :: T0, G_peak, fill_factor, T_amb_c, WS
    real(WP), intent(in)  :: degrad_rate, area, t_end, dt
    integer,  intent(in)  :: year
    real(WP), intent(out) :: T_final, P_final, degrad_fac

    real(WP) :: T, tt, G_eff, eta
    integer  :: n, i

    degrad_fac = max(0.5_WP, 1.0_WP - degrad_rate * real(year, WP))
    G_eff = G_peak * degrad_fac * fill_factor

    T  = T0
    tt = 0.0_WP
    if (dt <= 0.0_WP .or. t_end <= 0.0_WP) then
      T_final = T
      P_final = sandia_efficiency(T) * G_eff * area * degrad_fac
      return
    end if
    n = max(1, nint(t_end / dt))

    do i = 1, n
      T  = T + dt * faiman_rhs(T, G_eff, T_amb_c, WS)
      tt = tt + dt
    end do
    T_final = T
    eta     = sandia_efficiency(T_final) * degrad_fac
    if (eta < 0.0_WP) eta = 0.0_WP
    P_final = eta * G_peak * area
  end subroutine solve_euler_degradation

end module euler_degradation_solver
