! Ported from reference/original/materials/thermal_module.f90 on Day 2, stripped of ML/tuning/RL dependencies.
!
! SimV0 keeps only the Faiman steady-state panel temperature (Eq. 3.4), the
! effective thermal time constant (Eq. 3.6), and the Sandia efficiency
! derating (Eq. 3.2). The ~500-line layered-material property tables,
! temperature-dependent k(T) fits, and interface contact-resistance
! machinery from the reference version are unused by V0 and have been
! removed. Nothing here touches `tuning module` or the ML orchestrator.

module thermal_module
  use precision_module, only: WP
  use constants_module, only: U0, U1, TAU_0, BETA_PMAX, ETA_REF, T_REF_C
  implicit none
  private

  public :: faiman_steady_state
  public :: faiman_time_constant
  public :: sandia_efficiency

contains

  !> Faiman steady-state panel temperature (°C) from effective irradiance,
  !! ambient temperature and wind speed. Eq. 3.4.
  pure function faiman_steady_state(G_eff, T_amb_c, WS) result(T_panel_c)
    real(WP), intent(in) :: G_eff, T_amb_c, WS
    real(WP) :: T_panel_c
    real(WP) :: denom

    denom = U0 + U1 * max(0.0_WP, WS)
    if (denom <= 0.0_WP) denom = U0
    T_panel_c = T_amb_c + G_eff / denom
  end function faiman_steady_state

  !> Wind-speed-adjusted effective thermal time constant (s). Eq. 3.6.
  !! Faster ventilation → smaller tau.
  pure function faiman_time_constant(WS) result(tau)
    real(WP), intent(in) :: WS
    real(WP) :: tau, denom

    denom = 1.0_WP + (U1 / U0) * max(0.0_WP, WS)
    if (denom <= 0.0_WP) denom = 1.0_WP
    tau = TAU_0 / denom
  end function faiman_time_constant

  !> Sandia temperature-only efficiency derating (Eq. 3.2).
  !! Returns a fractional efficiency (e.g. 0.171 for a 18 % module at 45 °C).
  pure function sandia_efficiency(T_panel_c) result(eta)
    real(WP), intent(in) :: T_panel_c
    real(WP) :: eta

    eta = ETA_REF * (1.0_WP + BETA_PMAX * (T_panel_c - T_REF_C))
    if (eta < 0.0_WP) eta = 0.0_WP
    if (eta > 1.0_WP) eta = 1.0_WP
  end function sandia_efficiency

end module thermal_module
