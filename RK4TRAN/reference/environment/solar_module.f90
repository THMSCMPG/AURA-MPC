! Ported from reference/original/environment/solar_module.f90 on Day 2, stripped of ML/tuning/RL dependencies.
!
! SimV0 keeps two pure routines from the 1,200-line reference module:
!   * `solar_declination_spencer` — Spencer (1971) Fourier-series declination.
!   * `diurnal_irradiance`        — the half-sine diurnal G(t) model (Eq. 3.3).
!
! Removed from the reference version: the dependencies on
! `weather driver hook module`, `tuning module`, `climate zone module`, and
! the climate-zone lookup tables that require external data files.

module solar_module
  use precision_module, only: WP
  use constants_module, only: PI, DEG_TO_RAD, DAY_TO_SEC
  implicit none
  private

  public :: solar_declination_spencer
  public :: diurnal_irradiance

contains

  !> Spencer (1971) solar declination (radians).
  !! Input: day-of-year in [1, 365].
  pure function solar_declination_spencer(doy) result(delta)
    integer, intent(in) :: doy
    real(WP) :: delta
    real(WP) :: gamma

    gamma = 2.0_WP * PI * (real(max(1, min(365, doy)), WP) - 1.0_WP) / 365.0_WP
    delta = 0.006918_WP                                         &
          - 0.399912_WP * cos(gamma)                            &
          + 0.070257_WP * sin(gamma)                            &
          - 0.006758_WP * cos(2.0_WP * gamma)                   &
          + 0.000907_WP * sin(2.0_WP * gamma)                   &
          - 0.002697_WP * cos(3.0_WP * gamma)                   &
          + 0.001480_WP * sin(3.0_WP * gamma)
  end function solar_declination_spencer

  !> Diurnal irradiance (Eq. 3.3): half-sine bell pinned at solar noon with
  !! a peak of `g_peak`. `t_s` is seconds since local midnight.
  !!
  !! This is the SimV0 fallback used to populate `trajectory.T_*` values
  !! when only a single scalar G_poa is supplied on stdin.
  pure function diurnal_irradiance(t_s, g_peak) result(g)
    real(WP), intent(in) :: t_s, g_peak
    real(WP) :: g
    real(WP) :: arg

    arg = PI * t_s / DAY_TO_SEC
    g = max(0.0_WP, g_peak * sin(arg))
  end function diurnal_irradiance

end module solar_module
