! Ported from reference/original/environment/atmospheric_module.f90 on Day 2, stripped of ML/tuning/RL dependencies.
!
! SimV0 keeps only the cloud-cover / spectral-mismatch product used to
! derive effective in-plane irradiance (Eq. 3.5):
!
!   G_eff = G_poa * (1 - CC^gamma) * M_spectral
!
! with `gamma = 1.0` and `M_spectral = 1.0` hard-coded in this tier.
! All ~1,000 lines of the reference module that ingest atmospheric LUTs,
! aerosol tables, and spectrum interpolators have been removed; SimV4 owns
! that code path. Nothing here touches `tuning module` or the weather-driver
! hook.

module atmospheric_module
  use precision_module, only: WP
  implicit none
  private

  public :: effective_irradiance

  ! V0 fixed exponent and spectral factor (documented defaults, not tuning).
  real(WP), parameter, public :: CC_GAMMA    = 1.0_WP
  real(WP), parameter, public :: M_SPECTRAL0 = 1.0_WP

contains

  !> Effective POA irradiance reaching the cell (Eq. 3.5).
  !! Inputs are clamped to physically meaningful ranges by the caller.
  pure function effective_irradiance(G_poa, CC, M_spectral) result(G_eff)
    real(WP), intent(in) :: G_poa, CC, M_spectral
    real(WP) :: G_eff
    real(WP) :: cc_clamped

    cc_clamped = max(0.0_WP, min(1.0_WP, CC))
    G_eff = G_poa * (1.0_WP - cc_clamped**CC_GAMMA) * M_spectral
    if (G_eff < 0.0_WP) G_eff = 0.0_WP
  end function effective_irradiance

end module atmospheric_module
