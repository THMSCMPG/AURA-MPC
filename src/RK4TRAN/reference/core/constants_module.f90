! Ported from reference/original/core/constants_module.f90 on Day 2, stripped of ML/tuning/RL dependencies.
!
! SimV0 keeps only the constants and module-default physics parameters that
! the Faiman thermal model (Eqs. 3.1, 3.4, 3.6) and the Sandia efficiency
! derating (Eq. 3.2) actually need. Unit-conversion helpers, full CODATA
! electromagnetic constants, spectral-range constants, and the
! `print_constants_info` I/O helper are not used by the V0 physics path and
! have been removed. Nothing here depends on tuning, ML, RL, decision-tree,
! or weather-driver-hook modules.

module constants_module
  use precision_module, only: WP
  implicit none
  private

  ! Mathematical constants
  real(WP), parameter, public :: PI         = 3.14159265358979323846_WP
  real(WP), parameter, public :: TWO_PI     = 2.0_WP * PI
  real(WP), parameter, public :: DEG_TO_RAD = PI / 180.0_WP

  ! Stefan–Boltzmann (CODATA 2018)
  real(WP), parameter, public :: SIGMA_SB   = 5.670374419E-8_WP

  ! Time
  real(WP), parameter, public :: HOUR_TO_SEC = 3600.0_WP
  real(WP), parameter, public :: DAY_TO_SEC  = 86400.0_WP

  ! Temperature
  real(WP), parameter, public :: T_CELSIUS_TO_KELVIN = 273.15_WP

  !-----------------------------------------------------------------------
  ! Hard-coded module physics defaults (formerly exposed via tuning module).
  ! These are the values SimV0 uses when no runtime override is provided;
  ! they replace the call sites `call tune(...)` in the reference lofi
  ! solver with a static parameter lookup.
  !-----------------------------------------------------------------------

  ! Solar absorptance of the module front surface [-]
  real(WP), parameter, public :: ALPHA_ABS   = 0.90_WP
  ! Long-wave emissivity of the module front surface [-]
  real(WP), parameter, public :: EPS_LW      = 0.85_WP
  ! Natural + forced convection coefficient baseline [W m^-2 K^-1]
  real(WP), parameter, public :: H_CONV      = 10.0_WP

  ! Faiman thermal model coefficients (Eq. 3.4)
  !   T_panel = T_amb + G_eff / (U0 + U1 * WS)
  ! U0 in W m^-2 K^-1, U1 in W s m^-3 K^-1. Values are the PVsyst defaults
  ! widely used in the field for c-Si glass/backsheet modules.
  real(WP), parameter, public :: U0          = 25.0_WP
  real(WP), parameter, public :: U1          = 6.84_WP

  ! Effective thermal time constant [s] (Eq. 3.6). Typical 5–10 minutes.
  real(WP), parameter, public :: TAU_0       = 300.0_WP

  ! Sandia temperature coefficient of maximum power [/K], negative for c-Si.
  real(WP), parameter, public :: BETA_PMAX   = -0.0045_WP

  ! Reference module efficiency at STC [-]
  real(WP), parameter, public :: ETA_REF     = 0.18_WP

  ! STC reference temperature [°C]
  real(WP), parameter, public :: T_REF_C     = 25.0_WP
end module constants_module
