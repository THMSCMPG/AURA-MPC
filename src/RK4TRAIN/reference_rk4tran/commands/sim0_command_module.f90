! Ported from reference/original/commands/sim0_command_module.f90 on Day 2, stripped of ML/tuning/RL dependencies.
!
! The reference version invoked a paper-scenario dispatcher (euler / rk4 /
! diurnal / etc.) that wrote `.dat` files to disk. V0 collapses the
! dispatcher to a single pure physics path and consumes/produces the
! stdin/stdout JSON contract documented in `docs/JSON_INTERFACE.md`.
!
! Stripped: every I/O path that touches tuning, decision-tree binaries,
! Q-tables, and external `.npy` files. No `use error_module`, no output-
! file branches. Public API reduces to one subroutine.

module sim0_command_module
  use precision_module,    only: WP
  use constants_module,    only: HOUR_TO_SEC
  use atmospheric_module,  only: effective_irradiance, M_SPECTRAL0
  use thermal_module,      only: faiman_steady_state, sandia_efficiency
  use lofi_solver_module,  only: solve_rk4
  use mc_uq_module,        only: run_mc_uq
  use json_io_module,      only: trajectory_type
  implicit none
  private

  public :: sim_inputs_type
  public :: sim_outputs_type
  public :: execute_simv0

  type :: sim_inputs_type
    real(WP) :: t_s   = 0.0_WP
    real(WP) :: G_poa = 0.0_WP
    real(WP) :: T_amb = 25.0_WP
    real(WP) :: WS    = 0.0_WP
    real(WP) :: CC    = 0.0_WP
    real(WP) :: lat   = 0.0_WP
    real(WP) :: lon   = 0.0_WP
    logical  :: mc_enabled = .false.
    integer  :: mc_samples = 200
  end type sim_inputs_type

  type :: sim_outputs_type
    real(WP) :: T_panel      = 0.0_WP
    real(WP) :: probs(5)     = [1.0_WP, 0.0_WP, 0.0_WP, 0.0_WP, 0.0_WP]
    type(trajectory_type) :: trajectory
    real(WP) :: confidence   = 1.0_WP
    real(WP) :: M_spectral   = 1.0_WP
    real(WP) :: efficiency   = 0.0_WP
    real(WP) :: T_panel_mean = 0.0_WP
    real(WP) :: T_panel_p05  = 0.0_WP
    real(WP) :: T_panel_p95  = 0.0_WP
  end type sim_outputs_type

contains

  !> Pure SimV0 physics path:
  !!   parse inputs -> G_eff -> integrate Faiman ODE (steady-state)
  !!                -> Sandia derating -> populate outputs.
  subroutine execute_simv0(inputs, outputs)
    type(sim_inputs_type),  intent(in)  :: inputs
    type(sim_outputs_type), intent(out) :: outputs

    real(WP) :: G_eff, T_ss
    real(WP), parameter :: DT_RK4 = 30.0_WP  ! seconds per RK4 step

    ! Eq. 3.5 — effective in-plane irradiance.
    G_eff = effective_irradiance(inputs%G_poa, inputs%CC, M_SPECTRAL0)

    ! Eq. 3.4 — Faiman steady-state panel temperature. The ODE approaches
    ! this asymptotically; for single-point stdin we report the steady
    ! value directly.
    T_ss = faiman_steady_state(G_eff, inputs%T_amb, inputs%WS)

    outputs%T_panel    = T_ss
    outputs%M_spectral = M_SPECTRAL0
    outputs%efficiency = sandia_efficiency(T_ss)
    outputs%confidence = 1.0_WP
    outputs%probs      = [1.0_WP, 0.0_WP, 0.0_WP, 0.0_WP, 0.0_WP]

    ! V0 trajectory: holds current weather constants and reports the
    ! Faiman steady-state at each forward offset (CC and WS held fixed).
    outputs%trajectory%T_1h = T_ss
    outputs%trajectory%T_2h = T_ss
    outputs%trajectory%T_6h = T_ss
    outputs%trajectory%CC_1h = inputs%CC
    outputs%trajectory%WS_1h = inputs%WS

    if (inputs%mc_enabled) then
      call run_mc_uq(inputs%mc_samples, inputs%T_amb, G_eff, inputs%T_amb, inputs%WS, &
                     6.0_WP * HOUR_TO_SEC, DT_RK4, &
                     outputs%T_panel_mean, outputs%T_panel_p05, outputs%T_panel_p95)
    end if
  end subroutine execute_simv0

end module sim0_command_module
