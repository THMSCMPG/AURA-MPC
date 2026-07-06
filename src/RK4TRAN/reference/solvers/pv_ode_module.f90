! Ported from reference/original/solvers/pv_ode_module.f90 on Day 2, stripped of ML/tuning/RL dependencies.
!
! SimV0 keeps only the lumped-capacitance PV thermal ODE RHS (Eq. 3.1),
! exposed as a pure function. The reference version is a 1,100-line
! dispatcher covering Faiman, Sandia, Skoplaki, NOCT, and PINN-hybrid
! fidelity paths; V0 retains exactly one: the Faiman relaxation form,
! expressed as
!     dT/dt = (T_ss(G_eff, T_amb, WS) - T) / tau(WS)
! This matches the steady-state of the full convection+radiation balance
! under the Faiman lumped coefficients (U0, U1) — no separate
! `tuning module` or ML orchestrator is needed.

module pv_ode_module
  use precision_module, only: WP
  use thermal_module,   only: faiman_steady_state, faiman_time_constant
  implicit none
  private

  public :: faiman_rhs

contains

  !> Right-hand side of the Faiman lumped thermal ODE (Eq. 3.1).
  !!
  !! dT/dt = (T_ss - T) / tau
  !!
  !! Inputs are all in °C and m/s; output has units K/s (equivalently °C/s).
  !! Pure: safe to call from any integrator without side effects.
  pure function faiman_rhs(T_c, G_eff, T_amb_c, WS) result(dTdt)
    real(WP), intent(in) :: T_c, G_eff, T_amb_c, WS
    real(WP) :: dTdt
    real(WP) :: T_ss, tau

    T_ss = faiman_steady_state(G_eff, T_amb_c, WS)
    tau  = faiman_time_constant(WS)
    if (tau <= 0.0_WP) then
      dTdt = 0.0_WP
    else
      dTdt = (T_ss - T_c) / tau
    end if
  end function faiman_rhs

end module pv_ode_module
