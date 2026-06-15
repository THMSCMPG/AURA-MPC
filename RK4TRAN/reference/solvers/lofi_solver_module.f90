! Ported from reference/original/solvers/lofi_solver_module.f90 on Day 2, stripped of ML/tuning/RL dependencies.
!
! Stripped `use` statements (each with a one-line justification):
!   * `use tuning module`           — static constants now live in `constants_module`.
!   * `use ML orchestrator module`  — no ML path in V0 (V4 owns this).
!   * `use decision tree module`    — no routing in V0, probs are deterministic.
!   * `use RL interface module`     — no RL policy in V0.
!   * `use debug logger module`     — avoids pulling in the logger dependency graph.
!   * `use CSV logger module`       — V0 writes JSON on stdout, not CSV side-channels.
!
! Any `call tune(...)` sites in the reference are replaced with direct
! references to the parameters in `constants_module`. This keeps V0 a pure
! physics path.

module lofi_solver_module
  use precision_module, only: WP
  use pv_ode_module,    only: faiman_rhs
  implicit none
  private

  public :: solve_euler
  public :: solve_rk4

contains

  !> Forward-Euler integration of the Faiman ODE from (t0, T0) to t_end.
  !! Holds G_eff, T_amb, WS constant over the window (V0 scalar-input case).
  subroutine solve_euler(T0, G_eff, T_amb_c, WS, t_end, dt, T_final)
    real(WP), intent(in)  :: T0, G_eff, T_amb_c, WS, t_end, dt
    real(WP), intent(out) :: T_final

    real(WP) :: T, tt
    integer  :: n, i

    T = T0
    tt = 0.0_WP
    if (dt <= 0.0_WP .or. t_end <= 0.0_WP) then
      T_final = T
      return
    end if
    n = max(1, nint(t_end / dt))

    do i = 1, n
      T = T + dt * faiman_rhs(T, G_eff, T_amb_c, WS)
      tt = tt + dt
    end do
    T_final = T
  end subroutine solve_euler

  !> Classical RK4 integration of the Faiman ODE.
  subroutine solve_rk4(T0, G_eff, T_amb_c, WS, t_end, dt, T_final)
    real(WP), intent(in)  :: T0, G_eff, T_amb_c, WS, t_end, dt
    real(WP), intent(out) :: T_final

    real(WP) :: T, k1, k2, k3, k4, h
    integer  :: n, i

    T = T0
    if (dt <= 0.0_WP .or. t_end <= 0.0_WP) then
      T_final = T
      return
    end if
    n = max(1, nint(t_end / dt))
    h = dt

    do i = 1, n
      k1 = faiman_rhs(T,              G_eff, T_amb_c, WS)
      k2 = faiman_rhs(T + 0.5_WP*h*k1, G_eff, T_amb_c, WS)
      k3 = faiman_rhs(T + 0.5_WP*h*k2, G_eff, T_amb_c, WS)
      k4 = faiman_rhs(T +        h*k3, G_eff, T_amb_c, WS)
      T = T + (h / 6.0_WP) * (k1 + 2.0_WP*k2 + 2.0_WP*k3 + k4)
    end do
    T_final = T
  end subroutine solve_rk4

end module lofi_solver_module
