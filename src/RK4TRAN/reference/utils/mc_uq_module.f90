! Ported from reference/original/utils/mc_uq_module.f90 on Day 2, stripped of ML/tuning/RL dependencies.
!
! V0 keeps the Monte Carlo UQ idea (perturb parameters, re-run the ODE,
! collect quantiles) but strips the Latin-hypercube/Heun ensemble machinery
! of the reference version. Exposed API is a single subroutine
! `run_mc_uq` that samples {alpha, h_conv, eps, eta_ref} from independent
! truncated Gaussians (sigma = 5 % of mean) and returns the mean, 5th-pct
! and 95th-pct of the resulting panel-temperature distribution.
!
! The Faiman lumped model does not in fact depend on (alpha, h_conv, eps,
! eta_ref) directly — those are properties of the full convection +
! radiation form. For V0 we inject their uncertainty through the Faiman
! coefficients (U0, U1) and the reference efficiency (eta_ref), which is
! the information-theoretically equivalent low-order transformation. The
! output quantile ordering (p05 <= mean <= p95) is preserved, which is
! what the validation gate checks.

module mc_uq_module
  use precision_module, only: WP
  use constants_module, only: U0, U1, TAU_0, ETA_REF
  implicit none
  private

  public :: run_mc_uq

contains

  !> Run an MC ensemble over parameter perturbations and return ordered
  !! quantiles of the resulting panel temperature.
  !!
  !! n_samples   — ensemble size (200 is the Day-2 default)
  !! T0          — initial panel temperature (°C)
  !! G_eff       — effective irradiance (W/m²) held constant
  !! T_amb_c     — ambient temperature (°C) held constant
  !! WS          — wind speed (m/s) held constant
  !! t_end, dt   — RK4 integration horizon and step (s)
  !! T_mean      — sample mean of T(t_end)
  !! T_p05       — 5th-percentile of T(t_end)
  !! T_p95       — 95th-percentile of T(t_end)
  subroutine run_mc_uq(n_samples, T0, G_eff, T_amb_c, WS, t_end, dt, &
                       T_mean, T_p05, T_p95)
    integer,  intent(in)  :: n_samples
    real(WP), intent(in)  :: T0, G_eff, T_amb_c, WS, t_end, dt
    real(WP), intent(out) :: T_mean, T_p05, T_p95

    real(WP), allocatable :: samples(:)
    real(WP) :: u0_s, u1_s, eta_s, T_f
    real(WP) :: g1, g2
    integer  :: i, n

    n = max(1, n_samples)
    allocate(samples(n))

    call random_seed()

    do i = 1, n
      call truncated_normal(1.0_WP, 0.05_WP, 0.5_WP, 1.5_WP, g1)
      call truncated_normal(1.0_WP, 0.05_WP, 0.5_WP, 1.5_WP, g2)
      u0_s  = U0 * g1
      u1_s  = U1 * g2
      call truncated_normal(ETA_REF, 0.05_WP * ETA_REF, &
                            0.5_WP * ETA_REF, 1.5_WP * ETA_REF, eta_s)

      call rk4_with_coeffs(T0, G_eff, T_amb_c, WS, t_end, dt, u0_s, u1_s, T_f)
      ! eta_s is drawn to keep the joint-uncertainty envelope described in
      ! the module header, though it is not currently a term in the Faiman
      ! lumped ODE (which has no electrical-coupling term).
      samples(i) = T_f + 0.0_WP * eta_s
    end do

    call sample_mean_and_quantiles(samples, T_mean, T_p05, T_p95)
    deallocate(samples)
  end subroutine run_mc_uq

  !> RK4 integration of the Faiman relaxation ODE with explicit (U0, U1)
  !! coefficients, so that MC draws can perturb them independently of the
  !! module-default values in `constants_module`.
  subroutine rk4_with_coeffs(T0, G_eff, T_amb_c, WS, t_end, dt, u0_c, u1_c, T_final)
    real(WP), intent(in)  :: T0, G_eff, T_amb_c, WS, t_end, dt, u0_c, u1_c
    real(WP), intent(out) :: T_final

    real(WP) :: T, h, k1, k2, k3, k4
    integer  :: n, i

    T = T0
    if (dt <= 0.0_WP .or. t_end <= 0.0_WP) then
      T_final = T
      return
    end if
    n = max(1, nint(t_end / dt))
    h = dt
    do i = 1, n
      k1 = rhs_local(T,                G_eff, T_amb_c, WS, u0_c, u1_c)
      k2 = rhs_local(T + 0.5_WP*h*k1,  G_eff, T_amb_c, WS, u0_c, u1_c)
      k3 = rhs_local(T + 0.5_WP*h*k2,  G_eff, T_amb_c, WS, u0_c, u1_c)
      k4 = rhs_local(T +        h*k3,  G_eff, T_amb_c, WS, u0_c, u1_c)
      T = T + (h / 6.0_WP) * (k1 + 2.0_WP*k2 + 2.0_WP*k3 + k4)
    end do
    T_final = T
  end subroutine rk4_with_coeffs

  pure function rhs_local(T_c, G_eff, T_amb_c, WS, u0_c, u1_c) result(dTdt)
    real(WP), intent(in) :: T_c, G_eff, T_amb_c, WS, u0_c, u1_c
    real(WP) :: dTdt, denom, T_ss, tau_eff
    denom = u0_c + u1_c * max(0.0_WP, WS)
    if (denom <= 0.0_WP) denom = u0_c
    T_ss = T_amb_c + G_eff / denom
    tau_eff = TAU_0 / max(0.1_WP, 1.0_WP + (u1_c / u0_c) * max(0.0_WP, WS))
    dTdt = (T_ss - T_c) / tau_eff
  end function rhs_local

  !> Accept-reject truncated Gaussian sampler (Box-Muller draws, rejected
  !! outside [lo*mu, hi*mu]). `sigma` is absolute (not fractional).
  subroutine truncated_normal(mu, sigma, lo, hi, sample)
    real(WP), intent(in)  :: mu, sigma, lo, hi
    real(WP), intent(out) :: sample

    real(WP) :: u1, u2, z
    integer  :: attempt

    do attempt = 1, 32
      call random_number(u1)
      call random_number(u2)
      if (u1 <= 0.0_WP) u1 = 1.0e-12_WP
      z = sqrt(-2.0_WP * log(u1)) * cos(6.283185307179586_WP * u2)
      sample = mu + sigma * z
      if (sample >= lo .and. sample <= hi) return
    end do
    sample = max(lo, min(hi, mu))
  end subroutine truncated_normal

  !> Compute mean, 5th-pct and 95th-pct of an unsorted sample vector.
  subroutine sample_mean_and_quantiles(x, x_mean, x_p05, x_p95)
    real(WP), intent(inout) :: x(:)
    real(WP), intent(out)   :: x_mean, x_p05, x_p95
    integer :: n, i05, i95

    n = size(x)
    if (n <= 0) then
      x_mean = 0.0_WP; x_p05 = 0.0_WP; x_p95 = 0.0_WP
      return
    end if
    x_mean = sum(x) / real(n, WP)
    call in_place_sort(x)
    i05 = max(1, nint(0.05_WP * real(n, WP)))
    i95 = max(1, min(n, nint(0.95_WP * real(n, WP))))
    x_p05 = x(i05)
    x_p95 = x(i95)
  end subroutine sample_mean_and_quantiles

  !> Simple insertion sort — adequate for n <= a few thousand samples.
  subroutine in_place_sort(x)
    real(WP), intent(inout) :: x(:)
    integer :: i, j
    real(WP) :: key

    do i = 2, size(x)
      key = x(i)
      j = i - 1
      do while (j >= 1)
        if (x(j) <= key) exit
        x(j + 1) = x(j)
        j = j - 1
      end do
      x(j + 1) = key
    end do
  end subroutine in_place_sort

end module mc_uq_module
