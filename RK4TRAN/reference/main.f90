! Ported from reference/original/main.f90 shape on Day 2, patterned line-for-line on src/simv4/src/main.f90.
! Stripped of ML/tuning/RL dependencies: SimV0 is a pure physics path.
!
! Supports:
!   --test    run with hard-coded inputs matching SimV4's TEST_* literals
!   --mc      enable Monte Carlo UQ path (appends T_panel_{mean,p05,p95})

program simv0
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  use precision_module,   only: WP
  use json_io_module,     only: trajectory_type, read_json_input, &
                                write_json_output, write_json_output_mc
  use sim0_command_module, only: sim_inputs_type, sim_outputs_type, execute_simv0
  implicit none

  real(WP), allocatable :: img_features(:)
  type(sim_inputs_type)  :: inputs
  type(sim_outputs_type) :: outputs
  real(WP) :: t_s, G_poa, T_amb, WS, CC, lat, lon
  integer  :: ios
  logical  :: test_mode, mc_mode

  ! Test vector cross-tier comparable with SimV4.
  real(WP), parameter :: TEST_TIME_SECONDS = 43200.0_WP
  real(WP), parameter :: TEST_G_POA = 850.5_WP
  real(WP), parameter :: TEST_T_AMB = 28.3_WP
  real(WP), parameter :: TEST_WS    = 4.2_WP
  real(WP), parameter :: TEST_CC    = 0.15_WP
  real(WP), parameter :: TEST_LAT   = 36.5_WP
  real(WP), parameter :: TEST_LON   = -87.3_WP

  integer :: c0, c1, cr
  real(WP) :: runtime_ms

  call parse_cli(test_mode, mc_mode)

  call system_clock(c0, cr)

  if (test_mode) then
    t_s   = TEST_TIME_SECONDS
    G_poa = TEST_G_POA
    T_amb = TEST_T_AMB
    WS    = TEST_WS
    CC    = TEST_CC
    lat   = TEST_LAT
    lon   = TEST_LON
    allocate(img_features(5))
    img_features = [0.12_WP, 0.34_WP, 0.89_WP, 0.56_WP, 0.22_WP]
  else
    call read_json_input(5, t_s, G_poa, T_amb, WS, CC, lat, lon, img_features, ios)
    if (ios /= 0) then
      write(0, '(A,I0)') 'ERROR: malformed JSON input, code=', ios
      stop 1
    end if
  end if

  call sanitize_inputs(t_s, G_poa, T_amb, WS, CC, lat, lon, img_features)

  inputs%t_s   = t_s
  inputs%G_poa = G_poa
  inputs%T_amb = T_amb
  inputs%WS    = WS
  inputs%CC    = CC
  inputs%lat   = lat
  inputs%lon   = lon
  inputs%mc_enabled = mc_mode
  inputs%mc_samples = 200

  call execute_simv0(inputs, outputs)

  call system_clock(c1)
  if (cr > 0) then
    runtime_ms = 1000.0_WP * real(c1 - c0, WP) / real(cr, WP)
  else
    runtime_ms = 0.0_WP
  end if

  if (ieee_is_nan(outputs%T_panel)) then
    ! SimV4-style fallback: zero the envelope, write, exit 0.
    outputs%T_panel      = 0.0_WP
    outputs%probs        = 0.0_WP
    outputs%confidence   = 0.0_WP
    outputs%M_spectral   = 0.0_WP
    outputs%efficiency   = 0.0_WP
    outputs%trajectory   = trajectory_type()
    outputs%T_panel_mean = 0.0_WP
    outputs%T_panel_p05  = 0.0_WP
    outputs%T_panel_p95  = 0.0_WP
  end if

  if (mc_mode) then
    call write_json_output_mc(6, outputs%T_panel, outputs%probs, outputs%trajectory, &
                              outputs%confidence, runtime_ms, outputs%M_spectral, outputs%efficiency, &
                              outputs%T_panel_mean, outputs%T_panel_p05, outputs%T_panel_p95)
  else
    call write_json_output(6, outputs%T_panel, outputs%probs, outputs%trajectory, &
                           outputs%confidence, runtime_ms, outputs%M_spectral, outputs%efficiency)
  end if

contains

  subroutine parse_cli(test_mode, mc_mode)
    logical, intent(out) :: test_mode, mc_mode
    character(len=128) :: arg
    integer :: i, n

    test_mode = .false.
    mc_mode   = .false.
    n = command_argument_count()
    do i = 1, n
      call get_command_argument(i, arg)
      select case (trim(arg))
      case ('--test')
        test_mode = .true.
      case ('--mc')
        mc_mode = .true.
      case default
        write(0, '(A,A)') 'WARNING: unknown argument ignored: ', trim(arg)
      end select
    end do
  end subroutine parse_cli

  subroutine sanitize_inputs(t_s, G_poa, T_amb, WS, CC, lat, lon, img)
    real(WP), intent(inout) :: t_s, G_poa, T_amb, WS, CC, lat, lon
    real(WP), intent(inout) :: img(:)
    integer :: i

    call sanitize_scalar(t_s,   0.0_WP,    86400.0_WP, 't_s')
    call sanitize_scalar(G_poa, 0.0_WP,     1400.0_WP, 'G_poa')
    call sanitize_scalar(T_amb, -40.0_WP,     70.0_WP, 'T_amb')
    call sanitize_scalar(WS,    0.0_WP,       60.0_WP, 'WS')
    call sanitize_scalar(CC,    0.0_WP,        1.0_WP, 'CC')
    call sanitize_scalar(lat,   -90.0_WP,     90.0_WP, 'lat')
    call sanitize_scalar(lon,   -180.0_WP,   180.0_WP, 'lon')

    do i = 1, size(img)
      call sanitize_scalar(img(i), -5.0_WP, 5.0_WP, 'image_feature')
    end do
  end subroutine sanitize_inputs

  subroutine sanitize_scalar(x, xmin, xmax, name)
    real(WP), intent(inout) :: x
    real(WP), intent(in)    :: xmin, xmax
    character(len=*), intent(in) :: name

    if (ieee_is_nan(x)) then
      write(0, '(A,A)') 'WARNING: NaN input clamped for ', trim(name)
      x = 0.5_WP * (xmin + xmax)
      return
    end if
    x = max(xmin, min(xmax, x))
  end subroutine sanitize_scalar

end program simv0
