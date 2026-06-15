! Ported from src/simv4/src/json_io_module.f90 on Day 2 for SimV0.
! Minimal, dependency-free JSON reader/writer for stdin/stdout.
!
! Adaptations from the SimV4 original:
!   * `write_json_output` now emits the canonical short-name schema
!     documented in `docs/JSON_INTERFACE.md` (`T_panel`, `probs`,
!     `M_spectral`, `efficiency`, `runtime_ms`) rather than the SimV4
!     extended names.
!   * Adds `write_json_output_mc` which emits the `--mc` superset
!     (adds `T_panel_mean`, `T_panel_p05`, `T_panel_p95`).
!   * Input parser accepts `img_features` (V0 reads and discards it) and
!     also falls back to the SimV4 key `image_features` for cross-tier
!     compatibility during the schema transition.

module json_io_module
  use precision_module, only: WP
  implicit none
  private

  public :: trajectory_type
  public :: read_json_input
  public :: write_json_output
  public :: write_json_output_mc

  type :: trajectory_type
    real(WP) :: T_1h  = 0.0_WP
    real(WP) :: T_2h  = 0.0_WP
    real(WP) :: T_6h  = 0.0_WP
    real(WP) :: CC_1h = 0.0_WP
    real(WP) :: WS_1h = 0.0_WP
  end type trajectory_type

contains

  subroutine read_json_input(unit, t_s, G_poa, T_amb, WS, CC, lat, lon, img_features, ios)
    integer,  intent(in)               :: unit
    real(WP), intent(out)              :: t_s, G_poa, T_amb, WS, CC, lat, lon
    real(WP), allocatable, intent(out) :: img_features(:)
    integer,  intent(out)              :: ios

    character(len=20000) :: json
    character(len=512)   :: line
    integer :: ios_local, p
    logical :: found

    json = ''
    do
      read(unit, '(A)', iostat=ios_local) line
      if (ios_local /= 0) exit
      p = len_trim(json)
      if (p + len_trim(line) + 1 <= len(json)) then
        json(p+1:p+len_trim(line)) = trim(line)
      else
        write(0, '(A)') 'WARNING: JSON input exceeded 20000 character limit; trailing content truncated.'
        exit
      end if
    end do

    if (len_trim(json) == 0) then
      ios = 1
      return
    end if

    call parse_json_number(json, 't_s',   t_s,   found); if (.not. found) then; ios = 2; return; end if
    call parse_json_number(json, 'G_poa', G_poa, found); if (.not. found) then; ios = 3; return; end if
    call parse_json_number(json, 'T_amb', T_amb, found); if (.not. found) then; ios = 4; return; end if
    call parse_json_number(json, 'WS',    WS,    found); if (.not. found) then; ios = 5; return; end if
    call parse_json_number(json, 'CC',    CC,    found); if (.not. found) then; ios = 6; return; end if
    call parse_json_number(json, 'lat',   lat,   found); if (.not. found) then; ios = 7; return; end if
    call parse_json_number(json, 'lon',   lon,   found); if (.not. found) then; ios = 8; return; end if

    call parse_json_array(json, 'img_features', img_features, found)
    if (.not. found) call parse_json_array(json, 'image_features', img_features, found)
    if (.not. found) then
      if (allocated(img_features)) deallocate(img_features)
      allocate(img_features(5))
      img_features = 0.0_WP
    end if

    ios = 0
  end subroutine read_json_input

  !> Canonical plain output (no --mc).
  subroutine write_json_output(unit, T_panel, probs, trajectory, confidence, runtime_ms, M_spectral, efficiency)
    integer,  intent(in) :: unit
    real(WP), intent(in) :: T_panel, probs(5), confidence, runtime_ms, M_spectral, efficiency
    type(trajectory_type), intent(in) :: trajectory

    write(unit, '(A)')           '{'
    write(unit, '(A,F12.6,A)')   '  "T_panel": ',    T_panel, ','
    call write_probs_array(unit, probs)
    call write_trajectory(unit, trajectory)
    write(unit, '(A,F12.6,A)')   '  "confidence": ', confidence, ','
    write(unit, '(A,F12.6,A)')   '  "M_spectral": ', M_spectral, ','
    write(unit, '(A,F12.6,A)')   '  "efficiency": ', efficiency, ','
    write(unit, '(A,F12.3)')     '  "runtime_ms": ', runtime_ms
    write(unit, '(A)')           '}'
  end subroutine write_json_output

  !> Monte-Carlo output (adds three percentile fields).
  subroutine write_json_output_mc(unit, T_panel, probs, trajectory, confidence, runtime_ms, &
                                  M_spectral, efficiency, T_mean, T_p05, T_p95)
    integer,  intent(in) :: unit
    real(WP), intent(in) :: T_panel, probs(5), confidence, runtime_ms, M_spectral, efficiency
    real(WP), intent(in) :: T_mean, T_p05, T_p95
    type(trajectory_type), intent(in) :: trajectory

    write(unit, '(A)')           '{'
    write(unit, '(A,F12.6,A)')   '  "T_panel": ',       T_panel, ','
    call write_probs_array(unit, probs)
    call write_trajectory(unit, trajectory)
    write(unit, '(A,F12.6,A)')   '  "confidence": ',    confidence, ','
    write(unit, '(A,F12.6,A)')   '  "M_spectral": ',    M_spectral, ','
    write(unit, '(A,F12.6,A)')   '  "efficiency": ',    efficiency, ','
    write(unit, '(A,F12.6,A)')   '  "T_panel_mean": ',  T_mean, ','
    write(unit, '(A,F12.6,A)')   '  "T_panel_p05": ',   T_p05, ','
    write(unit, '(A,F12.6,A)')   '  "T_panel_p95": ',   T_p95, ','
    write(unit, '(A,F12.3)')     '  "runtime_ms": ',    runtime_ms
    write(unit, '(A)')           '}'
  end subroutine write_json_output_mc

  subroutine write_probs_array(unit, probs)
    integer,  intent(in) :: unit
    real(WP), intent(in) :: probs(5)
    integer :: i
    character(len=16) :: buf
    write(unit, '(A)', advance='no') '  "probs": ['
    do i = 1, 5
      write(buf, '(F10.6)') probs(i)
      if (i < 5) then
        write(unit, '(A,A)', advance='no') trim(adjustl(buf)), ', '
      else
        write(unit, '(A)', advance='no') trim(adjustl(buf))
      end if
    end do
    write(unit, '(A)') '],'
  end subroutine write_probs_array

  subroutine write_trajectory(unit, t)
    integer, intent(in) :: unit
    type(trajectory_type), intent(in) :: t

    write(unit, '(A)')          '  "trajectory": {'
    write(unit, '(A,F12.6,A)')  '    "T_1h": ',  t%T_1h,  ','
    write(unit, '(A,F12.6,A)')  '    "T_2h": ',  t%T_2h,  ','
    write(unit, '(A,F12.6,A)')  '    "T_6h": ',  t%T_6h,  ','
    write(unit, '(A,F12.6,A)')  '    "CC_1h": ', t%CC_1h, ','
    write(unit, '(A,F12.6)')    '    "WS_1h": ', t%WS_1h
    write(unit, '(A)')          '  },'
  end subroutine write_trajectory

  subroutine parse_json_number(json, key, value, found)
    character(len=*), intent(in) :: json, key
    real(WP), intent(out) :: value
    logical,  intent(out) :: found

    integer :: p_key, p_colon, p_end, n, ios
    character(len=128) :: token

    found = .false.
    value = 0.0_WP

    p_key = index(json, '"'//trim(key)//'"')
    if (p_key <= 0) return

    p_colon = index(json(p_key:), ':')
    if (p_colon <= 0) return
    p_colon = p_colon + p_key - 1

    p_end = p_colon + 1
    do while (p_end <= len_trim(json))
      if (json(p_end:p_end) == ',' .or. json(p_end:p_end) == '}' .or. json(p_end:p_end) == ']') exit
      p_end = p_end + 1
    end do

    token = ''
    n = min(len(token), max(0, p_end - p_colon - 1))
    if (n <= 0) return
    token(1:n) = adjustl(json(p_colon+1:p_colon+n))

    read(token, *, iostat=ios) value
    if (ios /= 0) return
    found = .true.
  end subroutine parse_json_number

  subroutine parse_json_array(json, key, values, found)
    character(len=*), intent(in) :: json, key
    real(WP), allocatable, intent(out) :: values(:)
    logical,  intent(out) :: found

    integer :: p_key, p_lbr, p_rbr, count, i, ios, cpos
    character(len=4096) :: body, temp
    character(len=128) :: token

    found = .false.
    if (allocated(values)) deallocate(values)
    allocate(values(0))

    p_key = index(json, '"'//trim(key)//'"')
    if (p_key <= 0) return

    p_lbr = index(json(p_key:), '[')
    if (p_lbr <= 0) return
    p_lbr = p_lbr + p_key - 1

    p_rbr = index(json(p_lbr:), ']')
    if (p_rbr <= 0) return
    p_rbr = p_rbr + p_lbr - 1

    if (p_rbr <= p_lbr + 1) then
      found = .true.
      return
    end if

    body = ''
    body(1:min(len(body), p_rbr-p_lbr-1)) = json(p_lbr+1:p_rbr-1)
    temp = trim(body)

    count = 1
    do i = 1, len_trim(temp)
      if (temp(i:i) == ',') count = count + 1
    end do

    if (allocated(values)) deallocate(values)
    allocate(values(count))

    do i = 1, count
      cpos = index(temp, ',')
      if (cpos == 0) then
        token = adjustl(trim(temp))
      else
        token = adjustl(trim(temp(:cpos-1)))
      end if
      read(token, *, iostat=ios) values(i)
      if (ios /= 0) values(i) = 0.0_WP
      if (cpos == 0) exit
      temp = adjustl(temp(cpos+1:))
    end do

    found = .true.
  end subroutine parse_json_array

end module json_io_module
