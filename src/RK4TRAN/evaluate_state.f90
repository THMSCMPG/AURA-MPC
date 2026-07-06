module pv_env_state_eval
    use, intrinsic :: ISO_C_BINDING
    implicit none

    real(c_double) :: env_lon, env_lat, env_alt
    real(c_double) :: env_min, env_hour, env_day, env_month, env_year
    real(c_double) :: env_T_amb
    real(c_double) :: env_wind
    real(c_double) :: env_winddir
    real(c_double) :: env_humid
    real(c_double) :: env_irr
    real(c_double) :: env_cloud
    real(c_double) :: env_press
    real(c_double) :: env_pv_h
    real(c_double) :: env_pitch
    real(c_double) :: env_roll
    real(c_double) :: env_yaw
    real(c_double) :: env_G_eff
end module pv_env_state_eval

module pv_ode_module_eval
    use, intrinsic :: ISO_C_BINDING
    use pv_env_state_eval
    implicit none

    real(c_double), parameter :: PI      = 3.14159265358979_c_double
    real(c_double), parameter :: D2R     = PI / 180.0_c_double
    real(c_double), parameter :: ETA_REF = 0.20_c_double
    real(c_double), parameter :: BETA_T  = 0.004_c_double
    real(c_double), parameter :: T_STC   = 298.15_c_double
    real(c_double), parameter :: ALPHA   = 0.90_c_double
    real(c_double), parameter :: C_TH    = 10000.0_c_double

contains

    subroutine compute_G_eff()
        real(c_double) :: decl, H, sin_elev, cos_aoi
        real(c_double) :: az_sun, az_panel

        decl = 23.45_c_double * sin(D2R * (284.0_c_double + env_day) * 360.0_c_double / 365.0_c_double)
        H = 15.0_c_double * (env_hour + env_min / 60.0_c_double - 12.0_c_double)

        sin_elev = sin(D2R * env_lat) * sin(D2R * decl) &
                 + cos(D2R * env_lat) * cos(D2R * decl) * cos(D2R * H)
        sin_elev = max(0.0_c_double, sin_elev)

        az_sun = atan2(sin(D2R * H), cos(D2R * H) * sin(D2R * env_lat) - tan(D2R * decl) * cos(D2R * env_lat)) / D2R
        az_panel = env_yaw

        cos_aoi = sin_elev * cos(D2R * env_pitch) &
                + sqrt(max(0.0_c_double, 1.0_c_double - sin_elev**2)) &
                * cos(D2R * (az_sun - az_panel)) * sin(D2R * env_pitch)
        cos_aoi = max(0.0_c_double, cos_aoi)
        cos_aoi = cos_aoi * cos(D2R * env_roll)

        env_G_eff = env_irr * cos_aoi * (1.0_c_double - env_cloud)
    end subroutine compute_G_eff

    function pv_ode(t, y) result(dydt)
        real(c_double), intent(in) :: t
        real(c_double), dimension(:), intent(in) :: y
        real(c_double), dimension(size(y)) :: dydt
        real(c_double) :: h_conv, dTdt

        h_conv = (5.7_c_double + 3.8_c_double * env_wind) &
               * min(1.2_c_double, 0.5_c_double + env_pv_h / 4.0_c_double)
        dTdt = (env_G_eff * ALPHA - env_G_eff * y(2) - h_conv * (y(1) - env_T_amb)) / C_TH
        dydt(1) = dTdt
        dydt(2) = -ETA_REF * BETA_T * dTdt
    end function pv_ode
end module pv_ode_module_eval

program evaluate_state
    use RK4TRAN
    use pv_env_state_eval
    use pv_ode_module_eval
    use, intrinsic :: ISO_C_BINDING
    implicit none

    integer, parameter :: EXPECTED_ARGS = 19
    real(c_double), parameter :: TOL = 1.0e-5_c_double
    character(len=64) :: arg
    character(len=512) :: json_path
    character(len=128) :: tmpfile
    integer :: c0, c1, cr, ios, i, suffix, json_unit, env_status
    real(c_double) :: ic(2), runtime_ms, r

    if (command_argument_count() /= EXPECTED_ARGS) then
        write(0, '(A,I0)') 'ERROR: expected ', EXPECTED_ARGS
        stop 1
    end if

    call read_arg(1, env_lon)
    call read_arg(2, env_lat)
    call read_arg(3, env_alt)
    call read_arg(4, env_min)
    call read_arg(5, env_hour)
    call read_arg(6, env_day)
    call read_arg(7, env_month)
    call read_arg(8, env_year)
    call read_arg(9, env_T_amb)
    call read_arg(10, env_wind)
    call read_arg(11, env_winddir)
    call read_arg(12, env_humid)
    call read_arg(13, env_irr)
    call read_arg(14, env_cloud)
    call read_arg(15, env_press)
    call read_arg(16, env_pv_h)
    call read_arg(17, env_pitch)
    call read_arg(18, env_roll)
    call read_arg(19, env_yaw)

    call compute_G_eff()
    ic(1) = env_T_amb
    ic(2) = ETA_REF * (1.0_c_double - BETA_T * (env_T_amb - T_STC))

    call random_seed()
    call random_number(r)
    suffix = int(r * 1000000.0_c_double)
    write(tmpfile, '(".rk45_eval_", I0, ".csv")') suffix

    call system_clock(c0, cr)
    call RK45(pv_ode, ic, 0.0_c_double, 600.0_c_double, TOL, trim(tmpfile))
    call read_last_row(trim(tmpfile), ic)
    call execute_command_line("rm -f " // trim(tmpfile), wait=.true.)
    call system_clock(c1)

    if (cr > 0) then
        runtime_ms = 1000.0_c_double * real(c1 - c0, c_double) / real(cr, c_double)
    else
        runtime_ms = 0.0_c_double
    end if

    call get_environment_variable("AURA_RK4_JSON_PATH", json_path, status=env_status)
    if (env_status == 0 .and. len_trim(json_path) > 0) then
        open(newunit=json_unit, file=trim(json_path), status='replace', action='write', iostat=ios)
        if (ios == 0) then
            call write_json_record(json_unit, ic, runtime_ms)
            flush(json_unit)
            close(json_unit)
        end if
    end if

    call write_json_record(6, ic, runtime_ms)
    flush(6)

contains

    subroutine read_arg(idx, value)
        integer, intent(in) :: idx
        real(c_double), intent(out) :: value
        character(len=64) :: raw
        integer :: local_ios

        call get_command_argument(idx, raw)
        read(raw, *, iostat=local_ios) value
        if (local_ios /= 0) then
            write(0, '(A,I0)') 'ERROR: malformed numeric argument at position ', idx
            stop 1
        end if
    end subroutine read_arg

    subroutine read_last_row(filename, y_final)
        character(len=*), intent(in) :: filename
        real(c_double), intent(out) :: y_final(:)
        integer :: unit, io
        real(c_double) :: t_tmp, y1_tmp, y2_tmp
        real(c_double) :: y1_last, y2_last

        open(newunit=unit, file=filename, status='old', action='read', iostat=ios)
        if (ios /= 0) then
            write(0, '(A)') 'ERROR: unable to read RK45 output'
            stop 1
        end if

        do
            read(unit, *, iostat=io) t_tmp, y1_tmp, y2_tmp
            if (io /= 0) exit
            y1_last = y1_tmp
            y2_last = y2_tmp
        end do
        close(unit)
        y_final(1) = y1_last
        y_final(2) = y2_last
    end subroutine read_last_row

    subroutine write_json_record(unit_no, y_final, elapsed_ms)
        integer, intent(in) :: unit_no
        real(c_double), intent(in) :: y_final(:)
        real(c_double), intent(in) :: elapsed_ms

        write(unit_no, '(A)') '{'
        write(unit_no, '(A,F12.6,A)') '  "T_operating": ', y_final(1), ','
        write(unit_no, '(A,F12.6,A)') '  "eta": ', y_final(2), ','
        write(unit_no, '(A,F12.6,A)') '  "G_eff": ', env_G_eff, ','
        write(unit_no, '(A,F12.3)') '  "runtime_ms": ', elapsed_ms
        write(unit_no, '(A)') '}'
    end subroutine write_json_record
end program evaluate_state
