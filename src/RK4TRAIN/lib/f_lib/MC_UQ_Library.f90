module MC_UQ_Library
    use, intrinsic :: ISO_C_BINDING
    implicit none

    real(c_double), parameter :: TWO_PI_MC = 6.283185307179586_c_double

    public :: box_muller_sample
    public :: mc_sigma_bounds
    public :: rk45_mc_step
    public :: live_mc_step
    public :: read_independent_csv, free_ic_data, get_next_prediction_filename
    public :: run_monte_carlo_step, generate_gp_script

    ! Interoperable C Layout matching csv_parser.h
    type, bind(C) :: InitialConditionC
        character(kind=c_char) :: var_name(64)
        type(c_ptr) :: values
        integer(c_int) :: num_elements
    end type InitialConditionC

    interface
        function get_directory_file_count(dir_path) bind(c, name="get_directory_file_count")
            import :: c_int, c_char
            character(kind=c_char), intent(in) :: dir_path(*)
            integer(c_int) :: get_directory_file_count
        end function get_directory_file_count

        subroutine get_directory_filepaths(dir_path, paths, start_idx, max_files) bind(c, name="get_directory_filepaths")
            import :: c_char, c_int
            character(kind=c_char), intent(in) :: dir_path(*)
            character(kind=c_char), intent(out) :: paths(256, *)
            integer(c_int), value :: start_idx, max_files
        end subroutine get_directory_filepaths

        subroutine read_independent_csv(filepath, ic_data) bind(c, name="read_independent_csv")
            import :: c_char, InitialConditionC
            character(kind=c_char), intent(in) :: filepath(*)
            type(InitialConditionC), intent(inout) :: ic_data
        end subroutine read_independent_csv

        subroutine free_ic_data(ic_data) bind(c, name="free_ic_data")
            import :: InitialConditionC
            type(InitialConditionC), intent(inout) :: ic_data
        end subroutine free_ic_data

        subroutine get_next_prediction_filename(filename, max_len) bind(c, name="get_next_prediction_filename")
            import :: c_char, c_size_t
            character(kind=c_char), intent(out) :: filename(*)
            integer(c_size_t), value :: max_len
        end subroutine get_next_prediction_filename
    end interface

contains

    !  Returns a single standard-normal deviate z ~ N(0,1).
    function box_muller_sample(which) result(z)
        integer, intent(in) :: which  ! 1 or 2
        real(c_double) :: z
        real(c_double) :: r1, r2

        call random_number(r1)
        call random_number(r2)
        ! Guard log(0)
        r1 = max(r1, 2.22e-16_c_double)
        r2 = max(r2, 2.22e-16_c_double)

        if (which == 1) then
            z = sqrt(-2.0_c_double * log(r1)) * cos(TWO_PI_MC * r2)
        else
            z = sqrt(-2.0_c_double * log(r1)) * sin(TWO_PI_MC * r2)
        end if
    end function box_muller_sample

    subroutine mc_sigma_bounds(DE_func, t, y_in, sigma_frac, N_MC, y_mean, y_sigma)
        interface
            function DE_func(t, y) result(dydt)
                use, intrinsic :: iso_c_binding
                real(c_double), intent(in) :: t
                real(c_double), dimension(:), intent(in) :: y
                real(c_double), dimension(size(y)) :: dydt
            end function DE_func
        end interface

        real(c_double), intent(in)  :: t
        real(c_double), intent(in)  :: y_in(:)
        real(c_double), intent(in)  :: sigma_frac
        integer,        intent(in)  :: N_MC
        real(c_double), intent(out) :: y_mean(size(y_in))
        real(c_double), intent(out) :: y_sigma(size(y_in))

        integer  :: mc, j, ndim
        real(c_double), allocatable :: y_pert(:), dydt_s(:)
        real(c_double), allocatable :: sum1(:), sum2(:)
        real(c_double) :: z

        ndim = size(y_in)
        allocate(y_pert(ndim), dydt_s(ndim), sum1(ndim), sum2(ndim))

        sum1 = 0.0_c_double
        sum2 = 0.0_c_double

        call random_seed()

        do mc = 1, N_MC
            ! Perturb each dimension independently
            do j = 1, ndim
                z = box_muller_sample(1)
                y_pert(j) = y_in(j) + sigma_frac * abs(y_in(j)) * z
            end do

            dydt_s = DE_func(t, y_pert)
            sum1   = sum1 + dydt_s
            sum2   = sum2 + dydt_s * dydt_s
        end do

        y_mean  = sum1 / real(N_MC, c_double)
        y_sigma = sqrt(max(0.0_c_double, sum2 / real(N_MC, c_double) - y_mean**2))

        deallocate(y_pert, dydt_s, sum1, sum2)
    end subroutine mc_sigma_bounds

    !  One adaptive RK45 step (Dormand-Prince / Cash-Karp variant) paired with
    !  MC sigma bounds.
    subroutine rk45_mc_step(DE_func, t, y, dt_try, tol, sigma_frac, N_MC, &
                             y_next, y_sigma, dt_used, dt_next)
        interface
            function DE_func(t, y) result(dydt)
                use, intrinsic :: iso_c_binding
                real(c_double), intent(in) :: t
                real(c_double), dimension(:), intent(in) :: y
                real(c_double), dimension(size(y)) :: dydt
            end function DE_func
        end interface

        real(c_double), intent(in)  :: t, dt_try, tol, sigma_frac
        real(c_double), intent(in)  :: y(:)
        integer,        intent(in)  :: N_MC
        real(c_double), intent(out) :: y_next(size(y)), y_sigma(size(y))
        real(c_double), intent(out) :: dt_used, dt_next

        ! Fehlberg 4th-order weights (CH) and 5th-order weights (CT)
        real(c_double), parameter :: CH(6) = [ &
            25.0_c_double/216.0_c_double,    &
            0.0_c_double,                    &
            1408.0_c_double/2565.0_c_double, &
            2197.0_c_double/4104.0_c_double, &
            -0.2_c_double,                   &
            0.0_c_double ]

        real(c_double), parameter :: CT(6) = [ &
            16.0_c_double/135.0_c_double,     &
            0.0_c_double,                     &
            6656.0_c_double/12825.0_c_double, &
            28561.0_c_double/56430.0_c_double,&
            -9.0_c_double/50.0_c_double,      &
            2.0_c_double/55.0_c_double ]

        integer  :: ndim
        real(c_double), allocatable :: k1(:),k2(:),k3(:),k4(:),k5(:),k6(:)
        real(c_double), allocatable :: y4(:), y5(:), y_mean_dummy(:)
        real(c_double) :: dt, max_err, s

        ndim = size(y)
        allocate(k1(ndim),k2(ndim),k3(ndim),k4(ndim),k5(ndim),k6(ndim))
        allocate(y4(ndim), y5(ndim), y_mean_dummy(ndim))

        dt = dt_try

        ! RK45 stage evaluations (Fehlberg coefficients, same as RK_Solver_Library)
        k1 = dt * DE_func(t, y)
        k2 = dt * DE_func(t + dt/4.0_c_double, &
                y + k1/4.0_c_double)
        k3 = dt * DE_func(t + dt*3.0_c_double/8.0_c_double, &
                y + k1*3.0_c_double/32.0_c_double + k2*9.0_c_double/32.0_c_double)
        k4 = dt * DE_func(t + dt*12.0_c_double/13.0_c_double, &
                y + k1*1932.0_c_double/2197.0_c_double &
                  - k2*7200.0_c_double/2197.0_c_double &
                  + k3*7296.0_c_double/2197.0_c_double)
        k5 = dt * DE_func(t + dt, &
                y + k1*439.0_c_double/216.0_c_double &
                  - k2*8.0_c_double &
                  + k3*3680.0_c_double/513.0_c_double &
                  - k4*845.0_c_double/4104.0_c_double)
        k6 = dt * DE_func(t + dt/2.0_c_double, &
                y - k1*8.0_c_double/27.0_c_double &
                  + k2*2.0_c_double &
                  - k3*3544.0_c_double/2565.0_c_double &
                  + k4*1859.0_c_double/4104.0_c_double &
                  - k5*11.0_c_double/40.0_c_double)

        y4 = y + CH(1)*k1 + CH(2)*k2 + CH(3)*k3 + CH(4)*k4 + CH(5)*k5 + CH(6)*k6
        y5 = y + CT(1)*k1 + CT(2)*k2 + CT(3)*k3 + CT(4)*k4 + CT(5)*k5 + CT(6)*k6

        max_err = maxval(abs(y5 - y4))

        ! Step-size controller (standard 0.84 safety factor)
        if (max_err == 0.0_c_double) then
            s = 2.0_c_double
        else
            s = 0.84_c_double * (tol / max_err)**0.25_c_double
        end if

        y_next   = y5
        dt_used  = dt
        dt_next  = dt * max(0.1_c_double, min(4.0_c_double, s))

        ! MC sigma estimation at the accepted state
        call mc_sigma_bounds(DE_func, t + dt, y5, sigma_frac, N_MC, &
                             y_mean_dummy, y_sigma)

        deallocate(k1,k2,k3,k4,k5,k6,y4,y5,y_mean_dummy)
    end subroutine rk45_mc_step

    subroutine live_mc_step(DE_func, t, y, dt_try, tol, sigma_frac, N_MC, &
                            lo_bound, hi_bound, y_settled, dt_used, dt_next)
        interface
            function DE_func(t, y) result(dydt)
                use, intrinsic :: iso_c_binding
                real(c_double), intent(in) :: t
                real(c_double), dimension(:), intent(in) :: y
                real(c_double), dimension(size(y)) :: dydt
            end function DE_func
        end interface

        real(c_double), intent(in)  :: t, dt_try, tol, sigma_frac
        real(c_double), intent(in)  :: y(:)
        integer,        intent(in)  :: N_MC
        real(c_double), intent(in)  :: lo_bound(:), hi_bound(:)
        real(c_double), intent(out) :: y_settled(size(y))
        real(c_double), intent(out) :: dt_used, dt_next

        integer :: ndim, j
        real(c_double), allocatable :: y_next(:), y_sigma(:)
        real(c_double) :: u, z_unit

        ndim = size(y)
        allocate(y_next(ndim), y_sigma(ndim))

        ! Deterministic RK45 step + uncertainty envelope
        call rk45_mc_step(DE_func, t, y, dt_try, tol, sigma_frac, N_MC, &
                          y_next, y_sigma, dt_used, dt_next)

        ! Stochastic settlement: draw position within +-1 sigma
        do j = 1, ndim
            call random_number(u)
            ! Map u in [0,1] to z in [-1, +1] linearly (uniform scatter within 1 sigma)
            ! This is simpler and more transparent than an inverse-normal for 1-sigma-only
            z_unit = 2.0_c_double * u - 1.0_c_double   ! z in [-1, +1]
            y_settled(j) = y_next(j) + y_sigma(j) * z_unit

            ! Enforce physical bounds
            y_settled(j) = max(lo_bound(j), min(hi_bound(j), y_settled(j)))
        end do

        deallocate(y_next, y_sigma)
    end subroutine live_mc_step

    ! Wrapper to read data into Fortran-usable native pointers
    subroutine read_independent_csv_c(filepath, out_name, out_array)
        character(len=*), intent(in) :: filepath
        character(len=64), intent(out) :: out_name
        real(c_double), allocatable, intent(out) :: out_array(:)
        
        type(InitialConditionC) :: c_struct
        real(c_double), pointer :: f_ptr(:)
        integer :: i
        
        ! Append null-terminator for C string standard
        call read_independent_csv(trim(filepath) // c_null_char, c_struct)
        
        if (c_struct%num_elements > 0) then
            allocate(out_array(c_struct%num_elements))
            call c_f_pointer(c_struct%values, f_ptr, [c_struct%num_elements])
            out_array = f_ptr ! Deep copy array elements to Fortran space
            
            ! Convert char array back to standard string
            out_name = ""
            do i = 1, 64
                if (c_struct%var_name(i) == c_null_char) exit
                out_name(i:i) = c_struct%var_name(i)
            end do
            
            call free_ic_data(c_struct) ! Free heap allocation on the C side safely
        else
            allocate(out_array(0))
            out_name = "EMPTY"
        end if
    end subroutine read_independent_csv_c

    subroutine get_next_prediction_filename_c(fortran_str)
        character(len=128), intent(out) :: fortran_str
        character(kind=c_char) :: c_str(128)
        integer :: i
        
        call get_next_prediction_filename(c_str, size(c_str, kind=c_size_t))
        
        fortran_str = ""
        do i = 1, 128
            if (c_str(i) == c_null_char) exit
            fortran_str(i:i) = c_str(i)
        end do
    end subroutine get_next_prediction_filename_c

    ! Noisy RK45 MC mapping step bounded inside 1 Standard Deviation
    subroutine run_monte_carlo_step(base_val, std_dev, final_val)
        real(c_double), intent(in) :: base_val, std_dev
        real(c_double), intent(out) :: final_val
        real(c_double) :: rand_norm
        
        call random_number(rand_norm)
        rand_norm = (rand_norm * 2.0_c_double) - 1.0_c_double ! Map to [-1.0, 1.0] 
        final_val = base_val + (rand_norm * std_dev)         ! Scale to match 1 Std Dev 
    end subroutine run_monte_carlo_step

    ! Automate Gnuplot script creation and terminal generation 
    subroutine generate_gp_script(csv_file)
        character(len=*), intent(in) :: csv_file
        integer :: gp_unit
        character(len=140) :: script_name
        
        script_name = trim(csv_file) // ".gp"
        open(newunit=gp_unit, file=trim(script_name), status='unknown')
        
        write(gp_unit, '(A)') "set datafile separator ','"
        write(gp_unit, '(A)') "set term pngcairo size 800,600 font 'Arial,10'"
        write(gp_unit, '(A)') "set output '" // trim(csv_file) // ".png'"
        write(gp_unit, '(A)') "set title '15-Minute Dynamic Trajectory (With Random Noise Walk)'"
        write(gp_unit, '(A)') "set xlabel 'Simulation Time Increments'"
        write(gp_unit, '(A)') "set ylabel 'State Vector Resolution'"
        write(gp_unit, '(A)') "set grid"
        write(gp_unit, '(A)') "plot '" // trim(csv_file) // "' using 1:2 with lines lw 2 title 'Predicted State Track'"
        
        close(gp_unit)
        call execute_command_line("gnuplot " // trim(script_name), wait=.true.)
    end subroutine generate_gp_script

end module MC_UQ_Library
