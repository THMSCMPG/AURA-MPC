module RK_Solver_Library
        use, intrinsic :: ISO_C_BINDING
        implicit none

        contains

                subroutine RK45(DE_func, initial_conditions, t_start, t_end, tol, filename)
                        interface
                                function DE_func(t, y) result(dydt)
                                        use, intrinsic :: iso_c_binding
                                        real(c_double), intent(in) :: t
                                        real(c_double), dimension(:), intent(in) :: y
                                        real(c_double), dimension(size(y)) :: dydt
                                end function DE_func
                        end interface
                        real(c_double), dimension(:), intent(in) :: initial_conditions
                        real(c_double), intent(in) :: t_start, t_end, tol
                        character(len=*), intent(in) :: filename

                        ! Fehlberg coefficients (4th-order accumulation)
                        real(c_double), parameter :: CH(6) = (/ &
                                25.0_c_double/216.0_c_double,   &
                                0.0_c_double,                   &
                                1408.0_c_double/2565.0_c_double,&
                                2197.0_c_double/4104.0_c_double,&
                                -0.2_c_double,                  &
                                0.0_c_double /)

                        ! Fehlberg coefficients (5th-order accumulation)
                        real(c_double), parameter :: CT(6) = (/ &
                                16.0_c_double/135.0_c_double,    &
                                0.0_c_double,                    &
                                6656.0_c_double/12825.0_c_double,&
                                28561.0_c_double/56430.0_c_double,&
                                -9.0_c_double/50.0_c_double,    &
                                2.0_c_double/55.0_c_double /)

                        real(c_double), dimension(size(initial_conditions)) :: y, y4, y5
                        real(c_double), dimension(size(initial_conditions)) :: k1, k2, k3, k4, k5, k6
                        real(c_double) :: t, dt, max_err, s
                        integer :: file_unit

                        t  = t_start
                        dt = 0.01_c_double
                        y  = initial_conditions

                        open(newunit=file_unit, file=filename, status='REPLACE', action='WRITE')
                        write(file_unit, '(F12.6, 100( ", ", F12.6))') t, y

                        do while (t < t_end)
                                if (t + dt > t_end) dt = t_end - t

                                k1 = dt * DE_func(t, y)
                                k2 = dt * DE_func(t + dt/4.0_c_double, &
                                        y + k1/4.0_c_double)
                                k3 = dt * DE_func(t + dt*3.0_c_double/8.0_c_double, &
                                        y + k1*3.0_c_double/32.0_c_double &
                                        + k2*9.0_c_double/32.0_c_double)
                                k4 = dt * DE_func(t + dt*12.0_c_double/13.0_c_double, &
                                        y + k1*1932.0_c_double/2197.0_c_double &
                                        - k2*7200.0_c_double/2197.0_c_double &
                                        + k3*7296.0_c_double/2197.0_c_double)  ! FIX ②: was 7296.0c_double
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

                                if (max_err <= tol) then
                                        t = t + dt
                                        y = y5
                                        write(file_unit, '(F12.6, 100( ", ", F12.6))') t, y
                                end if

                                if (max_err == 0.0_c_double) then
                                        s = 2.0_c_double
                                else
                                        s = 0.84_c_double * (tol / max_err)**0.25_c_double
                                end if

                                dt = dt * max(0.1_c_double, min(4.0_c_double, s))
                        end do

                        close(file_unit)
                end subroutine RK45

                subroutine RK4(DE_func, timestep, initial_conditions, end_condition, output_matrix)
                        interface
                                function DE_func(t, y) result(dydt)
                                        use, intrinsic :: iso_c_binding
                                        real(c_double), intent(in) :: t
                                        real(c_double), dimension(:), intent(in) :: y
                                        real(c_double), dimension(size(y)) :: dydt
                                end function DE_func
                        end interface
                        real(c_double), intent(in) :: timestep, end_condition
                        real(c_double), dimension(:), intent(in) :: initial_conditions
                        real(c_double), allocatable, intent(out) :: output_matrix(:,:)

                        real(c_double) :: t
                        real(c_double), dimension(size(initial_conditions)) :: y, k1, k2, k3, k4
                        integer :: steps, i

                        steps = int(end_condition / timestep)
                        allocate(output_matrix(steps+1, size(initial_conditions) + 1))  ! FIX ③: missing )

                        t = 0.0
                        y = initial_conditions
                        output_matrix(1, 1)  = t
                        output_matrix(1, 2:) = y

                        do i = 1, steps
                                k1 = timestep * DE_func(t, y)
                                k2 = timestep * DE_func(t + timestep/2.0, y + k1/2.0)
                                k3 = timestep * DE_func(t + timestep/2.0, y + k2/2.0)
                                k4 = timestep * DE_func(t + timestep,     y + k3)

                                y = y + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
                                t = t + timestep

                                output_matrix(i+1, 1)  = t
                                output_matrix(i+1, 2:) = y
                        end do
                end subroutine RK4

                subroutine write_csv_RK(matrix, filename)
                        real, intent(in) :: matrix(:,:)
                        character(len=*), intent(in) :: filename
                        integer :: file_unit, i, j

                        open(newunit=file_unit, file=filename, status='replace', action='write')

                        do i = 1, size(matrix, 1)
                                do j = 1, size(matrix, 2)
                                        if (j == size(matrix, 2)) then
                                                write(file_unit, '(f12.6)') matrix(i,j)
                                        else
                                                write(file_unit, '(f12.6, ", ")', advance='no') matrix(i,j)
                                        end if
                                end do
                        end do

                        close(file_unit)
                end subroutine write_csv_RK

end module RK_Solver_Library
