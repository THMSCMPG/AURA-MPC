module Plot_Library
        use, intrinsic :: ISO_C_BINDING
        implicit none

        ! declare C interfaces for compiler
        interface
                function c_open_gnuplot_pipe(command) result(stream) bind(C, name="c_open_gnuplot_pipe")
                        use, intrinsic :: ISO_C_BINDING
                        type(C_PTR) :: stream
                        character(kind=c_char), intent(in) :: command(*)
                end function c_open_gnuplot_pipe

                subroutine c_write_gnuplot_pipe(stream, str) bind(C, name="c_write_gnuplot_pipe")
                        use, intrinsic :: ISO_C_BINDING
                        type(c_ptr), value :: stream
                        character(kind=c_char), intent(in) :: str(*)
                end subroutine c_write_gnuplot_pipe

                subroutine c_close_gnuplot_pipe(stream) bind(C, name="c_close_gnuplot_pipe")
                        use, intrinsic :: ISO_C_BINDING
                        type(c_ptr), value :: stream
                end subroutine c_close_gnuplot_pipe
        end interface

contains
        subroutine plt_RK(template_num, data_file, color, linechoice)
        INTEGER, INTENT(IN) :: template_num
        CHARACTER(LEN=*), INTENT(IN) :: data_file
        CHARACTER(LEN=*), INTENT(IN) :: color
        CHARACTER(LEN=*), INTENT(IN) :: linechoice

        TYPE(C_PTR) :: gp_pipe
        CHARACTER(LEN=256) :: template_path
        CHARACTER(LEN=512) :: line_buffer
        INTEGER :: file_unit, io_status

        WRITE(template_path, '(A, I1, A)') "lib/gp_lib/template", template_num, ".gp"

        gp_pipe = c_open_gnuplot_pipe("gnuplot" // C_NULL_CHAR)
        
        IF (.NOT. C_ASSOCIATED(gp_pipe)) THEN
            PRINT *, "Error: Could not open connection pipe to Gnuplot."
            RETURN
        END IF

        OPEN(NEWUNIT=file_unit, FILE=TRIM(template_path), STATUS='OLD', ACTION='READ', IOSTAT=io_status)
        IF (io_status /= 0) THEN
            PRINT *, "Error: Template file not found: ", TRIM(template_path)
            CALL c_close_gnuplot_pipe(gp_pipe)
            RETURN
        END IF

        DO
            READ(file_unit, '(A)', IOSTAT=io_status) line_buffer
            IF (io_status /= 0) EXIT ! Reached End-of-File safely

            ! Substitute configuration details across templates
            CALL replace_string(line_buffer, "__DATAFILE__", TRIM(data_file))
            CALL replace_string(line_buffer, "__COLOR__", TRIM(color))
            CALL replace_string(line_buffer, "__STYLE__", TRIM(linechoice))

            ! Ship instructions down to C execution matrix
            CALL c_write_gnuplot_pipe(gp_pipe, TRIM(line_buffer) // NEW_LINE('A') // C_NULL_CHAR)
        END DO

        CLOSE(file_unit)
        
        CALL c_close_gnuplot_pipe(gp_pipe)

    END SUBROUTINE plt_RK

    SUBROUTINE replace_string(str, target, replacement)
        CHARACTER(LEN=*), INTENT(INOUT) :: str
        CHARACTER(LEN=*), INTENT(IN) :: target
        CHARACTER(LEN=*), INTENT(IN) :: replacement
        INTEGER :: pos, t_len, r_len

        t_len = LEN(target)
        r_len = LEN(replacement)
        pos = INDEX(str, target)

        DO WHILE (pos > 0)
            str = str(1:pos-1) // TRIM(replacement) // str(pos+t_len:)
            pos = INDEX(str, target)
        END DO
    END SUBROUTINE replace_string

END MODULE Plot_Library
