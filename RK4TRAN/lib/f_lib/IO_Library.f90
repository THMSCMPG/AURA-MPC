module IO_Library
    use, intrinsic :: ISO_C_BINDING
    implicit none

    integer, parameter :: N_SCEN_COLS = 19  ! lon..yaw, per independent/README.txt

    public :: load_independent_scenarios
    public :: next_run_number
    public :: zero_pad

contains

    ! Scan a directory for *.csv files, parse every non-comment / non-header
    ! row as a 19-column scenario vector, and return them stacked in
    ! scenarios(:, 1:N_SCEN_COLS). Any number of files / rows is supported;
    ! the array is allocated to the exact count discovered.
    subroutine load_independent_scenarios(dirpath, scenarios, n_found)
        character(len=*), intent(in)  :: dirpath
        real(c_double), allocatable, intent(out) :: scenarios(:,:)
        integer, intent(out) :: n_found

        character(len=4096) :: listing_file, line
        character(len=512)  :: fname
        integer :: u, lu, io, n_pass, pass
        real(c_double) :: row(N_SCEN_COLS)
        logical :: ok

        listing_file = ".independent_listing.tmp"

        ! Enumerate *.csv files in dirpath (sorted, one per line)
        call execute_command_line( &
            "ls " // trim(dirpath) // "/*.csv 2>/dev/null | sort > " // trim(listing_file), &
            wait=.true.)

        ! Pass 1: count valid data rows across all files
        ! Pass 2: fill the array
        do pass = 1, 2
            n_pass = 0

            open(newunit=u, file=trim(listing_file), status='old', action='read', iostat=io)
            if (io /= 0) then
                n_found = 0
                if (pass == 2) allocate(scenarios(0, N_SCEN_COLS))
                close(u, iostat=io)
                call execute_command_line("rm -f " // trim(listing_file), wait=.true.)
                return
            end if

            do
                read(u, '(A)', iostat=io) fname
                if (io /= 0) exit
                if (len_trim(fname) == 0) cycle

                open(newunit=lu, file=trim(fname), status='old', action='read', iostat=io)
                if (io /= 0) cycle

                do
                    read(lu, '(A)', iostat=io) line
                    if (io /= 0) exit
                    if (len_trim(line) == 0) cycle
                    if (line(1:1) == '#') cycle

                    call try_parse_row(line, row, ok)
                    if (.not. ok) cycle   ! header row or malformed line

                    n_pass = n_pass + 1
                    if (pass == 2) scenarios(n_pass, :) = row
                end do

                close(lu)
            end do
            close(u)

            if (pass == 1) then
                n_found = n_pass
                allocate(scenarios(max(n_found,0), N_SCEN_COLS))
            end if
        end do

        call execute_command_line("rm -f " // trim(listing_file), wait=.true.)
    end subroutine load_independent_scenarios

    ! Attempt to parse a comma-separated line into N_SCEN_COLS reals.
    ! Returns ok=.false. for header lines (non-numeric first field) or
    ! lines with the wrong column count.
    subroutine try_parse_row(line, row, ok)
        character(len=*), intent(in) :: line
        real(c_double), intent(out) :: row(N_SCEN_COLS)
        logical, intent(out) :: ok
        character(len=len(line)) :: work
        integer :: j, p, io
        real(c_double) :: v

        work = line
        ! Replace commas with spaces for list-directed read
        do j = 1, len_trim(work)
            if (work(j:j) == ',') work(j:j) = ' '
        end do

        read(work, *, iostat=io) (row(p), p = 1, N_SCEN_COLS)
        ok = (io == 0)
    end subroutine try_parse_row

    ! Find the lowest run number N (1..9999) such that
    ! "<prefix>NNNN<suffix>" does not already exist on disk, formatted with
    ! zero-padding determined by pad_width (e.g. pad_width=4 -> 0001).
    function next_run_number(prefix, suffix, pad_width) result(n)
        character(len=*), intent(in) :: prefix, suffix
        integer, intent(in) :: pad_width
        integer :: n
        character(len=512) :: candidate
        logical :: exists

        n = 1
        do
            candidate = trim(prefix) // trim(zero_pad(n, pad_width)) // trim(suffix)
            inquire(file=trim(candidate), exist=exists)
            if (.not. exists) return
            n = n + 1
            if (n > 9999) then
                n = 9999
                return
            end if
        end do
    end function next_run_number

    ! Zero-pad an integer to a fixed width, e.g. zero_pad(7,4) -> "0007"
    function zero_pad(n, width) result(s)
        integer, intent(in) :: n, width
        character(len=width) :: s
        character(len=16) :: fmt

        write(fmt, '(A,I0,A,I0,A)') "(I", width, ".", width, ")"
        write(s, fmt) n
    end function zero_pad

end module IO_Library
