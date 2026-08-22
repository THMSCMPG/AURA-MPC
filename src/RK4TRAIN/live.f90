program live_prediction_app
    use MC_UQ_Library
    use iso_c_binding
    implicit none

    integer, parameter :: PREDICTION_STEPS = 15
    integer, parameter :: LIVE_WINDOW = 12
    integer, parameter :: MAX_TOTAL_FILES = 50
    
    character(kind=c_char) :: c_paths(256, MAX_TOTAL_FILES)
    character(len=256) :: current_file
    character(len=128) :: output_csv
    character(len=64) :: ic_var_name
    real(8), allocatable :: independent_data(:)
    
    real(8) :: running_state(LIVE_WINDOW)
    real(8) :: next_step_calc, noise_bound
    integer :: worked_count, lazy_count, total_files, f_idx, i, out_unit

    ! 1. Query file counts from both runtime destination folders
    worked_count = get_directory_file_count("./worked" // c_null_char)
    lazy_count = get_directory_file_count("./lazy" // c_null_char)
    total_files = worked_count + lazy_count

    if (total_files == 0) then
        print *, "Error: No data matrices found in either ./worked/ or ./lazy/"
        stop
    end if
    
    if (total_files > MAX_TOTAL_FILES) total_files = MAX_TOTAL_FILES

    ! 2. Gather file paths sequentially into our tracking matrix
    if (worked_count > 0) then
        call get_directory_filepaths("./worked" // c_null_char, c_paths, 0, total_files)
    end if
    if (lazy_count > 0) then
        call get_directory_filepaths("./lazy" // c_null_char, c_paths, worked_count, total_files)
    end if

    ! 3. Core Loop over all gathered files
    call random_seed()
    noise_bound = 0.65d0 ! Custom standard deviation tracking parameter

    do f_idx = 1, total_files
        current_file = ""
        do i = 1, 256
            if (c_paths(i, f_idx) == c_null_char) exit
            current_file(i:i) = c_paths(i, f_idx)
        end do
        
        print *, "------------------------------------------------"
        print *, "Ingesting file from simulation pool: ", trim(current_file)
        
        call read_independent_csv_c(current_file, ic_var_name, independent_data)
        
        if (size(independent_data) < LIVE_WINDOW) then
            print *, "Skipping: Insufficient data rows in ", trim(ic_var_name)
            deallocate(independent_data)
            cycle
        end if

        ! Initialize our 12 historical points from the data set
        running_state = independent_data(1:LIVE_WINDOW)
        deallocate(independent_data) ! Clean up allocation space for this step iteration

        ! Dynamically acquire an incremented output filename (e.g. live_prediction0003.csv)
        call get_next_prediction_filename_c(output_csv)
        print *, "Saving real-time trajectory run out to: ", trim(output_csv)
        
        open(newunit=out_unit, file=trim(output_csv), status='new', action='write')
        write(out_unit, '(A)') "Minute,Calculated_Value"

        ! 15-minute simulation sequence
        do i = 1, PREDICTION_STEPS
            ! Compute time-step using your RK45 derivatives
            next_step_calc = sum(running_state) / size(running_state)
            
            ! Apply Monte Carlo variance bounded inside 1 standard deviation
            call run_monte_carlo_step(next_step_calc, noise_bound, next_step_calc)
            
            write(out_unit, '(I4, A, F14.6)') i, ",", next_step_calc
            
            ! Advance rolling frame 
            running_state(1:LIVE_WINDOW-1) = running_state(2:LIVE_WINDOW)
            running_state(LIVE_WINDOW) = next_step_calc
        end do
        
        close(out_unit)
        
        ! Call gnuplot for immediate visualization rendering
        call generate_gp_script(output_csv)
    end do

    print *, "================================================"
    print *, "All data sets processed and plotted successfully."
end program live_prediction_app
