module RK4TRAN
    use RK_Solver_Library
    use Plot_Library
    use MC_UQ_Library
    use IO_Library

    implicit none

    ! All symbols are inherited via USE association.
    ! Explicit re-export list keeps the interface surface visible in IDEs.
    public :: RK4, RK45, write_csv_RK   ! from RK_Solver_Library
    public :: plt_RK                     ! from Plot_Library
    public :: box_muller_sample          ! from MC_UQ_Library
    public :: mc_sigma_bounds            ! from MC_UQ_Library
    public :: rk45_mc_step               ! from MC_UQ_Library
    public :: live_mc_step               ! from MC_UQ_Library
    public :: load_independent_scenarios ! from IO_Library
    public :: next_run_number            ! from IO_Library
    public :: zero_pad                   ! from IO_Library

end module RK4TRAN
