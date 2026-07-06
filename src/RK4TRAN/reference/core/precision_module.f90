! Ported from reference/original/core/precision_module.f90 on Day 2, stripped of ML/tuning/RL dependencies.
!
! SimV0 trims this to the essentials: a working-precision kind (WP = REAL64)
! plus the two integer kinds used by the solvers. Everything else in the
! reference module (machine-epsilon parameters, tolerances, version strings)
! is unused by the V0 physics path and has been removed to keep the module
! dependency graph minimal.

module precision_module
  use, intrinsic :: iso_fortran_env, only: INT32, INT64, REAL64
  implicit none
  private

  integer, parameter, public :: I4 = INT32
  integer, parameter, public :: I8 = INT64
  integer, parameter, public :: WP = REAL64
end module precision_module
