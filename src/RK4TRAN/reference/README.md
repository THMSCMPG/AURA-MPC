# simv0 src layer

Lean rewritten source subset for simv0-only functionality.

## Build
`gfortran -O2 -o simv0_src core/*.f90 environment/*.f90 solvers/*.f90 utils/*.f90 commands/*.f90 main.f90`
