#!/bin/bash
set -e

TARGET="${1:-all}"   # default: build everything

echo "=== AURA_MFP build system ==="

mkdir -p build work lazy independent

echo "[1/8] Compiling C components..."
gcc -c lib/c_lib/sys_pipes.c -fPIC -o build/sys_pipes.o
gcc -c lib/c_lib/csv_parser.c -fPIC -o build/csv_parser.o

echo "[2/8] Compiling RK_Solver_Library..."
gfortran -c lib/f_lib/RK_Solver_Library.f90 \
    -J build -I build \
    -o build/RK_Solver_Library.o

echo "[3/8] Compiling Plot_Library..."
gfortran -c lib/f_lib/Plot_Library.f90 \
    -J build -I build \
    -o build/Plot_Library.o

echo "[4/8] Compiling MC_UQ_Library..."
gfortran -c lib/f_lib/MC_UQ_Library.f90 \
    -J build -I build \
    -o build/MC_UQ_Library.o

echo "[5/8] Compiling IO_Library..."
gfortran -c lib/f_lib/IO_Library.f90 \
    -J build -I build \
    -o build/IO_Library.o

echo "[6/8] Compiling RK4TRAN gateway..."
gfortran -c lib/RK4TRAN.f90 \
    -J build -I build \
    -o build/RK4TRAN.o

echo "[7/8] Compiling + linking main (data generator)..."
gfortran -c main.f90 -J build -I build -o build/main.o
gfortran build/main.o \
    build/sys_pipes.o \
    build/csv_parser.o \
    build/RK_Solver_Library.o \
    build/Plot_Library.o \
    build/MC_UQ_Library.o \
    build/IO_Library.o \
    build/RK4TRAN.o \
    -J build -I build \
    -o main -O3

gfortran -c live.f90 -J build -I build -o build/live.o
gfortran build/live.o \
    build/sys_pipes.o \
    build/csv_parser.o \
    build/RK_Solver_Library.o \
    build/Plot_Library.o \
    build/MC_UQ_Library.o \
    build/IO_Library.o \
    build/RK4TRAN.o \
    -J build -I build \
    -o live -O3

echo "[8/8] Done.  Run with:  ./main or ./live"
