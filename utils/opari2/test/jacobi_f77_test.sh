#!/bin/sh
set -e

opari_dir=`pwd`
test_dir="../test"
test_data_dir=/mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/data

CC="gcc -std=c99"
F77="gfortran"
INCDIR=.
OPARI2=/mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/../build-frontend/opari2 
OPARI2_CONFIG="/mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/../build-frontend/opari2-config --config=/mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/../build-frontend/opari2_config.dat"
LDFLAGS="-lm -L/mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/../build-frontend/.libs -lpomp"
FFLAGS="-g -O2"
OPENMP="-fopenmp"

rm -rf $test_dir/jacobi/F77
mkdir -p $test_dir/jacobi/F77
cp -r $test_data_dir/jacobi/F77/jacobi.f $test_dir/jacobi/F77
cp -r $test_data_dir/jacobi/F77/main.f $test_dir/jacobi/F77

cd $test_dir/jacobi/F77
#make instrument
$OPARI2 main.f 
$OPARI2 jacobi.f

#make compile
$F77 -I$INCDIR $OPENMP $FFLAGS -c jacobi.mod.F
$F77 -I$INCDIR $OPENMP $FFLAGS -c main.mod.F

#make build
/usr/bin/nm -B jacobi.mod.o main.mod.o | /bin/grep -E -i "T \.{0,1}_{0,2}pomp2_init_regions" | `$OPARI2_CONFIG --awk_cmd` -f `$OPARI2_CONFIG --awk_script`> pomp_init_file.c
$CC -I$INCDIR  -c pomp_init_file.c
$F77 $OPENMP $FFLAGS pomp_init_file.o jacobi.mod.o main.mod.o $LDFLAGS -o jacobi$EXEEXT

#make run
OMP_NUM_THREADS=4 ./jacobi$EXEEXT 2>jacobi_test.out

grep 0: jacobi_test.out > jacobi_test_0
grep 1: jacobi_test.out > jacobi_test_1
grep 2: jacobi_test.out > jacobi_test_2
grep 3: jacobi_test.out > jacobi_test_3

if diff $test_data_dir/jacobi/F77/jacobi_test_0.out jacobi_test_0 > /dev/null
  then
    true
  else
    echo "-------- ERROR: in program output --------"
    diff $test_data_dir/jacobi/F77/jacobi_test_0.out jacobi_test_0
    cd $opari_dir
    exit
fi
if diff $test_data_dir/jacobi/F77/jacobi_test_1.out jacobi_test_1 > /dev/null
  then
    true
  else
    echo "-------- ERROR: in program output --------"
    diff $test_data_dir/jacobi/F77/jacobi_test_1.out jacobi_test_1
    cd $opari_dir
    exit
fi
if diff $test_data_dir/jacobi/F77/jacobi_test_2.out jacobi_test_2 > /dev/null
  then
    true
  else
    echo "-------- ERROR: in program output --------"
    diff $test_data_dir/jacobi/F77/jacobi_test_2.out jacobi_test_2
    cd $opari_dir
    exit
fi
if diff $test_data_dir/jacobi/F77/jacobi_test_3.out jacobi_test_3 > /dev/null
  then
    true
  else
    echo "-------- ERROR: in program output --------"
    diff $test_data_dir/jacobi/F77/jacobi_test_3.out jacobi_test_3
    cd $opari_dir
    exit
fi

cd $opari_dir

