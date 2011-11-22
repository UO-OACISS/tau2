#!/bin/sh
set -e

opari_dir=`pwd`
test_dir="../test"
test_data_dir=/mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/data

CC="gcc -std=c99"
INCDIR=.
OPARI2=/mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/../build-frontend/opari2
OPARI2_CONFIG="/mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/../build-frontend/opari2-config --config=/mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/../build-frontend/opari2_config.dat"
LDFLAGS="-lm -L/mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/../build-frontend/.libs -lpomp"
OPENMP="-fopenmp"
CFLAGS="-g -O2 -I../include  -w       -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -fPIC "
NM="`$OPARI2_CONFIG --nm`"


rm -rf $test_dir/jacobi/C
mkdir -p $test_dir/jacobi/C
cp -r $test_data_dir/jacobi/C/jacobi.c                            $test_dir/jacobi/C
cp -r $test_data_dir/jacobi/C/jacobi.h                            $test_dir/jacobi/C
cp -r $test_data_dir/jacobi/C/main.c                              $test_dir/jacobi/C
cp /mnt/netapp/home1/smillst/fresh/tau2/utils/opari2/build-frontend/../test/../src/pomp-lib-dummy/pomp2_lib.h                 $test_dir/jacobi/C/

cd $test_dir/jacobi/C
# instrument
$OPARI2 main.c 
$OPARI2 jacobi.c

# compile
$CC -I$INCDIR $OPENMP $CFLAGS -c main.mod.c
$CC -I$INCDIR $OPENMP $CFLAGS -c jacobi.mod.c

#build
$NM jacobi.mod.o main.mod.o | /bin/grep -E -i "T \.{0,1}_{0,2}pomp2_init_regions" | `$OPARI2_CONFIG --awk_cmd` -f `$OPARI2_CONFIG --awk_script`> pomp_init_file.c
$CC -I$INCDIR -c pomp_init_file.c
$CC $OPENMP $CFLAGS pomp_init_file.o jacobi.mod.o main.mod.o $LDFLAGS -o jacobi$EXEEXT

#run
OMP_NUM_THREADS=4 ./jacobi$EXEEXT 2>jacobi_test.out

grep 0: jacobi_test.out > jacobi_test_0
grep 1: jacobi_test.out > jacobi_test_1
grep 2: jacobi_test.out > jacobi_test_2
grep 3: jacobi_test.out > jacobi_test_3

if diff $test_data_dir/jacobi/C/jacobi_test_0.out jacobi_test_0 > /dev/null
  then
    true
  else
    echo "-------- ERROR: in program output --------"
    diff $test_data_dir/jacobi/C/jacobi_test_0.out jacobi_test_0
    cd $opari_dir
    exit
fi
if diff $test_data_dir/jacobi/C/jacobi_test_1.out jacobi_test_1 > /dev/null
  then
    true
  else
    echo "-------- ERROR: in program output --------"
    diff $test_data_dir/jacobi/C/jacobi_test_1.out jacobi_test_1
    cd $opari_dir
    exit
fi
if diff $test_data_dir/jacobi/C/jacobi_test_2.out jacobi_test_2 > /dev/null
  then
    true
  else
    echo "-------- ERROR: in program output --------"
    diff $test_data_dir/jacobi/C/jacobi_test_2.out jacobi_test_2
    cd $opari_dir
    exit
fi
if diff $test_data_dir/jacobi/C/jacobi_test_3.out jacobi_test_3 > /dev/null
  then
    true
  else
    echo "-------- ERROR: in program output --------"
    diff $test_data_dir/jacobi/C/jacobi_test_3.out jacobi_test_3
    cd $opari_dir
    exit
fi

cd $opari_dir
