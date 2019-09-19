architecture="$(grep -oP '(?<=TAU_ARCH=).*' ../../include/Makefile)"
#MY_TAU_PATH="$(dirname $(dirname  `pwd`))"
MY_TAU_PATH="$(grep -oP '(?<=TAUROOT=).*' ../../include/Makefile)"
SOS_PATH=$MY_TAU_PATH/$architecture/sos/sos_flow_master/inst
mpicc report.c -L$SOS_PATH/lib -lsos -Wl,-rpath,$SOS_PATH/lib -I$SOS_PATH/include/ -o report
