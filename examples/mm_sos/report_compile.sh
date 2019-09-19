SOS_PATH="$(grep -oP '(?<=SOSDIR=).*' ../../include/Makefile)"
mpicc report.c -L$SOS_PATH/lib -lsos -Wl,-rpath,$SOS_PATH/lib -I$SOS_PATH/include/ -o report
