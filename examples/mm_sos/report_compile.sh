export SOS_PATH=/home/users/jalcaraz/tau2_merge1_cfgmodv2/x86_64/sos/sos_flow_master/inst
mpicc report.c -L$SOS_PATH/lib -lsos -Wl,-rpath,$SOS_PATH/lib -I$SOS_PATH/include/ -o report
