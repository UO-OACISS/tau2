#!/bin/sh

cp -r src/tau_conf bin
javac bin/tau_conf/TAU_Conf.java
jar -cvf bin/TAU_Conf.jar bin/tau_conf
rm -rf bin/tau_conf
