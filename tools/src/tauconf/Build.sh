#!/bin/sh

cp -r src/tau_conf .
javac tau_conf/TAU_Conf.java
jar -cf TAU_Conf.jar tau_conf
mv TAU_Conf.jar bin
rm -rf tau_conf
