#!/bin/sh

cp -r src/tau_conf .
rm -r ./tau_conf/CVS
javac -classpath ../contrib/swing-layout-0.7.1.jar tau_conf/TAUOption.java tau_conf/OptionSet.java tau_conf/TAU_Conf.java
jar -cf TAU_Conf.jar tau_conf
mv TAU_Conf.jar bin
rm -rf tau_conf
