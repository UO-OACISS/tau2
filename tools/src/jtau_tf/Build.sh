#!/bin/sh
cp -r src/edu/ .
javac edu/uoregon/tau/tau_tf/*
jar -cf TAU_tf.jar edu/
mv TAU_tf.jar bin/
rm -rf edu/
