#!/bin/sh
cp -r src/edu/ .
javac edu/uoregon/tau/trace/*.java
rm edu/uoregon/tau/trace/*.java
jar -cf TAU_tf.jar edu/
mv TAU_tf.jar bin/
rm -rf edu/
