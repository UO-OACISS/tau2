#!/bin/sh
cp -r src/edu/ .
rm -rf edu/CVS/
rm -rf edu/uoregon/CVS/
rm -rf edu/uoregon/tau/CVS/
rm -rf edu/uoregon/tau/trace/CVS/
javac edu/uoregon/tau/trace/*.java
rm edu/uoregon/tau/trace/*.java
jar -cf TAU_tf.jar edu/
mv TAU_tf.jar bin/
rm -rf edu/
