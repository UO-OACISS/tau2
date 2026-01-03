#!/bin/sh
cp -r src/edu/ .
javac -classpath ../jtau_tf/bin/TAU_tf.jar edu/uoregon/tau/multimerge/*.java
jar -cf tau_multimerge.jar edu/
mv tau_multimerge.jar bin/
rm -rf edu/
