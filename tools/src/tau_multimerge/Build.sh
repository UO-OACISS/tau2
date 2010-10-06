#!/bin/sh
cp -r src/edu/ .
#rm -rf edu/CVS/
#rm -rf edu/uoregon/CVS/
#rm -rf edu/uoregon/tau/CVS/
javac -classpath ../jtau_tf/bin/TAU_tf.jar edu/uoregon/tau/multimerge/*.java
jar -cf tau_multimerge.jar edu/
mv tau_multimerge.jar bin/
rm -rf edu/
