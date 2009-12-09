#!/bin/sh
cp -r src/edu/ .
rm -rf edu/CVS/
rm -rf edu/uoregon/CVS/
rm -rf edu/uoregon/tau/CVS/
javac -classpath ../jtau_tf/bin/TAU_tf.jar:../contrib/slog2sdk/lib/traceTOslog2.jar edu/uoregon/tau/*.java
jar -cf tau2slog2.jar edu/
mv tau2slog2.jar bin/
rm -rf edu/
