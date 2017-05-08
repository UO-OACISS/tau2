#! /usr/bin/env python

""" Wrapper to force the use of TAU's TauSparkProfiler in Spark """

import sys
import os
try:
    import pyspark
except ImportError:
    raise ImportError("pyspark module not found. Are you running outside Spark?")
try:
    import tau
except ImportError:
    raise ImportError("tau module not found. Make sure your PYTHONPATH is set properly.")
import runpy

# Exit early if TauSparkProfiler doesn't exist
try:
    x = tau.TauSparkProfiler
except Exception:
    print("ERROR: Unable to initialize TAU's PySpark profiling support. \
           Please check that you are using a supported version of Spark.")
    sys.exit(1)

# Remove the wrapper from argv so the script we're wrapping
# will have its own path as argv[0]
sys.argv = sys.argv[1:]

# Monkey patch SparkContext to use TauSparkProfiler as the default profiler
# and enable profiling by default
real_context_init = pyspark.SparkContext.__init__

def wrapped_context_init(self, master=None, appName=None, sparkHome=None, pyFiles=[],
             environment=None, batchSize=0, serializer=pyspark.serializers.PickleSerializer(),
             conf=pyspark.SparkConf(), gateway=None, jsc=None, profiler_cls=tau.TauSparkProfiler):
    if profiler_cls is not tau.TauSparkProfiler:
        # The user explicitly chose a different profiler
        print("WARNING: profiler_cls is explicitly set during SparkContext initialization, "\
              "overriding TAU's TauSparkProfiler. TAU will not profile PySpark tasks.")
    else:
        # Profiling must be enabled so that TauSparkProfiler.profile() wraps tasks
        conf.set("spark.python.profile", "true")
    # Distribute the script we're wrapping to the executors
    if sys.argv[0] != '-m':
        pyFiles.append(sys.argv[0])
    real_context_init(self, master=master, appName=appName, sparkHome=sparkHome, pyFiles=pyFiles,
            environment=environment, batchSize=batchSize, serializer=serializer, conf=conf,
            gateway=gateway, jsc=jsc, profiler_cls=profiler_cls)

pyspark.SparkContext.__init__ = wrapped_context_init

# Run the module or script we're wrapping
if sys.argv[0] == '-m':
    module_name = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)
else:
    runpy.run_path(sys.argv[0], run_name="__main__")


