from edu.uoregon.tau.perfexplorer.glue import *

True = 1
False = 0

print "--------------- JPython test script start ------------"
files = []
#files.append("/home/khuck/tau2/examples/NPB2.3/bin")
#input = DataSourceResult(DataSourceResult.TAUPROFILE, files, False);
#files.append("/home/khuck/tau2/examples/NPB2.3/bin/data.ppk")
#files.append("/home/khuck/data/Heatmap/matsc72.ppk")
files.append("/home/khuck/data/Heatmap/m1commerr.ppk")
#files.append("/home/khuck/tau2/examples/pdt_mpi/c")
input = DataSourceResult(DataSourceResult.PPK, files, False);
messageHeatMap = BuildMessageHeatMap(input)
messageHeatMap.processData()
print "---------------- JPython test script end -------------"
