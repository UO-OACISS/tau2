#!/usr/bin/python
#
# this script converts files with this format:
#
# <timestamp> | <delta-begin> | <delta-end> | <location> | <metric 1> ... <metric N> | <tau callpath>
# Metrics: TIME P_WALL_CLOCK_TIME PAPI_TOT_CYC PAPI_TOT_INS
# 1255877968013309 | 1131 | 45137 | x_solve_cell:/scratch/Computational/khuck/NPB3.3-MPI/BT/x_solve.inst.f:556 | 1131 45137 1123 44741 1606105 68463696 4640164 159173258 | .TAU application => X_SOLVE [{x_solve.f} {5,7}-{111,9}]
#
# ...into a file with this format:
#
# TASK THREAD FUNCTION-ID HWC-ID NORMALIZED-TIME NORMALIZED-HWC
# 1 1 16 42000023 0.16745 0.0255415

import getopt
import sys
import os
import math
from operator import itemgetter

# global dictionary of MPI Types
mpiTypes = dict([("MPI_Send", 50000001), \
("MPI_Recv", 50000001), \
("MPI_Isend", 50000001), \
("MPI_Irecv", 50000001), \
("MPI_Wait", 50000001), \
("MPI_Waitall", 50000001), \
("MPI_Bcast", 50000002), \
("MPI_Barrier", 50000002), \
("MPI_Reduce", 50000002), \
("MPI_Allreduce", 50000002), \
("MPI_Comm_rank", 50000003), \
("MPI_Comm_size", 50000003), \
("MPI_Comm_create", 50000003), \
("MPI_Comm_dup", 50000003), \
("MPI_Comm_split", 50000003), \
("MPI_Init", 50000003), \
("MPI_Finalize", 50000003)]) \

# global dictionary of MPI values
mpiValues = dict([("MPI_Send", 1), \
("MPI_Recv", 2), \
("MPI_Isend", 3), \
("MPI_Irecv", 4), \
("MPI_Wait", 5), \
("MPI_Waitall", 6), \
("MPI_Bcast", 7), \
("MPI_Barrier", 8), \
("MPI_Reduce", 9), \
("MPI_Allreduce", 10), \
("MPI_Comm_rank", 19), \
("MPI_Comm_size", 20), \
("MPI_Comm_create", 21), \
("MPI_Comm_dup", 22), \
("MPI_Comm_split", 23), \
("MPI_Init", 31), \
("MPI_Finalize", 32)]) \

# need global dictionary of counters to types
counterMap = dict([("TIME", 42000000), \
("P_WALL_CLOCK_TIME", 42000001), \
("PAPI_TOT_CYC",      42000002), \
("PAPI_TOT_INS",      42000003), \
("PAPI_TOT_CYC",      42000004), \
("PAPI_TOT_INS",      42000005), \
("PAPI_L1_DCH",       42000006), \
("PAPI_L1_DCM",       42000007), \
("PAPI_L1_DCA",       42000008), \
("PAPI_LD_INS",       42000009), \
("PAPI_SR_INS",       42000010), \
("PAPI_BR_INS",       42000011), \
("PAPI_TOT_INS",      42000050), \
("PAPI_TOT_CYC",      42000059)])
ufType = "60000019"
thread = 0
node = 0
callpathMap = {}
callDepthMap = {}
negatives = 0
total = 0
startTimestamp = 0
endTimestamp = 0
mpiCallerType = 70000000

def usage():
	print "\nUsage: process.py [-m --mpi] [-c --callpath]\n"
	print "Where:"
	print "\t-m, --mpi      : keep MPI events"
	print "\t-c, --callpath : keep callpath\n"
	sys.exit(1)

def getFileExtents(infname):
	global startTimestamp
	global endTimestamp
	print "Pre-processing", infname, "..."
	input = open(infname, 'r')
	localStartTimestamp = 0
	localEndTimestamp = 0
	for line in input:
		line = line.strip()
		# get the node number
		if line.startswith('# node:') == True:
			index = 0
		# get the metric header
		if line.startswith('# Metrics:') == True:
			index = 0
		# handle the sample
		elif line.startswith('#') == False and len(line) > 0:
			# split the line:
			# timestamp | delta-begin | delta-end | location | metrics | callpath
			tokens = line.split("|")
			timestamp = int(tokens[0].strip())

			# get the time spent in this function
			time_begin = int(tokens[1].strip())
			time_end = int(tokens[2].strip())
			timeRange = time_begin + time_end

			# if this is the first line, save the start time for the app
			if (localStartTimestamp == 0):
				localStartTimestamp = timestamp - time_begin
				# if this process started first, save its time
				if startTimestamp == 0 or (localStartTimestamp < startTimestamp):
					startTimestamp = localStartTimestamp

			# remove the start time from the timestamp
			timestamp = (timestamp - startTimestamp) - time_begin

			# save the end timestamp
			localEndTimestamp = timestamp + timeRange

		if localEndTimestamp > endTimestamp:
			endTimestamp = localEndTimestamp

def processFile(infname, traceFile):
	"""Write something like this: 2:cpu:appl:task:thread:time:type:value:type:value...
	   where 
	   	cpu: cpu rank for the node (for now, same as thread), 1 indexed
		appl: application id (always 1)
		task: mpi rank (node), 1 indexed
		thread: thread (same as cpu), 1 indexed
		time: timestamp, minus first timestamp value (starts at 0)
		type: 60000019 for user function, something else for mpi, or id for metric
		value: id for user function, or counter value for metric
	   example:
	   2:5:1:5:1:189691800:60000019:15:60000119:42:42000050:54343482:42000000:4782"""
	metrics = {}
	numMetrics = 1
	global thread
	global counterMap
	global mpiTypes
	global mpiValues
	global negatives
	global total
	global node
	global startTimestamp
	global mpiCallerType
	global callDepthMap
	cpu = 1
	appl = 1
	thread = 1
	eventSet = set()
	print "Processing", infname, "..."
	input = open(infname, 'r')
	currentCallpath = ""
	for line in input:
		total = total + 1
		line = line.strip()
		# get the node number
		if line.startswith('# node:') == True:
			index = 0
			for token in line.split(" "):
				token = token.strip()
				if token != "#" and token != "node:":
					node = (int(token) + 1) # indexed starting at 1
			numMetrics = index
		# get the metric header
		if line.startswith('# Metrics:') == True:
			index = 0
			for token in line.split(" "):
				token = token.strip()
				if token != "#" and token != "Metrics:":
					metrics[index] = token
					index = index + 1
			numMetrics = index
		# handle the sample
		elif line.startswith('#') == False and len(line) > 0:
			isMPI = False
			# split the line:
			# timestamp | delta-begin | delta-end | location | metrics | callpath
			tokens = line.split("|")
			timestamp = int(tokens[0].strip())

			# get the time spent in this function
			time_begin = int(tokens[1].strip())
			time_end = int(tokens[2].strip())
			timeRange = time_begin + time_end

			# remove the start time from the timestamp
			timestamp = (timestamp - startTimestamp) - time_begin

			# get the sample location
			location = tokens[3].strip()

			# get the metric deltas
			metricDeltas = tokens[4].strip()
			metricTokens = metricDeltas.split(" ")

			# get the callpath
			callpath = tokens[5].strip()

			if callpath.find("MPI") > 0:
				isMPI = True

			callpathTokens = callpath.split("=>")
			callpath = callpathTokens[len(callpathTokens)-1].strip()

			if isMPI:
				callpathType = mpiTypes[callpath.rstrip("()")]
				callpathID = mpiValues[callpath.rstrip("()")]
			else:
				callpathType = ufType
				if not callpath in callpathMap:
					callpathMap[callpath] = len(callpathMap)
				callpathID = callpathMap[callpath]

			# have we seen this timestamp already?
			if (str(timestamp) + ":" + str(callpathID)) not in eventSet:
				eventSet.add(str(timestamp) + ":" + str(callpathID))
			else:
				continue

			# initialize some values
			goodData = True
			event = "2:"
			state = "1:"
			event = event + str(cpu) + ":"
			state = state + str(cpu) + ":"
			event = event + str(appl) + ":"
			state = state + str(appl) + ":"
			event = event + str(node) + ":"
			state = state + str(node) + ":"
			event = event + str(thread) + ":"
			state = state + str(thread) + ":"
			endEvent = event
			mpiCallerEvent = event
			event = event + str(timestamp) + ":"
			state = state + str(timestamp) + ":"
			state = state + str(timestamp + timeRange) + ":1\n"
			endEvent = endEvent + str(timestamp + timeRange) + ":" + str(callpathType) + ":0\n"
			event = event + str(callpathType) + ":"
			event = event + str(callpathID)

			if isMPI:
				mpiCallerEvent = mpiCallerEvent + str(timestamp)
				for t in range(1,len(callpathTokens),1):
					typeIndex = (len(callpathTokens) - 1) - t
					tmpCaller = callpathTokens[typeIndex].strip()
					callpathID = callpathMap[tmpCaller]
					mpiCallerEvent = mpiCallerEvent + ":" + str(mpiCallerType + t) + ":" + str(callpathID)
					callDepthMap[mpiCallerType + t] = "MPI caller at level " + str(t)
				mpiCallerEvent = mpiCallerEvent + "\n"

			# split the metric values
			for m in range(numMetrics):
				start = int(metricTokens[m*2])
				end = int(metricTokens[(m*2)+1])
				metricRange = start + end
				if time_begin < 0.0 or start < 0.0:
					#print "ignoring negative value: ", line
					negatives = negatives + 1
					goodData = False

				event = event + ":" + str(counterMap[metrics[m]]) + ":"
				event = event + str(metricRange)
			event = event + "\n"

			if goodData:
				traceFile.write(state)
				if isMPI:
					traceFile.write(mpiCallerEvent)
				traceFile.write(event)
				traceFile.write(endEvent)

def sortedDictValues(adict):
	items = adict.items()
	items.sort()
	return [value for key, value in items]

def writePcfFile(callpathMap):
	global callDepthMap
	global mpiTypes
	global counterMap
	pcfname = "tracefile.pcf"
	pcfFile = open(pcfname, 'w')

	pcfFile.write("DEFAULT_OPTIONS\n\n")
	pcfFile.write("LEVEL               THREAD\n")
	pcfFile.write("UNITS               NANOSEC\n")
	pcfFile.write("LOOK_BACK           100\n")
	pcfFile.write("SPEED               1\n")
	pcfFile.write("FLAG_ICONS          ENABLED\n")
	pcfFile.write("NUM_OF_STATE_COLORS 1000\n")
	pcfFile.write("YMAX_SCALE          37\n\n\n")
	pcfFile.write("DEFAULT_SEMANTIC\n\n")
	pcfFile.write("THREAD_FUNC          State As Is\n\n")

	pcfFile.write("EVENT_TYPE\n")
	pcfFile.write("9    50000001    MPI Point-to-point\n")
	pcfFile.write("VALUES\n")
	for (k,v) in mpiValues.items():
		if mpiTypes[k] == 50000001:
			pcfFile.write(str(v) + "   " + str(k) + "\n")
	pcfFile.write(str(v) + "   End\n")
	pcfFile.write("\n\n")

	pcfFile.write("EVENT_TYPE\n")
	pcfFile.write("9    50000002    MPI Collective Comm\n")
	pcfFile.write("VALUES\n")
	for (k,v) in mpiValues.items():
		if mpiTypes[k] == 50000002:
			pcfFile.write(str(v) + "   " + str(k) + "\n")
	pcfFile.write(str(v) + "   End\n")
	pcfFile.write("\n\n")

	pcfFile.write("EVENT_TYPE\n")
	pcfFile.write("9    50000003    MPI Other\n")
	pcfFile.write("VALUES\n")
	for (k,v) in mpiValues.items():
		if mpiTypes[k] == 50000003:
			pcfFile.write(str(v) + "   " + str(k) + "\n")
	pcfFile.write(str(v) + "   End\n")
	pcfFile.write("\n\n")

	sortedList = sorted(counterMap.iteritems(), key=itemgetter(1))
	pcfFile.write("EVENT_TYPE\n")
	for i in sortedList:
		pcfFile.write("7  " + str(i[1]) + " " + str(i[0]) + "\n")
	pcfFile.write("\n\n")

	sortedList = sorted(callpathMap.iteritems(), key=itemgetter(1))
	pcfFile.write("EVENT_TYPE\n")
	for (k,v) in callDepthMap.items():
		pcfFile.write("0    " + str(k) + "    " + v + "\n")
	pcfFile.write("VALUES\n")
	for i in sortedList:
		pcfFile.write(str(i[1]) + "   " + str(i[0]) + "\n")
	pcfFile.write("\n\n")

	sortedList = sorted(callpathMap.iteritems(), key=itemgetter(1))
	pcfFile.write("EVENT_TYPE\n")
	pcfFile.write("0    60000019    User function\n")
	pcfFile.write("VALUES\n")
	for i in sortedList:
		pcfFile.write(str(i[1]) + "   " + str(i[0]) + "\n")
	pcfFile.write("\n\n")

	pcfFile.close()
	print pcfname, "mapping file created"

def writeRowFile(numFiles):
	rowfname = "tracefile.row"
	rowFile = open(rowfname, 'w')
	rowFile.write("LEVEL CPU SIZE " + str(numFiles) + " \n")
	for i in range(numFiles):
		rowFile.write(str(numFiles) + ".unknown\n")
	rowFile.write("\n\n")
	rowFile.write("LEVEL NODE SIZE " + str(1) + " \nunknown\n")
	rowFile.close()
	print rowfname, "row file created"

def createTraceFile(tracefname, numFiles):
	traceFile = open(tracefname, 'w')
	traceFile.write("#Paraver (08/09/2009 at 15:33):" + str(endTimestamp) + "_ns:" + str(numFiles) + "(")
	for i in range(numFiles):
		if (i > 0):
			traceFile.write(",")
		traceFile.write("1")
	traceFile.write("):1:" + str(numFiles) + "(")
	for i in range(numFiles):
		if (i > 0):
			traceFile.write(",")
		traceFile.write("1:" + str(i+1))
	traceFile.write("),0\n");

	# write the communicator header
	#traceFile.write("c:1:1:" + str(numFiles));
	#for i in range(numFiles):
	#traceFile.write(":" + str(i+1))
	#traceFile.write("\n");
	#for i in range(numFiles):
	#traceFile.write("c:1:" + str(i+2) + ":1:1\n")
	return traceFile

def main(argv):
	global callpathMap
	global negatives
	global total
	global node
	global thread
	
	callpathMap["End"] = len(callpathMap)
	callpathMap["Unresolved"] = len(callpathMap)
	callpathMap["_NOT_Found"] = len(callpathMap)

	dirList=os.listdir(".")
	files = set()
	for infname in dirList:
		if infname.startswith("ebstrace.processed."):
			files.add(infname)
		#break
	numFiles = len(files)

	for infname in files:
		getFileExtents(infname)

	tracefname = "tracefile.prv"
	traceFile = createTraceFile(tracefname, numFiles)
	for infname in files:
		processFile(infname, traceFile)

	traceFile.close()
	print tracefname, "trace file created"

	writePcfFile(callpathMap)
	writeRowFile(numFiles);

	print negatives, "negative values ignored out of", total, "total values"

if __name__ == "__main__":
	main(sys.argv[1:])
