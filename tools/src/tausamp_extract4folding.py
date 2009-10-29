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
thread = 0
node = 0
callpathMap = {}
ignoreMPI = True
ignoreCallpath = True
negatives = 0
total = 0

def usage():
	print "\nUsage: process.py [-m --mpi] [-c --callpath]\n"
	print "Where:"
	print "\t-m, --mpi      : keep MPI events"
	print "\t-c, --callpath : keep callpath\n"
	sys.exit(1)

def parseArgs(argv):
	global ignoreMPI
	global ignoreCallpath
	try:
		opts, args = getopt.getopt(argv, "hmc", ["help", "mpi", "callpath"])
	except getopt.GetoptError:
		usage()
	
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			usage()
		elif opt in ("-m", "--mpi"):
			ignoreMPI = False
		elif opt in ("-c", "--callpath"):
			ignoreCallpath = False
	
	return

def processFile(infname, outputFile):
	metrics = {}
	numMetrics = 1
	global thread
	global counterMap
	global ignoreMPI
	global ignoreCallpath
	global negatives
	global total
	global node
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
				if token != "#" and token != "Metrics:":
					node = token
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
			# split the line:
			# timestamp | delta-begin | delta-end | location | metrics | callpath
			tokens = line.split("|")
			timestamp = tokens[0].strip()
			delta_begin = tokens[1].strip()
			delta_end = tokens[2].strip()
			start = float(delta_begin)
			end = float(delta_end)
			timeRange = start + end
			normalizedTime = start / timeRange
			location = tokens[3].strip()
			metricDeltas = tokens[4].strip()
			metricTokens = metricDeltas.split(" ")

			callpath = tokens[5].strip()
			if callpath.find("MPI") > 0 and ignoreMPI:
				continue
			if ignoreCallpath:
				callpathTokens = callpath.split("=>")
				callpath = callpathTokens[len(callpathTokens)-1].strip()
			if not callpath in callpathMap:
				callpathMap[callpath] = len(callpathMap)+1
			callpathID = callpathMap[callpath]

			goodData = True
			tmp = ""

			for m in range(numMetrics):
				start = float(metricTokens[m*2])
				end = float(metricTokens[(m*2)+1])
				metricRange = start + end
				normalizedMetric = start / metricRange
				if normalizedTime < 0.0 or normalizedMetric < 0.0:
					#print "ignoring negative value: ", line
					negatives = negatives + 1
					goodData = False

				#tmp = tmp + str(callpathID) + " "
				#tmp = tmp + "S " + str(counterMap[metrics[m]]) + " "
				tmp = tmp + "S " + metrics[m] + " "
				tmp = tmp + str(normalizedTime) + " "
				tmp = tmp + str(normalizedMetric) + "\n"

			if goodData:
				if callpathID != currentCallpath:
					currentCallpath = callpathID
					#outputFile.write("T " + str(callpathID) + " " + str(timeRange) + "\n")
					outputFile.write("T " + callpath.replace(" ", ":") + " " + str(timeRange) + "\n")
				outputFile.write(tmp)

def sortedDictValues(adict):
	items = adict.items()
	items.sort()
	return [value for key, value in items]

def main(argv):
	global callpathMap
	global negatives
	global total
	global node
	global thread
	parseArgs(argv)
	dirList=os.listdir(".")
	print "Processing..."
	for infname in dirList:
		if infname.startswith("ebstrace.processed."):
			outfname = infname.replace("processed", "extracted")
			outputFile = open(outfname, 'w')
			processFile(infname, outputFile)
			outputFile.close()
			newOutfname = outfname.replace(".0.0.0", "." + node + "." + str(thread))
			os.rename(outfname, newOutfname)
			print infname, " --> ", newOutfname

	outfname = "ebstrace.extracted.maps.txt"
	outputFile = open(outfname, 'w')

	sortedList = sorted(callpathMap.iteritems(), key=itemgetter(1))
	outputFile.write("# function map \n")
	for i in sortedList:
		outputFile.write(str(i[1]) + " " + str(i[0]) + "\n")

	sortedList = sorted(counterMap.iteritems(), key=itemgetter(1))
	outputFile.write("# metric map \n")
	for i in sortedList:
		outputFile.write(str(i[1]) + " " + str(i[0]) + "\n")

	outputFile.close()
	print outfname, "mapping file created"
	print negatives, "of", total, "negative values ignored"

if __name__ == "__main__":
	main(sys.argv[1:])
