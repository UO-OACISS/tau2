from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfexplorer.rules import *
from edu.uoregon.tau.perfexplorer.client import *
from java.util import *
from java.lang import *
import sys
import time

#########################################################################################

True = 1
False = 0
inputData = "cpi.csv"
rules = "./CPI-Stack.drl"
fractionThreshold = 01.0        # ignore clusters smaller than this fraction of runtime
                                # Why: only examine significant clusters
cpiThreshold = 1.00             # ignore clusters with CPI smaller than this
                                # Why: no need to examine good CPI clusters
completionThreshold = 0.3       # ignore completion stalls less than this
                                # Why: should be .25, higher value means pipeline stalled
								# or flushed
gctThreshold = 0.15             # ignore GCT stalls less than this
                                # Why: This represents penalties from branch mispredictions
								# and the threshold is lower than for functional units
stallThreshold = 0.40           # ignore functional unit resource stalls less than this
                                # Why: this reprensents stalls in the functional units
branchStallThreshold = 0.05     # ignore branch resource stalls less than this
                                # Why: this should be close to zero. Higher means the
								# pipeline is draining from mispredictions.
longFPUStallThreshold = 0.10    # ignore long FPU resource stalls less than this
                                # Why: this is latency processing div/sqrt, which take ~40 cyles
flushThreshold = 0.25           # what fraction of flushes that are LSU are a problem?
                                # Why: most flushes should be due to branch mispredictions.
								# if this goes up, then L1 cache lines are getting evicted.
instPerFlushThreshold1 = 100.0  # what is a bad (very low) number of instructions per LSU flush?
instPerFlushThreshold2 = 1000.0 # what is a moderate (low) number of instructions per LSU flush?
instMixThreshold = 0.30         # ignore instructions less than this % of total
                                # Why: if the mix is high, there could be more functional unit
								# latency
longFPUInstMixThreshold = 0.01  # ignore instructions less than this % of total
                                # Why: if the mix is high, there could be more functional unit
								# latency
branchMixThreshold = 0.03       # ignore instructions less than this % of total
                                # Why: if we have a lot of branches, than mispredictions
								# can drain the pipeline
mispredictionThreshold = 0.05   # ignore mispredictions less than this % of total
                                # Why: if we have a lot of branches, than mispredictions
								# can drain the pipeline
mispredFlushThreshold = 400.0   # what is a bad number of instructions per pipeline flush?
                                # Why: if 5% of instructions are branches, and 5% of those
								# are mispredicted, then 0.05 * 0.05 = 1/400
completionRatioThreshold = 0.3  # ignore completion ratios more than this 
                                # Why: low ratios indicate speculative execution, which
								# occupies functional units
issueRateThreshold = 2.0        # ignore issue rates more than this 
                                # Why: if this is low, then...?

#########################################################################################

def getParameters():
	global inputData
	global rules
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	for key in keys:
		print key, parameterMap.get(key)
	inputData = parameterMap.get("inputData")
	rules = parameterMap.get("rules")

#########################################################################################

def makeTestStack():
	cpiStack = HashMap()
	# set more than 1.0 (or current threshold) to activate
	cpiStack.put(String("CPI"),Double(1.1))
	# set more than 0.25 (or current threshold) to activate
	cpiStack.put(String("<A>"),Double(.25))
	cpiStack.put(String("<A1>"),Double(.45))
	cpiStack.put(String("<A2>"),Double(.45))
	cpiStack.put(String("<B>"),Double(.25))
	cpiStack.put(String("<B1>"),Double(.45))
	cpiStack.put(String("<B2>"),Double(.45))
	cpiStack.put(String("<B4>"),Double(.45))
	cpiStack.put(String("<C>"),Double(.85))
	cpiStack.put(String("<C1>"),Double(.45))
	cpiStack.put(String("<C1A>"),Double(.55))
	cpiStack.put(String("<C1A1>"),Double(.65))
	cpiStack.put(String("<C1A2>"),Double(.65))
	cpiStack.put(String("<C1B>"),Double(.55))
	cpiStack.put(String("<C1C>"),Double(.55))
	cpiStack.put(String("<C2>"),Double(.45))
	cpiStack.put(String("<C2A>"),Double(.55))
	cpiStack.put(String("<C2C>"),Double(.55))
	cpiStack.put(String("<C3>"),Double(.45))
	cpiStack.put(String("<C3A>"),Double(.55))
	cpiStack.put(String("<C3B>"),Double(.55))
	cpiStack.put(String("<C4>"),Double(.45))
	return cpiStack

#########################################################################################

def parseCounters(info, threshold):
	# the cluster_info.csv looks something like this:
	# 
	# Cluster Name,NOISE,Cluster 1,Cluster 2,Cluster 3,Cluster 4,Cluster 5,Cluster 6,Cluster 7
	# Density,181,6278,5280,308,352,66,21,26
	# Total duration,1305911719,25687351615,6767546266,4734675853,1395746567,237486001,97304570,87153356
	# Avg. duration,7214981,4091645,1281732,15372324,3965189,3598272,4633550,3352052
	# % Total duration,0.00000,0.65853,0.17349,0.12138,0.03578,0.00609,0.00249,0.00223
	# PM_INST_CMPL,9739345,7757544,607944,18125535,1271464,7454971,7837559,7017536
	# PM_CYC,16335615,9256833,2901958,34820904,8946617,8110893,10502522,7551541
	# ...

	numFunctions=0
	numClusters=1

	print "Parsing cluster info from: ", info
	i = open(info, 'r')

	names=[]
	densities=[]
	totalDurations=[]
	averageDurations=[]
	percentDurations=[]
	counters={} # this will be a dictionary of lists
	maxClusters=0

	for line in i.readlines():
		tokens = line.strip().split(',')
		if tokens[0].strip("\"") == "Cluster Name":
			for j in range(1,len(tokens)):
				names.append(tokens[j].strip("\""))
		elif tokens[0].strip("\"") == "Density":
			for j in range(1,len(tokens)):
				densities.append(tokens[j].strip("\""))
		elif tokens[0].strip("\"") == "Total duration":
			for j in range(1,len(tokens)):
				totalDurations.append(int(tokens[j]))
		elif tokens[0].strip("\"") == "Avg. duration":
			for j in range(1,len(tokens)):
				averageDurations.append(int(tokens[j]))
		elif tokens[0].strip("\"") == "% Total duration" or tokens[0].strip("\"") == "% Total Duration":
			for j in range(1,len(tokens)):
				percentDurations.append(float(tokens[j]))
		else:
			counts=[]
			for j in range(1,len(tokens)):
				if tokens[j] == "nan" or tokens[j] == "-nan":
					counts.append(0)
				else:
					counts.append(int(tokens[j]))
			counters[tokens[0].strip("\"")] = counts
	i.close()

	tmpVal = float(threshold)*0.01
	maxClusters = len(percentDurations)
	for p in range(1,len(percentDurations)):
		if percentDurations[p] < tmpVal:
			maxClusters=p
			break
	print "threshold = " + str(threshold) + "%   maxClusters = " + str(maxClusters-1)

	return names, densities, totalDurations, averageDurations, percentDurations, counters, maxClusters

#########################################################################################

def parseBenchmarkCounters(info, threshold):
	# the cluster_info.csv looks something like this:
	# 
# benchmark,size,PM_INST_CMPL_0,PM_CYC_0,PM_GCT_EMPTY_CYC,PM_LSU_LMQ_SRQ_EMPTY_CYC,PM_HV_CYC,PM_1PLUS_PPC_CMPL,PM_GRP_CMPL,PM_TB_BIT_TRANS,TIME_0,PM_INST_CMPL_1,PM_CYC_1,PM_GCT_EMPTY_CYC,PM_FLUSH_BR_MPRED,PM_BR_MPRED_TA,PM_GCT_EMPTY_IC_MISS,PM_GCT_EMPTY_BR_MPRED,PM_L1_WRITE_CYC,TIME_1,PM_INST_CMPL_2,PM_CYC_2,PM_LSU0_BUSY,PM_LSU1_BUSY,PM_LSU_FLUSH,PM_FLUSH_LSU_BR_MPRED,PM_CMPLU_STALL_LSU,PM_CMPLU_STALL_ERAT_MISS,TIME_2,PM_INST_CMPL_3,PM_CYC_3,PM_CMPLU_STALL_OTHER,PM_LD_MISS_L1,PM_CMPLU_STALL_DCACHE_MISS,PM_LSU_DERAT_MISS,PM_CMPLU_STALL_REJECT,PM_LD_REF_L1,TIME_3,PM_INST_CMPL_4,PM_CYC_4,PM_GCT_EMPTY_SRQ_FULL,PM_FXU_FIN,PM_FPU_FIN,PM_CMPLU_STALL_FXU,PM_FXU_BUSY,PM_CMPLU_STALL_DIV,TIME_4,PM_INST_CMPL_5,PM_CYC_5,PM_FPU_FDIV,PM_FPU_FMA,PM_IOPS_CMPL,PM_CMPLU_STALL_FDIV,PM_FPU_FSQRT,PM_CMPLU_STALL_FPU,TIME_5,PM_INST_CMPL_6,PM_CYC_6,PM_INST_DISP,PM_DATA_FROM_MEM,PM_LD_MISS_L1,PM_GCT_FULL_CYC,PM_ST_REF_L1,PM_LD_REF_L1,TIME_6,PM_INST_CMPL_7,PM_CYC_7,PM_DTLB_MISS,PM_ITLB_MISS,PM_LD_MISS_L1,PM_ST_MISS_L1,PM_ST_REF_L1,PM_LD_REF_L1,TIME_7,PM_INST_CMPL_8,PM_CYC_8,PM_FPU_FDIV,PM_FPU_FMA,PM_FPU0_FIN,PM_FPU1_FIN,PM_FPU_STF,PM_LSU_LDF,TIME_8,PM_INST_CMPL_9,PM_CYC_9,PM_DATA_FROM_L2,PM_DATA_FROM_MEM,PM_LD_MISS_L1_LSU0,PM_1PLUS_PPC_CMPL,PM_LD_MISS_L1_LSU1,PM_LD_REF_L1,TIME_9
# fxu_kernel_1,1000,500,724,115,270,0,136,126,3,1,496,641,82,14,11,48,25,31,0,496,649,32,53,0,13,169,4,0,496,617,0,14,72,4,9,207,0,498,684,0,297,3,197,95,138,0,496,651,0,0,385,5,0,6,0,496,667,1347,0,14,133,61,202,0,496,644,0,0,15,1,62,207,1,496,626,0,0,1,1,1,1,0,496,636,11,0,6,132,8,227,1
# fxu_kernel_6,1000,529,752,75,216,0,147,136,3,1,529,680,81,15,12,55,21,44,0,529,636,34,25,3,15,148,7,0,529,731,0,19,77,6,6,216,0,531,708,0,345,4,196,126,140,0,529,689,0,0,418,6,0,8,1,529,683,1140,0,15,65,63,204,0,529,717,0,0,16,2,63,204,1,529,666,0,0,2,2,2,2,0,529,695,15,0,9,148,11,200,0
# fxu_kernel_12,1000,566,752,87,217,0,155,144,3,0,562,760,116,13,11,82,28,71,0,562,730,39,26,2,15,250,5,0,562,716,0,20,72,1,2,212,0,564,762,0,339,3,158,134,92,0,562,712,0,0,451,5,0,6,0,562,738,1241,0,13,1,66,214,1,562,734,0,0,21,0,66,233,0,562,729,0,0,1,1,1,1,0,562,743,11,0,6,157,8,216,0
	# ...

	numFunctions=0
	numClusters=1

	print "Parsing benchmark info from: ", info
	i = open(info, 'r')

	names=["NOISE"]
	densities=[0]
	totalDurations=[0]
	averageDurations=[0]
	percentDurations=[0.0]
	counters={} # this will be a dictionary of lists
	maxClusters=1
	tmpCounterNames=[]

	for line in i.readlines():
		tokens = line.strip().split(',')
		if tokens[0].find("benchmark") == 0:
			for j in range(2,len(tokens)):
				counters[tokens[j].strip("_0")] = [0]
				tmpCounterNames.append(tokens[j].strip("_0"))
		else:
			names.append(tokens[0] + "_" + tokens[1])
			densities.append(tokens[1])
			totalDurations.append(int(100))
			averageDurations.append(int(100))
			percentDurations.append(float(100))
			j = 2
			for hwc in tmpCounterNames:
				if len(counters[hwc]) == maxClusters:
					counters[hwc].append(int(tokens[j]))
				j = j + 1
			maxClusters = maxClusters + 1
	i.close()

	print "threshold = " + str(threshold) + "%   maxClusters = " + str(maxClusters)

	return names, densities, totalDurations, averageDurations, percentDurations, counters, maxClusters

#########################################################################################

def handleNone(dictionary, name, index):
	if name in dictionary.keys():
		return dictionary[name][index]
	else:
		return 0

def handleNone1(dictionary, name, index):
	if name in dictionary.keys():
		if dictionary[name][index] > 0.0:
			return dictionary[name][index]
		else:
			return 1
	else:
		return 1

#########################################################################################

def computeCPIStats(names, percentDurations, counters, currentCluster):
	cpiStack = HashMap()

	# output cluster names
	j = currentCluster
	divisor = float(handleNone1(counters,"PM_INST_CMPL",j))
	cpiStack.put(String("Fraction of Total Duration"),Double(percentDurations[j]))
	cpiStack.put(String("CPI"),Double(float(handleNone(counters,"PM_CYC",j)) / divisor))
	# output completion cycles <A> 
	cpiStack.put(String("<A>"),Double(float(handleNone(counters,"PM_GRP_CMPL",j)) / divisor))
	cpiStack.put(String("<A1>"),Double(float(handleNone(counters,"PM_1PLUS_PPC_CMPL",j)) / divisor))
	cpiStack.put(String("<A2>"),Double(float(handleNone(counters,"PM_GRP_CMPL",j) - handleNone(counters,"PM_1PLUS_PPC_CMPL",j)) / divisor))

	# output empty cycles <B> 
	cpiStack.put(String("<B>"),Double(float(handleNone(counters,"PM_GCT_EMPTY_CYC",j)) / divisor))
	cpiStack.put(String("<B1>"),Double(float(handleNone(counters,"PM_GCT_EMPTY_IC_MISS",j)) / divisor))
	cpiStack.put(String("<B2>"),Double(float(handleNone(counters,"PM_GCT_EMPTY_BR_MPRED",j)) / divisor))
	cpiStack.put(String("<B4>"),Double(float(handleNone(counters,"PM_GCT_EMPTY_CYC",j) - handleNone(counters,"PM_GCT_EMPTY_IC_MISS",j) - handleNone(counters,"PM_GCT_EMPTY_BR_MPRED",j)) / divisor))

	# output stall cycles <C>
	cpiStack.put(String("<C>"),Double(float(handleNone(counters,"PM_CYC",j) - handleNone(counters,"PM_GRP_CMPL",j) - handleNone(counters,"PM_GCT_EMPTY_CYC",j)) / divisor))
	cpiStack.put(String("<C1>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_LSU",j)) / divisor))
	cpiStack.put(String("<C1A>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_REJECT",j)) / divisor))
	cpiStack.put(String("<C1A1>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_ERAT_MISS",j)) / divisor))
	cpiStack.put(String("<C1A2>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_REJECT",j) - handleNone(counters,"PM_CMPLU_STALL_ERAT_MISS",j)) / divisor))
	cpiStack.put(String("<C1B>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_DCACHE_MISS",j)) / divisor))
	cpiStack.put(String("<C1C>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_LSU",j) - handleNone(counters,"PM_CMPLU_STALL_REJECT",j) - handleNone(counters,"PM_CMPLU_STALL_DCACHE_MISS",j)) / divisor))
	cpiStack.put(String("<C2>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_FXU",j)) / divisor))
	cpiStack.put(String("<C2A>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_DIV",j)) / divisor))
	cpiStack.put(String("<C2C>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_FXU",j) - handleNone(counters,"PM_CMPLU_STALL_DIV",j)) / divisor))
	cpiStack.put(String("<C3>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_FPU",j)) / divisor))
	cpiStack.put(String("<C3A>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_FDIV",j)) / divisor))
	cpiStack.put(String("<C3B>"),Double(float(handleNone(counters,"PM_CMPLU_STALL_FPU",j) - handleNone(counters,"PM_CMPLU_STALL_FDIV",j)) / divisor))
	cpiStack.put(String("<C4>"),Double(float(handleNone(counters,"PM_CYC",j) - handleNone(counters,"PM_GRP_CMPL",j) - handleNone(counters,"PM_GCT_EMPTY_CYC",j) - handleNone(counters,"PM_CMPLU_STALL_LSU",j) - handleNone(counters,"PM_CMPLU_STALL_FXU",j) - handleNone(counters,"PM_CMPLU_STALL_FPU",j)) / divisor))

	# do the flush statistics
	#loads flushed
	divisor = float(handleNone1(counters,"PM_INST_CMPL",j))
	cpiStack.put(String("Loads Flushed"),Double(float(handleNone(counters,"PM_LSU_FLUSH",j)) / divisor))
	#unaligned load percentage
	cpiStack.put(String("Unaligned Loads %"),Double(float(handleNone(counters,"PM_LSU_FLUSH_ULD",j)) / divisor))
	#unaligned store percentage
	cpiStack.put(String("Unaligned Stores %"),Double(float(handleNone(counters,"PM_LSU_FLUSH_UST",j)) / divisor))

	divisor = float(handleNone1(counters,"PM_CYC",j))
	#unaligned load rate
	cpiStack.put(String("Unaligned Load Rate"),Double(float(handleNone(counters,"PM_LSU_FLUSH_ULD",j)) / divisor))
	#unaligned store rate
	cpiStack.put(String("Unaligned Store Rate"),Double(float(handleNone(counters,"PM_LSU_FLUSH_UST",j)) / divisor))
	#lsu flush rate
	cpiStack.put(String("LSU Flush Rate"),Double(float(handleNone(counters,"PM_LSU_FLUSH",j)) / divisor))
	#lrq flush rate
	cpiStack.put(String("LRQ Flush Rate"),Double(float(handleNone(counters,"PM_LSU_FLUSH_LRQ",j)) / divisor))
	#srq flush rate
	cpiStack.put(String("SRQ Flush Rate"),Double(float(handleNone(counters,"PM_LSU_FLUSH_SRQ",j)) / divisor))
	# full queues
	cpiStack.put(String("Cycles FPU0 Queue Full"),Double(float(handleNone(counters,"PM_FPU0_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles FPU1 Queue Full"),Double(float(handleNone(counters,"PM_FPU1_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles FXU0/LSU0 Queue Full"),Double(float(handleNone(counters,"PM_FXLS0_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles FXU1/LSU1 Queue Full"),Double(float(handleNone(counters,"PM_FXLS1_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles LSU LRQ Queue Full"),Double(float(handleNone(counters,"PM_LSU_LRQ_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles LSU SRQ Queue Full"),Double(float(handleNone(counters,"PM_LSU_SRQ_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles BR Queue Full"),Double(float(handleNone(counters,"PM_BRQ_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles CR Queue Full"),Double(float(handleNone(counters,"PM_CRQ_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles CR Map Full"),Double(float(handleNone(counters,"PM_CR_MAP_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles LR_CTR Map Full"),Double(float(handleNone(counters,"PM_LR_CTR_MAP_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles XER Map Full"),Double(float(handleNone(counters,"PM_XER_MAP_FULL_CYC",j)) / divisor))
	cpiStack.put(String("Cycles Group Dispatch Block Map Full"),Double(float(handleNone(counters,"PM_GRP_DISP_BLK_SB_CYC",j)) / divisor))
	cpiStack.put(String("Cycles Group Completion Table Full"),Double(float(handleNone(counters,"PM_GCT_FULL_CYC",j)) / divisor))

	#pct of flushes due to lsu
	cpiStack.put(String("LSU Flush %"),Double(float(handleNone(counters,"PM_LSU_FLUSH",j)) / float(handleNone1(counters,"PM_FLUSH_LSU_BR_MPRED",j))))

	dividend = float(handleNone(counters,"PM_INST_CMPL",j))
	#instructions per lsu flush
	cpiStack.put(String("Instructions per LSU Flush"),Double(dividend / float(handleNone1(counters,"PM_LSU_FLUSH",j))))
	#instructions per lsu flush
	cpiStack.put(String("Instructions per Mispred. Flush"),Double(dividend / float(handleNone1(counters,"PM_FLUSH_BR_MPRED",j))))
	#instructions per lrq flush
	cpiStack.put(String("Instructions per LRQ Flush"),Double(dividend / float(handleNone1(counters,"PM_LSU_FLUSH_LRQ",j))))
	#instructions per srq flush
	cpiStack.put(String("Instructions per SRQ Flush"),Double(dividend / float(handleNone1(counters,"PM_LSU_FLUSH_SRQ",j))))
	# instruction completion %
	cpiStack.put(String("Instruction Completion %"),Double(dividend / float(handleNone1(counters,"PM_INST_DISP",j))))
	# instructions issued per cycle rate
	cpiStack.put(String("Instruction Issue Rate"),Double(float(handleNone(counters,"PM_INST_DISP",j)) / float(handleNone1(counters,"PM_CYC",j))))

	divisor = float(handleNone1(counters,"PM_INST_CMPL",j))
	# FPU % of total instructions
	cpiStack.put(String("FPU Instruction Mix"),Double(float(handleNone(counters,"PM_FPU_FIN",j))/divisor))
	# FPU % of total instructions
	cpiStack.put(String("FPU Long Instruction Mix"),Double(float(handleNone(counters,"PM_FPU_FSQRT",j)+handleNone(counters,"PM_FPU_FDIV",j))/divisor))
	# FXU % of total instructions
	cpiStack.put(String("FXU Instruction Mix"),Double(float(handleNone(counters,"PM_FXU_FIN",j))/divisor))
	# BRU % of total instructions
	cpiStack.put(String("BRU Instruction Mix"),Double(float(handleNone(counters,"PM_BR_ISSUED",j))/divisor))
	# LSU % of total instructions
	cpiStack.put(String("LSU Instruction Mix (guess)"),Double((divisor-float(handleNone(counters,"PM_FPU_FIN",j))-float(handleNone(counters,"PM_FXU_FIN",j))-float(handleNone(counters,"PM_BR_ISSUED",j)))/divisor))
	cpiStack.put(String("LSU Instruction Mix (overestimate)"),Double(float(handleNone(counters,"PM_LD_REF_L1",j)+handleNone(counters,"PM_LD_REF_L1",j))/divisor))
	# Groups completed per instruction
	cpiStack.put(String("Groups per Instruction"),Double(float(handleNone(counters,"PM_GRP_CMPL",j))/divisor))
	
	# branch misprediction rate
	divisor = float(handleNone1(counters,"PM_BR_ISSUED",j))
	cpiStack.put(String("BRU Misprediction %"),Double(1.0-(float(handleNone(counters,"PM_BR_ISSUED",j)-handleNone(counters,"PM_BR_MPRED_CR",j)-handleNone(counters,"PM_BR_MPRED_TA",j))/divisor)))
	
	return cpiStack

#########################################################################################

def processRules(cpiStack):
	global rules
	ruleHarness = RuleHarness.useGlobalRules(rules)
	# have to use assertObject2, for some reason - why is the rule harness instance null?
	ruleHarness.setGlobal("cpiThreshold",Double(cpiThreshold))
	ruleHarness.setGlobal("completionThreshold",Double(completionThreshold))
	ruleHarness.setGlobal("gctThreshold",Double(gctThreshold))
	ruleHarness.setGlobal("stallThreshold",Double(stallThreshold))
	ruleHarness.setGlobal("branchStallThreshold",Double(branchStallThreshold))
	ruleHarness.setGlobal("longFPUStallThreshold",Double(longFPUStallThreshold))
	ruleHarness.setGlobal("flushThreshold",Double(flushThreshold))
	ruleHarness.setGlobal("instPerFlushThreshold1",Double(instPerFlushThreshold1))
	ruleHarness.setGlobal("instPerFlushThreshold2",Double(instPerFlushThreshold2))
	ruleHarness.setGlobal("instMixThreshold",Double(instMixThreshold))
	ruleHarness.setGlobal("longFPUInstMixThreshold",Double(longFPUInstMixThreshold))
	ruleHarness.setGlobal("branchMixThreshold",Double(branchMixThreshold))
	ruleHarness.setGlobal("mispredictionThreshold",Double(mispredictionThreshold))
	ruleHarness.setGlobal("mispredFlushThreshold",Double(mispredFlushThreshold))
	ruleHarness.setGlobal("completionRatioThreshold",Double(completionRatioThreshold))
	ruleHarness.setGlobal("issueRateThreshold",Double(issueRateThreshold))
	fact = FactWrapper("Overall", "CPI Stack", cpiStack)
	handle = ruleHarness.assertObject(fact)
	fact.setFactHandle(handle)
	ruleHarness.processRules()

	return

#########################################################################################

def main(argv):
	print "--------------- JPython test script start ------------"
	getParameters()
	global fractionThreshold
	names, densities, totalDurations, averageDurations, percentDurations, counters, maxClusters = parseCounters(inputData, fractionThreshold)
	# cpiStack = makeTestStack()
	for i in range(1,maxClusters):
		print "\n>>>>>>>>>>>>>>>> Analyzing Stalls for Cluster", i, "<<<<<<<<<<<<<<<<\n"
		cpiStack = computeCPIStats(names, percentDurations, counters, i)
		processRules(cpiStack)
	print "---------------- JPython test script end -------------"

#########################################################################################

if __name__ == "__main__":
	main(sys.argv[1:])

#########################################################################################

