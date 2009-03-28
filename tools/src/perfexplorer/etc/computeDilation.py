from edu.uoregon.tau.perfexplorer.client import *
from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *

import time

True = 1
False = 0
inApp = "FACETS"
inExp = "FACETS Sigma Regression"
inTrial = ""

def load():
	print "loading data..."
	parameterMap = PerfExplorerModel.getModel().getScriptParameters()
	keys = parameterMap.keySet()
	for key in keys:
		print key, parameterMap.get(key)
	config = parameterMap.get("config")
	inApp = parameterMap.get("app")
	inExp = parameterMap.get("exp")
	inTrial = parameterMap.get("trial")
	Utilities.setSession(config)
	#trial = Utilities.getTrial("s3d", "intrepid-c2h4-spacemap", "1728")
	trial = Utilities.getTrial(inApp, inExp, inTrial)
	print "...done."
	return trial

def computeDilation(networkSize, senderCoords, receiverCoords):
	tokens = networkSize.replace('(','').replace(')','').split(',')
	maxX = int(tokens[0])
	maxY = int(tokens[1])
	maxZ = int(tokens[2])
	tokens = senderCoords.replace('(','').replace(')','').split(',')
	sendX = int(tokens[0])
	sendY = int(tokens[1])
	sendZ = int(tokens[2])
	tokens = receiverCoords.replace('(','').replace(')','').split(',')
	recvX = int(tokens[0])
	recvY = int(tokens[1])
	recvZ = int(tokens[2])

	distX = abs(sendX-recvX)
	distX = min(distX, maxX-distX)

	distY = abs(sendY-recvY)
	distY = min(distY, maxY-distY)

	distZ = abs(sendZ-recvZ)
	distZ = min(distZ, maxZ-distZ)

	return distX+distY+distZ

print "--------------- JPython test script start ------------"
trial = load()
start = time.clock()
print "getting thread metadata"
metadata = TrialThreadMetadata(trial)
print "getting common metadata"
commonMetadata = TrialMetadata(trial)
end = time.clock()
print "metadata time:", end-start, "seconds"

networkSize = commonMetadata.getCommonAttributes().get("BGP Size")
tauConfig =  commonMetadata.getCommonAttributes().get("TAU Config")
if tauConfig != None:
	haveSendData = (tauConfig.find("TAU_EACH_SEND") > 0)
else:
	haveSendData = False

dilation = 0
worst = 0
count = 0

start = time.clock()
if haveSendData:
	print "TAU_EACH_SEND data found."
	start = time.clock()
	input = TrialResult(trial)
	end = time.clock()
	print "loading time:", end-start, "seconds"
	for thread in input.getThreads():
		for event in input.getUserEvents():
			senderCoords = metadata.getNameValue(thread, "BGP Coords")
			if event.startswith("Message size sent to node "):
				# split the string
				st = StringTokenizer(event, "Message size sent to node ")
				if st.hasMoreTokens():
 					receiver = int(st.nextToken())
					if (input.getDataPoint(thread,event,None,AbstractResult.USEREVENT_NUMEVENTS) > 0):
						receiverCoords = metadata.getNameValue(receiver, "BGP Coords")
						#print networkSize, senderCoords, receiverCoords
						dilation = dilation + computeDilation(networkSize, senderCoords, receiverCoords)
						count = count + 1
else:
	totalThreads = metadata.getThreadCount()
	if totalThreads == 8:
		maxX = maxY = maxZ = 2
	if totalThreads == 64:
		maxX = maxY = maxZ = 4
	if totalThreads == 512:
		maxX = maxY = maxZ = 8
	if totalThreads == 1728:
		maxX = maxY = maxZ = 12
	if totalThreads == 4096:
		maxX = maxY = maxZ = 16
	if totalThreads == 6400:
		maxX = maxY = 20
		maxZ = 16
	if totalThreads == 8000:
		maxX = maxY = maxZ = 20
	if totalThreads == 12000:
		maxX = 30
		maxY = maxZ = 20
	if totalThreads == 30000:
		maxX = 40
		maxY = 30
		maxZ = 25
	
	data = ChartData();
	data.addRow("Dilation");
	for i in range(0,maxX):
		for j in range(0,maxY):
			for k in range(0,maxZ):
				myAddress = (k * maxY * maxX) + (j * maxX) + i
				up = (k * maxY * maxX) + (((j+maxY-1)%maxY) * maxX) + i
				down = (k * maxY * maxX) + (((j+1)%maxY) * maxX) + i
				left = (k * maxY * maxX) + (j * maxX) + ((i+maxX-1)%maxX)
				right = (k * maxY * maxX) + (j * maxX) + ((i+1)%maxX)
				back = (((k+maxZ-1)%maxZ) * maxY * maxX) + (j * maxX) + i
				front = (((k+1)%maxZ) * maxY * maxX) + (j * maxX) + i
				myNeighbors = [up,down,left,right,back,front]
				for receiver in myNeighbors:
					senderCoords = metadata.getNameValue(myAddress, "BGP Coords")
					receiverCoords = metadata.getNameValue(receiver, "BGP Coords")
					#print networkSize, senderCoords, receiverCoords
					dil = computeDilation(networkSize, senderCoords, receiverCoords)
					if dil > worst:
						worst = dil
					dilation = dilation + dil
					count = count + 1
				data.addColumn(0, dilation/6.0, dilation/6.0)

end = time.clock()
print "computation time:", end-start, "seconds"
				
avgDilation = float(dilation) / float(count)
print avgDilation, worst
PerfExplorerHistogramChart.doHistogram(data);
	
print "---------------- JPython test script end -------------"
