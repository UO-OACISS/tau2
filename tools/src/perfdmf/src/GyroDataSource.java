/* 
   Name:        GyroDataSource.java
   Author:      Kevin Huck
   Description: Parse Gyro Perc data files.  This parser parses output files from the
                ORNL Gyro software.
*/

/*
  To do: 
*/

package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.util.*;

public class GyroDataSource extends DataSource {

    public GyroDataSource(Object initializeObject){
		super();
		this.setMetrics(new Vector<Metric>());
		this.initializeObject = initializeObject;
    }

	private Object initializeObject;

    public void cancelLoad() {
        return;
    }

    public int getProgress() {
        return 0;
    }

    public void load () throws FileNotFoundException, IOException{
		boolean firstFile = true;
    	v = (java.util.List) initializeObject;
		System.out.println(v.size() + " files");
    	for(int index = 0 ; index < v.size() ; index++){
			files = (File[]) v.get(index);
			for (int i = 0 ; i < files.length ; i++) {
				System.out.println("Processing data file, please wait ......");
				long time = System.currentTimeMillis();

				// initialize our data structures
				methodIndexes = new Hashtable<String, Integer>();
				methodNames = new Vector<String>();
				wallTime = new double[20];
				phaseValues = new Hashtable<String, double[]>();

				// get the number of processes
				parseThreadsFromFilename(files[i].getName());

				FileInputStream fileIn = new FileInputStream(files[i]);
				InputStreamReader inReader = new InputStreamReader(fileIn);
				br = new BufferedReader(inReader);

				// increment the node counter - there's a file for each node.
				nodeID++;
     
				// find the statistical data
				boolean processLine = false;
				while((inputString = br.readLine()) != null){
					if (inputString.trim().length() == 0) {
						// do nothing
					} else if (inputString.trim().toUpperCase().startsWith("NL")) {
						// this is the header.  Each instrumented
						// section has a column.
						processHeader(inputString);
					} else if (inputString.trim().startsWith("---------")) {
					} else if (inputString.trim().startsWith("0.000E+00 0.000E+00")) {
						// do nothing
					} else {
						// only process every other line.
						if (processLine)
							processTimers(inputString);
						processLine = processLine ? false : true;
					} 
				}

				//Close the file.
				br.close();

				saveMappings();

				time = (System.currentTimeMillis()) - time;
				System.out.println("Done processing data file!");
				System.out.println("Time to process file (in milliseconds): " + time);
			}
    	}

		//Generate derived data.
		this.generateDerivedData();
    }
    
    //####################################
    //Private Section.
    //####################################

	private void initializeThread() {
		// create the mapping, if necessary
		function = this.addFunction(eventName, 1);

		// make sure we start at zero for all counters
		nodeID = (nodeID == -1) ? 0 : nodeID;
		contextID = (contextID == -1) ? 0 : contextID;
		threadID = (threadID == -1) ? 0 : threadID;

		//Get the node,context,thread.
		node = this.getNode(nodeID);
		if(node==null)
			node = this.addNode(nodeID);
		context = node.getContext(contextID);
		if(context==null)
			context = node.addContext(contextID);
		thread = context.getThread(threadID);
		if(thread==null){
			thread = context.addThread(threadID);
			//thread.setDebug(this.debug());
			//thread.initializeFunctionList(this.getGlobalMapping().getNumberOfMappings(0));
			//thread.initializeUsereventList(this.getGlobalMapping().getNumberOfMappings(2));
		}

		functionProfile = thread.getFunctionProfile(function);
		if(functionProfile == null) {
			functionProfile = new FunctionProfile(function);
			thread.addFunctionProfile(functionProfile);
		}
	}

	private void parseThreadsFromFilename(String string) {
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, ".");
			String str = new String();
			while (st1.hasMoreTokens()) {
				str = st1.nextToken();  // all the equal signs
				try {
					numThreads = Integer.parseInt(str);
				} catch (NumberFormatException e) {
					// not a number, so continue
				}
			}
		} catch(Exception e) {
	    	System.out.println("An error occurred while parsing the header!");
	    	e.printStackTrace();
		}
	}

	private void processHeader(String string) {
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, " ");
			String str = new String();
			int counter = 0;
			while (st1.hasMoreTokens()) {
				str = st1.nextToken();  // all the equal signs
				methodIndexes.put(str, new Integer(counter));
				methodNames.add(str);
				counter++;
			}
		} catch(Exception e) {
	    	System.out.println("An error occurred while parsing the header!");
	    	e.printStackTrace();
		}
	}

	private void processTimers(String string) {
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, " ");
			String str = new String();
			int counter = 0;
			while (st1.hasMoreTokens()) {
				str = st1.nextToken();
				double phaseTotal = 0;
				String name = "Iteration " + phaseCounter;
				try {
					double value = Double.parseDouble(str);
					// the last column is an accumulated value, so don't increment
					// it, just re-set it
					if (counter == 10) {
						wallTime[counter] = value;
					} else {
						// all other columns are aggregates
						wallTime[counter] += value;
					}
					double[] values = new double[2];
					// the first 8 columns are individual events
					if (counter < 8) {
						runningTotal += value;
						phaseTotal += value;
						values[0] = value;
						values[1] = value;
						String tmp = methodNames.elementAt(counter);
						phaseValues.put(new String(name + " => " + tmp), values);
					}
					// the ninth (zero indexed) column is a phase aggregate
					if (counter == 8) {
						runningTotal2 += value;
						// save the value for this phase
						values[0] = value;
						// get the difference between the phase total and our aggregate
						double tmp = value - phaseTotal;
						values[1] = (tmp < 0.0) ? 0.0 : tmp;
						phaseValues.put(new String("RUNTIME => " + name), values);
						phaseValues.put(name, values);
						phaseCounter++;
					}
				} catch (NumberFormatException e) {
					// do nothing
				}
				counter++;
			}
	    	//System.out.println("runningTotal: " + runningTotal);
	    	//System.out.println("runningTotal2: " + runningTotal2);
		} catch(Exception e) {
	    	System.out.println("An error occurred while parsing the header!");
	    	e.printStackTrace();
		}
	}

	private void saveMappings() {
		try{
			Enumeration<String> e = methodIndexes.keys();
			while (e.hasMoreElements()) {
				eventName = e.nextElement();
				if (!saveMappingsInner(false))
					continue;
			}
			e = phaseValues.keys();
			while (e.hasMoreElements()) {
				eventName = e.nextElement();
				if (!saveMappingsInner(true))
					continue;
			}
		} catch(Exception e) {
	    	System.out.println("An error occurred while parsing the callsite data!");
	    	e.printStackTrace();
		}
    }

	private boolean saveMappingsInner(boolean doingPhases) throws Exception {
		Integer index = null;
		if (!doingPhases)
			index = methodIndexes.get(eventName);
		boolean inclusiveEqualsExclusive = true;

		if (eventName.toUpperCase().equals("STEP"))
			return false;
		if (eventName.toUpperCase().equals("ELAPSED"))
			return false;
		if (eventName.toUpperCase().equals("RUNTIME"))
			inclusiveEqualsExclusive = false;

		for (int i = 0; i < numThreads ; i++) {
			threadID = i;
			// make sure we have a mapping for this event
			initializeThread();
			// save the first metric
			if (!doingPhases) {
				if (inclusiveEqualsExclusive) {
					saveMappingData ("Time", (wallTime[index.intValue()]), (wallTime[index.intValue()]));
				} else {
					double tmpVal = wallTime[index.intValue()] - (runningTotal);
					tmpVal = tmpVal < 0 ? 0 : tmpVal;
					saveMappingData ("Time", (wallTime[index.intValue()]), tmpVal);
				}
			} else {
				double[] values = (phaseValues.get(eventName));
				saveMappingData ("Time", values[0], values[1]);
			}
			// save the data common to all metrics
			functionProfile.setNumCalls(1);
			if (!eventName.toUpperCase().equals("RUNTIME"))
				functionProfile.setNumSubr(0);
			else {
				functionProfile.setNumSubr(8);
				function.addGroup(this.addGroup("TAU_PHASE"));
			}

			if (doingPhases) {
				if (eventName.indexOf("=>") == -1)
					function.addGroup(this.addGroup("TAU_PHASE"));
				else
					function.addGroup(this.addGroup("TAU_CALLPATH"));
			} else {
				// add the group
				Group group = null;
				if (eventName.toUpperCase().endsWith("_TR"))
					group = this.addGroup("TRANSPOSE");
				else
					group = this.addGroup("CALCULATION");
				function.addGroup(group); 
			}
		}
		return true;
	}

    private void saveMappingData(String metricName, double inclusiveValue, double exclusiveValue){
		metric = this.getNumberOfMetrics();
		//Set the metric name.
		Metric newMetric = this.addMetric(metricName);
		metric = newMetric.getID();
		functionProfile.setExclusive(metric, exclusiveValue*1000000);
		functionProfile.setInclusive(metric, inclusiveValue*1000000);
		//functionProfile.setInclusivePerCall(metric, inclusiveValue);
	}

    //####################################
    //End - Private Section.
    //####################################

	//######
	//Frequently used items.
	//######
	private int metric = 0;
	private Function function = null;
	private FunctionProfile functionProfile = null;
	private Node node = null;
	private Context context = null;
	private edu.uoregon.tau.perfdmf.Thread thread = null;
	private int nodeID = -1;
	private int contextID = -1;
	private int threadID = -1;
	private String inputString = null;
	private String s1 = null;
	private String s2 = null;
	private String tokenString;
	private String groupNamesString = null;
	private StringTokenizer genericTokenizer;
	private int mappingID = -1;
	private java.util.List v = null;
	private File[] files = null;
	private BufferedReader br = null;
	private Hashtable<String, Integer> methodIndexes = null;
	private Vector<String> methodNames = null;
	private double wallTime[] = null;
	private String eventName = null;
	private double runningTotal = 0.0;
	private double runningTotal2 = 0.0;
	private int phaseCounter = 0;
	private Hashtable<String, double[]> phaseValues = null;
	private int numThreads = 1;
	//######
	//End - Frequently used items.
	//######

    //####################################
    //Instance data.
    //####################################
    private LineData lineData = new LineData();
    //####################################
    //End - Instance data.
    //####################################
}
