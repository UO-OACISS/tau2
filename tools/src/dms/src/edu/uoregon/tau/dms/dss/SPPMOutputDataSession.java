/* 
   Name:        SPPMOutputDataSession.java
   Author:      Kevin Huck
   Description: Parse sPPM data files.  This parser parses output files from the
                LLNL Purple Benchmark sPPM software.
*/

/*
  To do: 
*/

package edu.uoregon.tau.dms.dss;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.io.*;
import java.util.*;

public class SPPMOutputDataSession extends ParaProfDataSession{

    public SPPMOutputDataSession(){
		super();
		this.setMetrics(new Vector());
    }

    public void run(){
		boolean firstFile = true;
		try{
	    	v = (Vector) initializeObject;
			System.out.println(v.size() + " files");
	    	for(Enumeration e = v.elements(); e.hasMoreElements() ;){
				files = (File[]) e.nextElement();
				for (int i = 0 ; i < files.length ; i++) {
					System.out.println("Processing data file, please wait ......");
					long time = System.currentTimeMillis();

					// initialize our data structures
					methodIndexes = new Hashtable();
					cpuTime = new double[20];
					wallTime = new double[20];
					calls = new int[20];
					subroutines = new int[20];

					// reset the counters
					deltaCount = 0;
					timestepCount = 0;

					FileInputStream fileIn = new FileInputStream(files[i]);
					InputStreamReader inReader = new InputStreamReader(fileIn);
					br = new BufferedReader(inReader);

					// increment the node counter - there's a file for each node.
					nodeID++;
      
					//####################################
					//First Line
					//####################################
					//This line is not required. Check to make sure that it is there however.
					inputString = br.readLine();
					if(inputString == null)
			    		return;
					//####################################
					//End - First Line
					//####################################
      
					// find the statistical data
					while((inputString = br.readLine()) != null){
						if (inputString.length() == 0) {
							// do nothing
						} else if (inputString.trim().startsWith("==================> Begin Double Timestep")) {
							// this is the beginning of a timestep.  We will get some data in the 
							// next couple of lines.  For now, get the timestep index.
							processTimestepHeader(inputString);
						} else if (inputString.trim().indexOf("threads update a") >= 0) {
							processThreadCount(inputString);
						} else if (inputString.trim().startsWith("DELTA-HYD cpu, wall, ratio:")) {
							processEvent(inputString, 0);
							deltaCount++;
						} else if (inputString.trim().startsWith("TSTEP-HYD cpu, wall, ratio:")) {
							processEvent(inputString, deltaCount);
							timestepCount++;
							deltaCount = 0;
						} else if (inputString.trim().startsWith("TOTAL-HYD cpu, wall, ratio:") && inputString.trim().indexOf("Finished Calculation") >= 0) {
							processEvent(inputString, timestepCount);
						} else {
							// do nothing
						} 
					}

					//Close the file.
					br.close();

					saveMappings();

					if(UtilFncs.debug){
			    		System.out.println("The total number of threads is: " + this.getNCT().getTotalNumberOfThreads());
			    		System.out.println("The number of mappings is: " + this.getGlobalMapping().getNumberOfMappings(0));
		    			System.out.println("The number of user events is: " + this.getGlobalMapping().getNumberOfMappings(2));
					}

					//Set firstRead to false.
					this.setFirstMetric(false);

					time = (System.currentTimeMillis()) - time;
					System.out.println("Done processing data file!");
					System.out.println("Time to process file (in milliseconds): " + time);
				}
	    	}

		//Generate derived data.
		this.generateDerivedData(0);
		//Remove after testing is complete.
		//this.setMeanDataAllMetrics(0);	
			    
	    	//Need to notify observers that we are done.  Be careful here.
	    	//It is likely that they will modify swing elements.  Make sure
	    	//to dump request onto the event dispatch thread to ensure
	    	//safe update of said swing elements.  Remember, swing is not thread
	    	//safe for the most part.
	    	EventQueue.invokeLater(new Runnable(){
		    	public void run(){
					SPPMOutputDataSession.this.notifyObservers();
		    	}
			});
		} catch(Exception e) {
	    	UtilFncs.systemError(e, null, "SSD01");
		}
    }
    
    //####################################
    //Private Section.
    //####################################

	private void initializeThread() {
		// create the mapping, if necessary
		mappingID = this.getGlobalMapping().addGlobalMapping(eventName, 0, 1);
		globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);

		// make sure we start at zero for all counters
		nodeID = (nodeID == -1) ? 0 : nodeID;
		contextID = (contextID == -1) ? 0 : contextID;
		threadID = (threadID == -1) ? 0 : threadID;

		//Get the node,context,thread.
		node = this.getNCT().getNode(nodeID);
		if(node==null)
			node = this.getNCT().addNode(nodeID);
		context = node.getContext(contextID);
		if(context==null)
			context = node.addContext(contextID);
		thread = context.getThread(threadID);
		if(thread==null){
			thread = context.addThread(threadID);
			thread.setDebug(this.debug());
			thread.initializeFunctionList(this.getGlobalMapping().getNumberOfMappings(0));
			thread.initializeUsereventList(this.getGlobalMapping().getNumberOfMappings(2));
		}

		globalThreadDataElement = thread.getFunction(mappingID);
		if(globalThreadDataElement == null) {
			globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(mappingID, 0), false);
			thread.addFunction(globalThreadDataElement, mappingID);
		}
	}

	private void processTimestepHeader(String string) {
		// System.out.print("Beginning of timestep: ");
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, " ");

			// get the first name/value pair
			string = st1.nextToken();  // all the equal signs
			string = st1.nextToken();  // Begin
			string = st1.nextToken();  // Double 
			string = st1.nextToken();  // Timestep
			// get the value
	    	lineData.i0 = Integer.parseInt(st1.nextToken().trim()); // timestep ID
	    	// System.out.println (lineData.i0);
		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the header!");
	    	e.printStackTrace();
		}
	}

	private void processThreadCount(String string) {
		// System.out.print("Thread Count: ");
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, " ");
			// get the first value
	    	lineData.i1 = Integer.parseInt(st1.nextToken().trim()); // thread count
	    	// System.out.println (lineData.i1);
		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the header!");
	    	e.printStackTrace();
		}
	} 

	private void processEvent(String string, int subroutineCount) {
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, " ");
	    	lineData.s0 = st1.nextToken().trim(); // procedure name
	    	lineData.s1 = st1.nextToken().trim(); // first metric
			lineData.s1 = lineData.s1.replaceAll(",",""); // remove the trailing comma
	    	lineData.s2 = st1.nextToken().trim(); // second metric
			lineData.s2 = lineData.s2.replaceAll(",",""); // remove the trailing comma
	    	lineData.s3 = st1.nextToken().trim(); // third metric
			lineData.s3 = lineData.s3.replaceAll(":",""); // remove the trailing colon
			lineData.d0 = Double.parseDouble(st1.nextToken().trim()); // first metric value
			lineData.d0 = lineData.d0 / lineData.i1;  // divde by #threads
			lineData.d1 = Double.parseDouble(st1.nextToken().trim()); // second metric value
			lineData.d2 = Double.parseDouble(st1.nextToken().trim()); // third metric value
			while (st1.hasMoreTokens()) {
				String tmpToken = st1.nextToken().trim();
				if (tmpToken.equals("@")) // don't add the clock time
					break;
	    		lineData.s0 += " " + tmpToken; // add to procedure name
			}

			boolean inclusiveEqualsExclusive = false;
			if (subroutineCount == 0)
				inclusiveEqualsExclusive = true;

			Integer index = (Integer)methodIndexes.get(lineData.s0);
			if (index == null) {
				index = new Integer(methodIndexes.size());
				methodIndexes.put(lineData.s0, index);
				cpuTime[index.intValue()] = lineData.d0;
				wallTime[index.intValue()] = lineData.d1;
				calls[index.intValue()] = 1;
				subroutines[index.intValue()] = subroutineCount;
			} else {
				cpuTime[index.intValue()] += lineData.d0;
				wallTime[index.intValue()] += lineData.d1;
				calls[index.intValue()] ++;
				subroutines[index.intValue()] = subroutineCount;
			}
		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the callsite data!");
	    	e.printStackTrace();
		}
    }

	private void saveMappings() {
		try{
			Enumeration e = methodIndexes.keys();
			while (e.hasMoreElements()) {
				eventName = (String)e.nextElement();
				Integer index = (Integer)methodIndexes.get(eventName);
				boolean inclusiveEqualsExclusive = false;
				if (subroutines[index.intValue()] == 0)
					inclusiveEqualsExclusive = true;

				for (int i = 0; i < lineData.i1 ; i++) {
					threadID = i;
					// make sure we have a mapping for this event
					initializeThread();
					// save the first metric
					saveMappingData ("cpu", cpuTime[index.intValue()], inclusiveEqualsExclusive);
					// increment the storage to allow for second metric
					thread.incrementStorage();
					globalMappingElement.incrementStorage();
					globalThreadDataElement.incrementStorage();
					// save the second metric
					saveMappingData ("wall", wallTime[index.intValue()], inclusiveEqualsExclusive);
					// save the data common to all metrics
					globalThreadDataElement.setNumberOfCalls(calls[index.intValue()]);
					if (globalMappingElement.getMaxNumberOfCalls() < calls[index.intValue()])
						globalMappingElement.setMaxNumberOfCalls(calls[index.intValue()]);
					globalThreadDataElement.setNumberOfSubRoutines(subroutines[index.intValue()]);
					if (globalMappingElement.getMaxNumberOfSubRoutines() < subroutines[index.intValue()])
						globalMappingElement.setMaxNumberOfSubRoutines(subroutines[index.intValue()]);
				}
			}
		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the callsite data!");
	    	e.printStackTrace();
		}
    }

    private void saveMappingData(String metricName, double value, boolean inclusiveEqualsExclusive){
		metric = this.getNumberOfMetrics();
		//Set the metric name.
		Metric newMetric = this.addMetric(metricName);
		if (metric < this.getNumberOfMetrics()) {
			//Need to call increaseVectorStorage() on all objects that require it.
			this.getGlobalMapping().increaseVectorStorage();
		}
		metric = newMetric.getID();

		if (inclusiveEqualsExclusive) {
			globalThreadDataElement.setExclusiveValue(metric, value);
			if((globalMappingElement.getMaxExclusiveValue(metric)) < value) {
				globalMappingElement.setMaxExclusiveValue(metric, value);
			}
		} else {
			globalThreadDataElement.setExclusiveValue(metric, 0.0);
		}

		globalThreadDataElement.setInclusiveValue(metric, value);
		if((globalMappingElement.getMaxInclusiveValue(metric)) < value) {
			globalMappingElement.setMaxInclusiveValue(metric, value);
		}

		globalThreadDataElement.setUserSecPerCall(metric, value);
		if(globalMappingElement.getMaxUserSecPerCall(metric) < value)
			globalMappingElement.setMaxUserSecPerCall(metric, value);

		//globalThreadDataElement.setExclusivePercentValue(metric, 0);
		//globalThreadDataElement.setInclusivePercentValue(metric, 0);
		//globalMappingElement.setMaxExclusivePercentValue(metric, 0.0);
		//globalMappingElement.setMaxInclusivePercentValue(metric, 0.0);
	}

    //####################################
    //End - Private Section.
    //####################################

	//######
	//Frequently used items.
	//######
	private int metric = 0;
	private GlobalMappingElement globalMappingElement = null;
	private GlobalThreadDataElement globalThreadDataElement = null;
	private Node node = null;
	private Context context = null;
	private edu.uoregon.tau.dms.dss.Thread thread = null;
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
	private Vector v = null;
	private File[] files = null;
	private BufferedReader br = null;
	private int deltaCount = 0;
	private int timestepCount = 0;
	private Hashtable methodIndexes = null;
	private double cpuTime[] = null;
	private double wallTime[] = null;
	private int calls[] = null;
	private int subroutines[] = null;
	private String eventName = null;
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
