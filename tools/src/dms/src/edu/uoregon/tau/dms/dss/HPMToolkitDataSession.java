/* 
   Name:		HPMToolkitDataSession.java
   Author:	  Kevin Huck
   Description: Parse an mpiP data file.
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

public class HPMToolkitDataSession extends ParaProfDataSession{

    public HPMToolkitDataSession(){
	super();
	this.setMetrics(new Vector());
    }

    public void initialize(Object initializeObject){
	    
	boolean firstFile = true;
	try{
	    v = (Vector) initializeObject;
	    for(Enumeration e = v.elements(); e.hasMoreElements() ;){
		files = (File[]) e.nextElement();
		for (int i = 0 ; i < files.length ; i++) {
		    System.out.println("Processing data file, please wait ......");
		    long time = System.currentTimeMillis();

		    FileInputStream fileIn = new FileInputStream(files[i]);
		    InputStreamReader inReader = new InputStreamReader(fileIn);
		    br = new BufferedReader(inReader);

		    // increment the node counter - there's a file for each node.
		    nodeID++;

		    // clear the event names
		    eventNames = new Hashtable();
	  
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
	  
		    // find the end of the resource statistics
		    while((inputString = br.readLine()) != null){
			if (inputString.trim().startsWith("Instrumented section:")) {
			    // System.out.println("Found beginning of data: ");
			    // exit this while loop
			    break;
			}
		    }

		    boolean first = true; // we are currently defining the boundaries between
		                          // instrumented sections to be the line "Instrumented section:"
		                          // so we copy inclusive -> exclusive at this time, but we don't
		                          // want to do it the first time.
		    boolean exclusiveSet = false;
		    // find the callsite data
		    while(inputString != null){
			if (inputString.length() == 0) {
			    // do nothing
			} else if (inputString.trim().startsWith("Instrumented section:")) {

			    if (!first && !exclusiveSet) {
				// there was no exclusive time for this instrumented section, this means
				// that the exclusive time = inclusive time (it did not call any 
				// instrumented functions.
				// copy inclusive over to exclusive
				globalThreadDataElement.setExclusiveValue(timeMetric, globalThreadDataElement.getInclusiveValue(timeMetric));
				if (globalThreadDataElement.getExclusiveValue(timeMetric) > globalMappingElement.getMaxExclusiveValue(timeMetric)) {
				    globalMappingElement.setMaxExclusiveValue(timeMetric, globalThreadDataElement.getExclusiveValue(timeMetric));
				}
			    }
			    first = false;
			    exclusiveSet = false;
			    processHeaderLine1(inputString);
			} else if (inputString.trim().startsWith("file:")) {
			    processHeaderLine2(inputString);
			} else if (inputString.trim().startsWith("Count:")) {
			    processHeaderLine3(inputString);
			} else if (inputString.trim().startsWith("Wall Clock Time:")) {
			    //processHardwareCounter(inputString);
			    processTime(inputString, true);
			} else if (inputString.trim().startsWith("Average duration:")) {
				// because I can't figure out how to get the metrics allocated
				// correctly, just toss this value out.  I don't have this metric
				// for every measurement, so the metric indexing gets all hosed up.
				// - kevin, Nov 15, 2004
			    // processHardwareCounter(inputString);
			    processHeaderLine4(inputString, 1);
			} else if (inputString.trim().startsWith("Standard deviation:")) {
				// because I can't figure out how to get the metrics allocated
				// correctly, just toss this value out.  I don't have this metric
				// for every measurement, so the metric indexing gets all hosed up.
			    // processHardwareCounter(inputString);
			    processHeaderLine4(inputString, 2);
			} else if (inputString.trim().startsWith("Exclusive duration:")) {
				// because I can't figure out how to get the metrics allocated
				// correctly, just toss this value out.  I don't have this metric
				// for every measurement, so the metric indexing gets all hosed up.
			    // processHardwareCounter(inputString);
			    //processHeaderLine4(inputString, 3);
			    processTime(inputString, false);
			    exclusiveSet = true;
			} else if (inputString.trim().startsWith("Total time in user mode:")) {
			    processHardwareCounter(inputString);
			} else {
			    processHardwareCounter(inputString);
			} 
			inputString = br.readLine();
		    }

		  
		    // we also have to do this here to get the last instrumented section
		    if (!first && !exclusiveSet) {
			// there was no exclusive time for this instrumented section, this means
			// that the exclusive time = inclusive time (it did not call any 
			// instrumented functions.
			// copy inclusive over to exclusive
			globalThreadDataElement.setExclusiveValue(timeMetric, globalThreadDataElement.getInclusiveValue(timeMetric));
			if (globalThreadDataElement.getExclusiveValue(timeMetric) > globalMappingElement.getMaxExclusiveValue(timeMetric)) {
			    globalMappingElement.setMaxExclusiveValue(timeMetric, globalThreadDataElement.getExclusiveValue(timeMetric));
			    
			}
		    }

		    //Close the file.
		    br.close();

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
		
	} catch(Exception e) {
	    UtilFncs.systemError(e, null, "SSD01");
	}
    }
	
    //####################################
    //Private Section.
    //####################################

    private void initializeThread() {
	// create the mapping, if necessary
	if (header2.s1 == null)
	    mappingID = this.getGlobalMapping().addGlobalMapping(header1.s0 + ", " + header2.s0, 0, 1);
	else
	    mappingID = this.getGlobalMapping().addGlobalMapping(header1.s0 + ", " + header2.s0 + " lines " +header2.s1, 0, 1);
	globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);
	// System.out.println("**** " + header1.s0 + ", " + header2.s0 + " lines " +header2.s1 + " " + threadID + " ****");
	
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
	
	initialized = true;
    }

    private void processHeaderLine1(String string) {
	// System.out.println("Header line 1");
	try{
	    StringTokenizer st1 = new StringTokenizer(string, "-");

	    // get the first name/value pair
	    string = st1.nextToken();
	    StringTokenizer st2 = new StringTokenizer(string, ":");
	    // ignore the "instrumented section" label
	    string = st2.nextToken();
	    // get the value
	    header1.i0 = Integer.parseInt(st2.nextToken().trim()); // section id

	    // get the next name/value pair
	    string = st1.nextToken();
	    st2 = new StringTokenizer(string, ":");
	    // ignore the "Label" label
	    string = st2.nextToken();
	    // get the value
	    header1.s0 = st2.nextToken().trim(); // label value

	    // get the next name/value pair
	    string = st1.nextToken();
	    st2 = new StringTokenizer(string, ":");
	    // ignore the "process" label
	    string = st2.nextToken();
	    // get the value
	    header1.i1 = Integer.parseInt(st2.nextToken().trim()); // process id

	    // if this label is the same as the previous, increment the thread ID
	    // otherwise, reset the thread ID to 0.
	    if (eventNames.containsKey(header1.s0)) {
		Integer tmpID = (Integer)eventNames.get(header1.s0);
		threadID = tmpID.intValue();
		threadID++;
		eventNames.put(header1.s0, new Integer(threadID));
	    } else {
		threadID = 0;
		eventNames.put(header1.s0, new Integer(threadID));
	    }
	    initialized = false;
	} catch(Exception e) {
	    System.out.println("An error occurred while parsing the header!");
	    e.printStackTrace();
	}
    }

    private void processHeaderLine2(String string) {
	// System.out.println("Header line 2");
	header2.s0 = null;
	header2.s1 = null;
	try{
	    StringTokenizer st1 = new StringTokenizer(string, ",");

	    // get the first name/value pair
	    string = st1.nextToken();
	    StringTokenizer st2 = new StringTokenizer(string, ":");
	    // ignore the "file" label
	    string = st2.nextToken();
	    // get the value
	    header2.s0 = st2.nextToken().trim(); // section id

	    // get the next name/value pair
	    string = st1.nextToken();
	    st2 = new StringTokenizer(string, ":");
	    // ignore the "lines" label
	    string = st2.nextToken();
	    // get the value
	    header2.s1 = st2.nextToken().trim(); // label value
	} catch(Exception e) {
	    System.out.println("An error occurred while parsing the header!");
	    e.printStackTrace();
	}
    }

    private void processHeaderLine3(String string) {
	// System.out.println("Header line 3");
	try{
	    StringTokenizer st1 = new StringTokenizer(string, ":");
	    // ignore the "count" label
	    string = st1.nextToken();
	    // get the value
	    header3.i0 = Integer.parseInt(st1.nextToken().trim()); // section id
	} catch(Exception e) {
	    System.out.println("An error occurred while parsing the header!");
	    e.printStackTrace();
	}
    }

    private void processHeaderLine4(String string, int variable) {
	// System.out.println("Header line 4");
	try{
	    StringTokenizer st1 = new StringTokenizer(string, ":");
	    // ignore the "Wall Clock Time" label
	    string = st1.nextToken();
	    // get the value
	    string = st1.nextToken();
	    StringTokenizer st2 = new StringTokenizer(string, " ");
	    if (variable == 0) header4.d0 = Double.parseDouble(st2.nextToken()); // section id
	    else if (variable == 1) header4.d1 = Double.parseDouble(st2.nextToken()); // section id
	    else if (variable == 2) header4.d2 = Double.parseDouble(st2.nextToken()); // section id
	    else if (variable == 3) header4.d3 = Double.parseDouble(st2.nextToken()); // section id
	} catch(Exception e) {
	    System.out.println("An error occurred while parsing the header!");
	    e.printStackTrace();
	}
    }

    private void processHeaderLine5(String string) {
	// System.out.println("Header line 5");
	try{
	    StringTokenizer st1 = new StringTokenizer(string, ":");
	    // ignore the "Total time in user mode" label
	    string = st1.nextToken();
	    // get the value
	    string = st1.nextToken();
	    StringTokenizer st2 = new StringTokenizer(string, " ");
	    header5.d0 = Double.parseDouble(st2.nextToken()); // section id
	} catch(Exception e) {
	    System.out.println("An error occurred while parsing the header!");
	    e.printStackTrace();
	}
	initializeThread();
    }

    private void processHardwareCounter(String string) {
	if (!initialized) {
	    if (header2.s0 == null)
		header2.s0 = new String("Entire Program");
	    if (header3.i0 == 0)
		header3.i0 = 1;
	    initializeThread();
	} else {
	    // thread.incrementStorage();
	    globalMappingElement.incrementStorage();
	    globalThreadDataElement.incrementStorage();
	}
	// System.out.println("Hardwoare counter");
	try{
	    StringTokenizer st1 = new StringTokenizer(string, ":");
	    String metricName = st1.nextToken().trim(); // hardware counter name
	    String tmpStr = st1.nextToken().trim();
	    int metricCount = 0, newMetricCount = 0;
	    // need to clean stuff out of the value, like % and M and whatnot
	    st1 = new StringTokenizer(tmpStr, " ");
	    double dEventValue = 0.0;
	    int iEventValue = 0;
	    tmpStr = st1.nextToken().trim();
	    boolean typeDouble = false;
	    //		if (tmpStr.indexOf(".") > -1) {
	    
	    try {
		dEventValue = Double.parseDouble(tmpStr); // callsite index
	    } catch (NumberFormatException e) {
		dEventValue = 0;
	    }

	    typeDouble = true;
	    //		} else {
	    //		iEventValue = Integer.parseInt(tmpStr); // callsite index
	    //		}
	    if (st1.hasMoreTokens())
		metricName += " (" + st1.nextToken() + ")";
	    // System.out.println(metricName + " = " + dEventValue);

	    metricCount = this.getNumberOfMetrics();
	    //Set the metric name.
	    Metric newMetric = this.addMetric(metricName);
	    metric = newMetric.getID();
	    newMetricCount = this.getNumberOfMetrics();

	    // System.out.println("" + metricCount + ", " + newMetricCount + ", " + (metric+1));
	    while (metricCount < newMetricCount) {
		//Need to call increaseVectorStorage() on all objects that require it.
		this.getGlobalMapping().increaseVectorStorage();
		metricCount++;
	    }
	    while (thread.getNumberOfMetrics() < newMetricCount) {
		thread.incrementStorage();
	    }
	    while (globalMappingElement.getStorageSize() < newMetricCount) {
		globalMappingElement.incrementStorage();
	    }
	    while (globalThreadDataElement.getStorageSize() < newMetricCount) {
		globalThreadDataElement.incrementStorage();
	    }
	    // new code
	    if (typeDouble) {
		// System.out.println("\t" + metricName + " " + metric);
		globalThreadDataElement.setExclusiveValue(metric, dEventValue);
		globalThreadDataElement.setInclusiveValue(metric, dEventValue);
		double tmpValue = dEventValue / ((double)(header3.i0));
		globalThreadDataElement.setUserSecPerCall(metric, tmpValue);
		if((globalMappingElement.getMaxExclusiveValue(metric)) < dEventValue) {
		    globalMappingElement.setMaxExclusiveValue(metric, dEventValue);
		    globalMappingElement.setMaxInclusiveValue(metric, dEventValue);
		}
		if(globalMappingElement.getMaxUserSecPerCall(metric) < (dEventValue / header3.i0))
		    globalMappingElement.setMaxUserSecPerCall(metric, (dEventValue / header3.i0));
	    } else {
		globalThreadDataElement.setExclusiveValue(metric, iEventValue);
		globalThreadDataElement.setInclusiveValue(metric, iEventValue);
		double tmpValue = iEventValue / ((double)(header3.i0));
		globalThreadDataElement.setUserSecPerCall(metric, tmpValue);
		if((globalMappingElement.getMaxExclusiveValue(metric)) < iEventValue) {
		    globalMappingElement.setMaxExclusiveValue(metric, iEventValue);
		    globalMappingElement.setMaxInclusiveValue(metric, iEventValue);
		}
		if(globalMappingElement.getMaxUserSecPerCall(metric) < (iEventValue / header3.i0))
		    globalMappingElement.setMaxUserSecPerCall(metric, (iEventValue / header3.i0));
	    }
	    globalThreadDataElement.setExclusivePercentValue(metric, 0);
	    globalThreadDataElement.setInclusivePercentValue(metric, 0);
	    globalThreadDataElement.setNumberOfCalls(header3.i0);
	    globalThreadDataElement.setNumberOfSubRoutines(0);
	    globalMappingElement.setMaxExclusivePercentValue(metric, 0.0);
	    globalMappingElement.setMaxInclusivePercentValue(metric, 0.0);
	    if(globalMappingElement.getMaxNumberOfCalls() < header3.i0)
		globalMappingElement.setMaxNumberOfCalls(header3.i0);
	    globalMappingElement.setMaxNumberOfSubRoutines(0);

	} catch(Exception e) {
	    System.out.println("An error occurred while parsing the callsite data!");
	    e.printStackTrace();
	}
    }

    // Exclusive Duration lines occurr when the instrumented section has a child
    // instrumented section.

    // inclusive = false means we are processing an 'exclusive' line
    private void processTime(String string, boolean inclusive) {
	if (!initialized) {
	    if (header2.s0 == null)
		header2.s0 = new String("Entire Program");
	    if (header3.i0 == 0)
		header3.i0 = 1;
	    initializeThread();
	} else {
	    // thread.incrementStorage();
	    globalMappingElement.incrementStorage();
	    globalThreadDataElement.incrementStorage();
	}
	try {
	    StringTokenizer st1 = new StringTokenizer(string, ":");
	    String metricName = st1.nextToken().trim(); // hardware counter name

	    metricName = "Time";

	    String tmpStr = st1.nextToken().trim();
	    int metricCount = 0, newMetricCount = 0;
	    // need to clean stuff out of the value, like % and M and whatnot
	    st1 = new StringTokenizer(tmpStr, " ");
	    double dEventValue = 0.0;
	    int iEventValue = 0;
	    tmpStr = st1.nextToken().trim();
	    
	    try {
		dEventValue = Double.parseDouble(tmpStr); // callsite index
	    } catch (NumberFormatException e) {
		dEventValue = 0;
	    }


	    //if (st1.hasMoreTokens())
	    //metricName += " (" + st1.nextToken() + ")";
	    // System.out.println(metricName + " = " + dEventValue);

	    metricCount = this.getNumberOfMetrics();
	    //Set the metric name.
	    Metric newMetric = this.addMetric(metricName);
	    metric = newMetric.getID();
	    timeMetric = metric;
	    newMetricCount = this.getNumberOfMetrics();

	    // System.out.println("" + metricCount + ", " + newMetricCount + ", " + (metric+1));
	    while (metricCount < newMetricCount) {
		//Need to call increaseVectorStorage() on all objects that require it.
		this.getGlobalMapping().increaseVectorStorage();
		metricCount++;
	    }
	    while (thread.getNumberOfMetrics() < newMetricCount) {
		thread.incrementStorage();
	    }
	    while (globalMappingElement.getStorageSize() < newMetricCount) {
		globalMappingElement.incrementStorage();
	    }
	    while (globalThreadDataElement.getStorageSize() < newMetricCount) {
		globalThreadDataElement.incrementStorage();
	    }

	    dEventValue = dEventValue * 1000 * 1000;

	    if (inclusive) {
		globalThreadDataElement.setInclusiveValue(metric, dEventValue);
		//globalThreadDataElement.setExclusiveValue(metric, dEventValue);
	    } else {
		globalThreadDataElement.setExclusiveValue(metric, dEventValue);
	    }

	    double tmpValue = dEventValue / ((double)(header3.i0));
	    globalThreadDataElement.setUserSecPerCall(metric, tmpValue);

	    if (inclusive) {
		if ((globalMappingElement.getMaxInclusiveValue(metric)) < dEventValue) {
		    globalMappingElement.setMaxInclusiveValue(metric, dEventValue);
		}
	    } else {
		if ((globalMappingElement.getMaxExclusiveValue(metric)) < dEventValue) {
		    globalMappingElement.setMaxExclusiveValue(metric, dEventValue);
		}
	    }

	    if (globalMappingElement.getMaxUserSecPerCall(metric) < (dEventValue / header3.i0))
		globalMappingElement.setMaxUserSecPerCall(metric, (dEventValue / header3.i0));
	    
	    globalThreadDataElement.setExclusivePercentValue(metric, 0);
	    globalThreadDataElement.setInclusivePercentValue(metric, 0);
	    globalThreadDataElement.setNumberOfCalls(header3.i0);
	    globalThreadDataElement.setNumberOfSubRoutines(0);
	    globalMappingElement.setMaxExclusivePercentValue(metric, 0.0);
	    globalMappingElement.setMaxInclusivePercentValue(metric, 0.0);
	    if(globalMappingElement.getMaxNumberOfCalls() < header3.i0)
		globalMappingElement.setMaxNumberOfCalls(header3.i0);
	    globalMappingElement.setMaxNumberOfSubRoutines(0);

	} catch(Exception e) {
	    System.out.println("An error occurred while parsing the wall clock time!");
	    e.printStackTrace();
	}
    }


    private String getMetricName(String inString){
	try{
	    String tmpString = null;
	    int tmpInt = inString.indexOf("_MULTI_");

	    if(tmpInt > 0){
		//We are reading data from a multiple counter run.
		//Grab the counter name.
		tmpString = inString.substring(tmpInt+7);
		return tmpString;
	    }
	    //We are not reading data from a multiple counter run.
	    return tmpString; 
	} catch(Exception e) {
	    UtilFncs.systemError(e, null, "SSD26");
	}
	return null;
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
    private int timeMetric = -1;
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
    boolean initialized = false;
    Hashtable eventNames = new Hashtable();
    //######
    //End - Frequently used items.
    //######

    //####################################
    //Instance data.
    //####################################
    private LineData header1 = new LineData();
    private LineData header2 = new LineData();
    private LineData header3 = new LineData();
    private LineData header4 = new LineData();
    private LineData header5 = new LineData();
    //####################################
    //End - Instance data.
    //####################################
}
