/* 
   Name:        HPMToolkitDataSession.java
   Author:      Kevin Huck
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

    public void run(){
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
      
	  /*
					if (firstFile) {
						//Set the metric name.
						String metricName = "HPM Toolkit Counters";
      
						//Need to call increaseVectorStorage() on all objects that require it.
						this.getGlobalMapping().increaseVectorStorage();
      
						// System.out.println("Metric name is: " + metricName);
      
						metric = this.getNumberOfMetrics();
						this.addMetric(metricName);
						firstFile = false;
					}
					*/

					// find the end of the resource statistics
					while((inputString = br.readLine()) != null){
						if (inputString.trim().startsWith("#######  End of Resource Statistics")) {
							// System.out.println("Found beginning of data: ");
							// exit this while loop
							break;
						}
					}

					// find the callsite data
					while((inputString = br.readLine()) != null){
						if (inputString.length() == 0) {
							// do nothing
						} else if (inputString.trim().startsWith("Instrumented section:")) {
							processHeaderLine1(inputString);
						} else if (inputString.trim().startsWith("file:")) {
							processHeaderLine2(inputString);
						} else if (inputString.trim().startsWith("Count:")) {
							processHeaderLine3(inputString);
						} else if (inputString.trim().startsWith("Wall Clock Time:")) {
							processHeaderLine4(inputString);
						} else if (inputString.trim().startsWith("Total time in user mode:")) {
							processHeaderLine5(inputString);
						} else {
							processHardwareCounter(inputString);
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

	    	//Need to notify observers that we are done.  Be careful here.
	    	//It is likely that they will modify swing elements.  Make sure
	    	//to dump request onto the event dispatch thread to ensure
	    	//safe update of said swing elements.  Remember, swing is not thread
	    	//safe for the most part.
	    	EventQueue.invokeLater(new Runnable(){
		    	public void run(){
					HPMToolkitDataSession.this.notifyObservers();
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
			thread.initializeFunctionList(this.getGlobalMapping().getNumberOfMappings(0));
			thread.initializeUsereventList(this.getGlobalMapping().getNumberOfMappings(2));
		}
		initialized = true;
	}

	private void processHeaderLine1(String string) {
		// System.out.println("Header line 1");
		contextID++;
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

			initialized = false;
		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the header!");
	    	e.printStackTrace();
		}
	}

	private void processHeaderLine2(String string) {
		// System.out.println("Header line 2");
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
			// ignore the "Label" label
			string = st2.nextToken();
			// get the value
	    	header2.s1 = st2.nextToken().trim(); // label value
		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the header!");
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
	    	System.out.println("An error occured while parsing the header!");
	    	e.printStackTrace();
		}
	}

	private void processHeaderLine4(String string) {
		// System.out.println("Header line 4");
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, ":");
			// ignore the "Wall Clock Time" label
			string = st1.nextToken();
			// get the value
			string = st1.nextToken();
	    	StringTokenizer st2 = new StringTokenizer(string, " ");
	    	header4.d0 = Double.parseDouble(st2.nextToken()); // section id
		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the header!");
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
	    	System.out.println("An error occured while parsing the header!");
	    	e.printStackTrace();
		}
		initializeThread();
	}

	private void processHardwareCounter(String string) {
		if (!initialized) {
			initializeThread();
		}
		// System.out.println("Hardwoare counter");
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, ":");
	    	String eventName = st1.nextToken().trim(); // hardware counter name
			String tmpStr = st1.nextToken().trim();
			// need to clean stuff out of the value, like % and M and whatnot
	    	st1 = new StringTokenizer(tmpStr, " ");
	    	double dEventValue = 0.0;
	    	int iEventValue = 0;
			tmpStr = st1.nextToken().trim();
			if (tmpStr.indexOf(".") > -1)
	    		dEventValue = Double.parseDouble(tmpStr); // callsite index
			else
	    		iEventValue = Integer.parseInt(tmpStr); // callsite index
			if (st1.hasMoreTokens())
				eventName += " (" + st1.nextToken() + ")";
			// System.out.println(eventName + " = " + eventValue);
			// save the thing...
			mappingID = this.getGlobalMapping().addGlobalMapping(eventName, 2, 1);
			globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 2);
			globalThreadDataElement = thread.getUserevent(mappingID);

			if(globalThreadDataElement == null) {
				globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(mappingID, 2), true);
				thread.addUserevent(globalThreadDataElement, mappingID);
			}
			globalThreadDataElement.setUserEventNumberValue(iEventValue);
			globalThreadDataElement.setUserEventMinValue(dEventValue);
			globalThreadDataElement.setUserEventMaxValue(dEventValue);
			globalThreadDataElement.setUserEventMeanValue(dEventValue);

			if((globalMappingElement.getMaxUserEventNumberValue()) < iEventValue)
				globalMappingElement.setMaxUserEventNumberValue(iEventValue);
			if((globalMappingElement.getMaxUserEventMaxValue()) < dEventValue)
				globalMappingElement.setMaxUserEventMaxValue(dEventValue);
   			if((globalMappingElement.getMaxUserEventMinValue()) < dEventValue)
				globalMappingElement.setMaxUserEventMinValue(dEventValue);
   			if((globalMappingElement.getMaxUserEventMeanValue()) < dEventValue)
				globalMappingElement.setMaxUserEventMeanValue(dEventValue);
		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the callsite data!");
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
	private String[] eventNames = null;
    //####################################
    //End - Instance data.
    //####################################
}
