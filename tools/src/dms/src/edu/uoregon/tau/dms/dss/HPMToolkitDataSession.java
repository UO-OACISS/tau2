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
		try{
	    	v = (Vector) initializeObject;
	    	for(Enumeration e = v.elements(); e.hasMoreElements() ;){
				files = (File[]) e.nextElement();
				System.out.println("Processing data file, please wait ......");
				long time = System.currentTimeMillis();

				FileInputStream fileIn = new FileInputStream(files[0]);
				InputStreamReader inReader = new InputStreamReader(fileIn);
				br = new BufferedReader(inReader);
      
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
      
				//Set the metric name.
				String metricName = "HPM Toolkit Counters";
      
				//Need to call increaseVectorStorage() on all objects that require it.
				this.getGlobalMapping().increaseVectorStorage();
      
				System.out.println("Metric name is: " + metricName);
      
				metric = this.getNumberOfMetrics();
				this.addMetric(metricName);

				// find the end of the resource statistics
				while((inputString = br.readLine()) != null){
					if (inputString.trim().startsWith("#######  End of Resource Statistics")) {
						System.out.println("Found beginning of data: ");
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
		System.out.println("Header line 1");
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, "-");
			// get the first name/value pair
			string = st1.nextToken();
	    	StringTokenizer st2 = new StringTokenizer(string, ":");
			// ignore the "instrumented section" label
			string = st1.nextToken();
			// get the value
	    	contextID = Integer.parseInt(st1.nextToken()) - 1; // context index
			initialized = false;
		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the header!");
	    	e.printStackTrace();
		}
	}

	private void processHeaderLine2(String string) {
		System.out.println("Header line 2");
	}

	private void processHeaderLine3(String string) {
		System.out.println("Header line 3");
	}

	private void processHeaderLine4(String string) {
		System.out.println("Header line 4");
	}

	private void processHeaderLine5(String string) {
		System.out.println("Header line 5");
		initializeThread();
	}

	private void processHardwareCounter(String string) {
		if (!initialized)
			initializeThread();
		// System.out.println("Hardwoare counter");
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, ":");
	    	String eventName = st1.nextToken().trim(); // hardware counter name
			String tmpStr = st1.nextToken().trim();
			// need to clean stuff out of the value, like % and M and whatnot
	    	st1 = new StringTokenizer(tmpStr, " ");
	    	double dEventValue = 0.0;
	    	int iEventValue = 0;
			tmpStr = st1.nextToken();
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

    private int[] getNCT(String string){
		int[] nct = new int[3];
		StringTokenizer st = new StringTokenizer(string, " ,\t\n\r");
		nct[0] = Integer.parseInt(st.nextToken());
		nct[1] = Integer.parseInt(st.nextToken());
		nct[2] = Integer.parseInt(st.nextToken());
		return nct;
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
	private int nodeID = 0;
	private int contextID = 0;
	private int threadID = 0;
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
    private LineData callsiteHeader = new LineData();
    private LineData callsiteData = new LineData();
	private String[] eventNames = null;
    //####################################
    //End - Instance data.
    //####################################
}
