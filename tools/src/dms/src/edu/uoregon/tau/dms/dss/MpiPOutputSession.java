/* 
   Name:        MpiPOutputSession.java
   Author:      Kevin Huck
   Description: Parse an mpiP data file.
*/

/*
  To do: 
  The mpiP data has min, mean, max values.  What should be done with these values?
  Should they be stored in a user event?
*/

package edu.uoregon.tau.dms.dss;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.io.*;
import java.util.*;

public class MpiPOutputSession extends ParaProfDataSession{

    public MpiPOutputSession(){
		super();
		this.setMetrics(new Vector());
    }

    public void initialize(Object initializeObject){
		try{
	    	//######
	    	//Frequently used items.
	    	//######
	    	int metric = 0;
	    	GlobalMappingElement globalMappingElement = null;
	    	GlobalThreadDataElement globalThreadDataElement = null;
	    
	    	Node node = null;
	    	Context context = null;
	    	edu.uoregon.tau.dms.dss.Thread thread = null;
	    	int nodeID = -1;
	    	int contextID = -1;
	    	int threadID = -1;
	    
	    	String inputString = null;
	    	String s1 = null;
	    	String s2 = null;
	    
	    	String tokenString;
	    	String groupNamesString = null;
	    	StringTokenizer genericTokenizer;
	    
	    	int mappingID = -1;
	    
	    	Vector v = null;
	    	File[] files = null;
	    	//######
	    	//End - Frequently used items.
	    	//######

	    	v = (Vector) initializeObject;
	    	for(Enumeration e = v.elements(); e.hasMoreElements() ;){
				files = (File[]) e.nextElement();
				System.out.println("Processing data file, please wait ......");
				long time = System.currentTimeMillis();

				FileInputStream fileIn = new FileInputStream(files[0]);
				InputStreamReader inReader = new InputStreamReader(fileIn);
				BufferedReader br = new BufferedReader(inReader);
      
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
				String metricName = "Time";
      
				//Need to call increaseVectorStorage() on all objects that require it.
				this.getGlobalMapping().increaseVectorStorage();
      
				System.out.println("Metric name is: " + metricName);
      
				metric = this.getNumberOfMetrics();
				this.addMetric(metricName);

				int eventCount = 0;
				// find the callsite names
				while((inputString = br.readLine()) != null){
					if (inputString.startsWith("@--- Callsites:")) {
						// System.out.print("Found callsites: ");
		    			genericTokenizer = new StringTokenizer(inputString, ":");
						// left side
						inputString = genericTokenizer.nextToken();
						// right side
						inputString = genericTokenizer.nextToken();
	    				genericTokenizer = new StringTokenizer(inputString, " ");
						// get the callsite count
						eventCount = Integer.parseInt(genericTokenizer.nextToken());
						// the callsite names are indexed at 1, not 0.
						eventNames = new String[eventCount+1];
						// System.out.println(eventCount + " callsites found.");
						// ignore the next two lines
						br.readLine();
						br.readLine();
						// exit this while loop
						break;
					}
				}

				if (inputString != null) {
					// parse each of the event names
					for (int i = 1 ; i <= eventCount ; i++) {
						inputString = br.readLine();
						getCallsiteHeaders(inputString);
						eventNames[i] = new String(callsiteHeader.s1 + " -> " + "MPI_" + callsiteHeader.s0);
					}
				}

				// find the callsite data
				int eventDataCount = 0;
				while((inputString = br.readLine()) != null){
					if (inputString.startsWith("@--- Callsite statistics")) {
						// exit this while loop
						// System.out.print("Found callsite data: ");
		    			genericTokenizer = new StringTokenizer(inputString, ":");
						// left side
						inputString = genericTokenizer.nextToken();
						// right side
						inputString = genericTokenizer.nextToken();
	    				genericTokenizer = new StringTokenizer(inputString, " ");
						// get the callsite count
						eventDataCount = Integer.parseInt(genericTokenizer.nextToken());
						// System.out.println(eventDataCount + " callsite data lines found.");
						// ignore the next two lines
						br.readLine();
						br.readLine();
						break;
					}
				}

				if (inputString != null) {
					// parse each of the event names
					for (int i = 0 ; i < eventCount ; i++) {
						inputString = br.readLine();
						while (inputString != null && (inputString.length() == 0))
							inputString = br.readLine();
						if (inputString != null) {
							getCallsiteData(inputString);
							mappingID = this.getGlobalMapping().addGlobalMapping(eventNames[callsiteData.i0], 0, 1);
							globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);
							if (callsiteData.i1 >= 0) {
								if((globalMappingElement.getMaxExclusiveValue(metric)) < callsiteData.d5) {
			    					globalMappingElement.setMaxExclusiveValue(metric, callsiteData.d5);
			    					globalMappingElement.setMaxInclusiveValue(metric, callsiteData.d5);
								}
								if((globalMappingElement.getMaxExclusivePercentValue(metric)) < callsiteData.d4) {
			    					globalMappingElement.setMaxExclusivePercentValue(metric, callsiteData.d4);
			    					globalMappingElement.setMaxInclusivePercentValue(metric, callsiteData.d4);
								}
								if(globalMappingElement.getMaxNumberOfCalls() < callsiteData.i2)
			    					globalMappingElement.setMaxNumberOfCalls(callsiteData.i2);
			    				globalMappingElement.setMaxNumberOfSubRoutines(0);
								if(globalMappingElement.getMaxUserSecPerCall(metric) < (callsiteData.d1))
			    					globalMappingElement.setMaxUserSecPerCall(metric, (callsiteData.d1));
								// get the node data
								nodeID = callsiteData.i1;
								contextID = 0;
								threadID = 0;
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
								}
								globalThreadDataElement = thread.getFunction(mappingID);
								if(globalThreadDataElement == null){
			    					globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(mappingID, 0), false);
			    					thread.addFunction(globalThreadDataElement, mappingID);
								}
								globalThreadDataElement.setExclusiveValue(metric, callsiteData.d5);
								globalThreadDataElement.setExclusivePercentValue(metric, callsiteData.d4);
								globalThreadDataElement.setInclusiveValue(metric, callsiteData.d5);
								globalThreadDataElement.setInclusivePercentValue(metric, callsiteData.d4);
								globalThreadDataElement.setNumberOfCalls(callsiteData.i2);
								globalThreadDataElement.setNumberOfSubRoutines(0);
								globalThreadDataElement.setUserSecPerCall(metric, callsiteData.d1);

								//Now check the max values on this thread.
								if(thread.getMaxNumberOfCalls() < callsiteData.i2)
			    					thread.setMaxNumberOfCalls(callsiteData.i2);
			    				thread.setMaxNumberOfSubRoutines(0);
								if(thread.getMaxUserSecPerCall(metric) < callsiteData.d1)
			    					thread.setMaxUserSecPerCall(metric, callsiteData.d1);
								if((thread.getMaxExclusiveValue(metric)) < callsiteData.d5) {
			    					thread.setMaxExclusiveValue(metric, callsiteData.d5);
			    					thread.setMaxInclusiveValue(metric, callsiteData.d5);
								}
								if((thread.getMaxExclusivePercentValue(metric)) < callsiteData.d4) {
			    					thread.setMaxExclusivePercentValue(metric, callsiteData.d4);
			    					thread.setMaxInclusivePercentValue(metric, callsiteData.d4);
								}
							} else {
								// save the total data
							}

						}
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

    //######
    //Pprof.dat string processing methods.
    //######

    private void getCallsiteHeaders(String string){
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, " ");
	    	callsiteHeader.i0 = Integer.parseInt(st1.nextToken()); // callsite index
	    	callsiteHeader.s0 = st1.nextToken(); // MPI function
	    	callsiteHeader.s1 = st1.nextToken(); // Parent Function
	    	callsiteHeader.s2 = st1.nextToken(); // Filename
	    	callsiteHeader.i1 = Integer.parseInt(st1.nextToken()); // Line
	    	callsiteHeader.s3 = st1.nextToken(); // PC
		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the callsite header!");
	    	e.printStackTrace();
		}
    }

    private void getCallsiteData(String string){
		try{
	    	StringTokenizer st1 = new StringTokenizer(string, " ");
	    	callsiteData.s0 = st1.nextToken(); // MPI function
	    	callsiteData.i0 = Integer.parseInt(st1.nextToken()); // callsite index
			String tmpString = st1.nextToken(); // rank
			if (tmpString.equals("*"))
	    		callsiteData.i1 = -1;
			else 
	    		callsiteData.i1 = Integer.parseInt(tmpString); // rank
	    	callsiteData.i2 = Integer.parseInt(st1.nextToken()); // count
	    	callsiteData.d0 = Double.parseDouble(st1.nextToken()); // Max
	    	callsiteData.d1 = Double.parseDouble(st1.nextToken()); // Mean
	    	callsiteData.d2 = Double.parseDouble(st1.nextToken()); // Min
	    	callsiteData.d3 = Double.parseDouble(st1.nextToken()); // App%
	    	callsiteData.d4 = Double.parseDouble(st1.nextToken()); // MPI%
	    	callsiteData.d5 = callsiteData.d1 * callsiteData.i2; // Total time for this node
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
    //######
    //End - Pprof.dat string processing methods.
    //######

    //####################################
    //End - Private Section.
    //####################################

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
