/* 
   DynaprofOutputSession.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

/*
  To do: 
  1) Add some sanity checks to make sure that multiple metrics really do belong together.
  For example, wrap the creation of nodes, contexts, threads, global mapping elements, and
  the like so that they do not occur after the first metric has been loaded.  This will
  not of course ensure 100% that the data is consistent, but it will at least prevent the
  worst cases.
*/

package paraprof;



import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

import java.io.*;
import java.util.*;
import dms.dss.*;

public class DynaprofOutputSession extends ParaProfDataSession{

    public DynaprofOutputSession(){
	super();
    }

    /**
     * Initialize the DataSession object.
     *
     * @param	obj	an implementation-specific object required to initialize the DataSession
     */
    public void initialize(Object obj){
	try{
	    //######
	    //Frequently used items.
	    //######
	    int metric = 0;
	    GlobalMappingElement globalMappingElement = null;
	    GlobalThreadDataElement globalThreadDataElement = null;
	    
	    Node node = null;
	    Context context = null;
	    Thread thread = null;
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
	    
	    int numberOfLines = 0;

	    Vector v = null;
	    File[] files = null;
	    //######
	    //End - Frequently used items.
	    //######
	    v = (Vector) obj;
	    for(Enumeration e = v.elements(); e.hasMoreElements() ;){
		System.out.println("Processing data, please wait ......");
		long time = System.currentTimeMillis();
		//Need to call increaseVectorStorage() on all objects that require it.
		this.getGlobalMapping().increaseVectorStorage();
		
		files = (File[]) e.nextElement();
		for(int i=0;i<files.length;i++){
		    if(this.debug()){
			System.out.println("######");
			System.out.println("Processing file: " + files[i].getName());
			System.out.println("######");
		    }

		    FileInputStream fileIn = new FileInputStream(files[i]);
		    InputStreamReader inReader = new InputStreamReader(fileIn);
		    BufferedReader br = new BufferedReader(inReader);

		    //For Dynaprof, we assume that the number of metrics remain fixed over
		    //all threads. Since this information is stored in the header of the papiprobe
		    //output of each file, we only need to get this information from the first file.

		    //Since each file contains all metric information for that thread, reset to
		    //indicate that we are back at the first metric.
		    this.setFirstMetric(true);

		    if(!(this.headerProcessed())){
			//If header is present, its lines will begin with '#'
			inputString = br.readLine();
			if(inputString == null){
			    System.out.println("Error processing file: " + files[i].getName());
			    System.out.println("Unexpected end of file!");
			    return;
			}
			else if((inputString.charAt(0)) == '#'){
			    if(this.debug())
				System.out.println("Header present");
			    //Do not need second header line at the moment..
			    br.readLine();
			    //Third header line contains the number of metrics.
			    genericTokenizer = new StringTokenizer( br.readLine(), " \t\n\r");
			    genericTokenizer.nextToken();
			    tokenString = genericTokenizer.nextToken();
			    for(int j=(Integer.parseInt(tokenString));j>0;j--){
				inputString = br.readLine();
				//Metric name is second token on line.
				genericTokenizer = new StringTokenizer(inputString, " :\t\n\r");
				genericTokenizer.nextToken();
				Metric metricRef = this.addMetric();
				metricRef.setName(genericTokenizer.nextToken());
				if(this.debug())
				    System.out.println("metric name found: " + metricRef.getName());
			    }
			    if(this.debug())
				System.out.println("Number of metrics: " + this.getNumberOfMetrics());
			}
			else{
			    if(this.debug())
				System.out.println("No header present");
			    Metric metricRef = this.addMetric();
			    metric = metricRef.getID();
			    metricRef.setName("default");
			}
			
			for(int j=this.getNumberOfMetrics();j>0;j--)
			    this.getGlobalMapping().increaseVectorStorage();
			//this.increaseVectorStorage();
			this.setHeaderProcessed(true);
		    }
		    
		    //Metrics and names should be set by now.
		    int[] nct = this.getNCT(files[i].getName());
		    nodeID = nct[0];
		    contextID = nct[1];
		    threadID = nct[2];
		    
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
			for(int j=this.getNumberOfMetrics();j>0;j--)
			    thread.incrementStorage();
			//For Dynaprof, we can be reasonable sure that functions are the only
			//things likely to be tracked.  See comment before Thread.initializeFunctionList(...)
			//in TauOutputSession for more details on the positioning in that class.
			thread.initializeFunctionList(this.getNumberOfMappings());
		    }
		    if(this.debug())
			System.out.println("n,c,t: " + nct[0] + "," + nct[1] + "," + nct[2]);

		    //####################################
		    //First  Line
		    //####################################
		    while((inputString = br.readLine()) != null){
			this.getFunctionDataLine(inputString);
			boolean totalLine = (inputString.indexOf("TOTAL")==0);

			//The start of a new metric is indicated by the presence of TOTAL.
			//if(inputString.indexOf("TOTAL")==0){
			if(totalLine){
			    if(!(this.firstMetric())){
				//The thread object takes care of computing maximums and totals for a given metric, as
				//well as the percent.  Must do the order correctly to get the correct results.
				thread.setThreadSummaryData(metric);
				thread.setPercentData(metric);
				//Call the setThreadSummaryData function again on this thread so that
				//it can fill in all the summary data.
				thread.setThreadSummaryData(metric);
				this.getGlobalMapping().computeMeanData(0,metric);
				metric++;
			    }
			    else
				this.setFirstMetric(false);
			}

			//Calculate usec/call
			double usecCall = functionDataLine.d0/functionDataLine.i1;
			if(this.debug()){
			    System.out.println("function line: " + inputString);
			    System.out.println("name:"+functionDataLine.s0);
			    System.out.println("number_of_children:"+functionDataLine.i0);
			    System.out.println("excl.total:"+functionDataLine.d0);
			    System.out.println("excl.calls:"+functionDataLine.i1);
			    System.out.println("excl.min:"+functionDataLine.d1);
			    System.out.println("excl.max:"+functionDataLine.d2);
			    System.out.println("incl.total:"+functionDataLine.d3);
			    System.out.println("incl.calls:"+functionDataLine.i2);
			    System.out.println("incl.min:"+functionDataLine.d4);
			    System.out.println("incl.max:"+functionDataLine.d5);
			}
			if(!totalLine && functionDataLine.i1 !=0){
			    mappingID = this.getGlobalMapping().addGlobalMapping(functionDataLine.s0, 0);
			    globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);
			    for(int j=this.getNumberOfMetrics();j>0;j--)
				globalMappingElement.incrementStorage();
			    globalMappingElement.incrementCounter();
			    globalThreadDataElement = thread.getFunction(mappingID);
			    
			    if(globalThreadDataElement == null){
				globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(mappingID, 0), false);
				for(int j=this.getNumberOfMetrics();j>0;j--)
				    globalThreadDataElement.incrementStorage();
				thread.addFunction(globalThreadDataElement, mappingID);
			    }
			    
			    globalThreadDataElement.setMappingExists();
			    globalThreadDataElement.setExclusiveValue(metric, functionDataLine.d0);
			    globalThreadDataElement.setInclusiveValue(metric, functionDataLine.d3);
			    globalThreadDataElement.setNumberOfCalls(functionDataLine.i1);
			    globalThreadDataElement.setNumberOfSubRoutines(functionDataLine.i0);
			    globalThreadDataElement.setUserSecPerCall(metric, usecCall);
			    
			    globalMappingElement.incrementTotalExclusiveValue(functionDataLine.d0);
			    globalMappingElement.incrementTotalInclusiveValue(functionDataLine.d3);
			    
			    //Set the max values.
			    if((globalMappingElement.getMaxExclusiveValue(metric)) < functionDataLine.d0)
				globalMappingElement.setMaxExclusiveValue(metric, functionDataLine.d0);
			    if((thread.getMaxExclusiveValue(metric)) < functionDataLine.d0)
				thread.setMaxExclusiveValue(metric, functionDataLine.d0);
			    
			    if((globalMappingElement.getMaxInclusiveValue(metric)) < functionDataLine.d3)
				globalMappingElement.setMaxInclusiveValue(metric, functionDataLine.d3);
			    if((thread.getMaxInclusiveValue(metric)) < functionDataLine.d3)
				thread.setMaxInclusiveValue(metric, functionDataLine.d3);
			    
			    if(globalMappingElement.getMaxNumberOfCalls() < functionDataLine.i1)
				globalMappingElement.setMaxNumberOfCalls(functionDataLine.i1);
			    if(thread.getMaxNumberOfCalls() < functionDataLine.i1)
				thread.setMaxNumberOfCalls(functionDataLine.i1);
			    
			    if(globalMappingElement.getMaxNumberOfSubRoutines() < functionDataLine.i0)
				globalMappingElement.setMaxNumberOfSubRoutines(functionDataLine.i0);
			    if(thread.getMaxNumberOfSubRoutines() < functionDataLine.i0)
				thread.setMaxNumberOfSubRoutines(functionDataLine.i0);
			    
			    if(globalMappingElement.getMaxUserSecPerCall(metric) < usecCall)
				globalMappingElement.setMaxUserSecPerCall(metric, usecCall);
			    if(thread.getMaxUserSecPerCall(metric) < usecCall)
				thread.setMaxUserSecPerCall(metric, usecCall);
			}
			for(int j=0;j<functionDataLine.i0;j++){
			    inputString = br.readLine();
			    System.out.println("function child line: " + inputString);
			    this.getFunctionChildDataLine(inputString);
			    if(this.debug()){
				System.out.println("function child line: " + inputString);
				System.out.println("name:"+functionDataLine.s0);
				System.out.println("incl.total:"+functionDataLine.d3);
				System.out.println("incl.calls:"+functionDataLine.i2);
				System.out.println("incl.min:"+functionDataLine.d4);
				System.out.println("incl.max:"+functionDataLine.d5);
			    }
			    if(functionDataLine.i1 !=0){
				mappingID = this.getGlobalMapping().addGlobalMapping(functionDataLine.s0+" > child", 0);
				globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);
				for(int k=this.getNumberOfMetrics();k>0;k--)
				    globalMappingElement.incrementStorage();
				globalMappingElement.incrementCounter();
				globalThreadDataElement = thread.getFunction(mappingID);
				
				if(globalThreadDataElement == null){
				    globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(mappingID, 0), false);
				    for(int k=this.getNumberOfMetrics();k>0;k--)
					globalThreadDataElement.incrementStorage();
				    thread.addFunction(globalThreadDataElement, mappingID);
				}
				
				globalThreadDataElement.setMappingExists();
				//Since this is the child thread, increment the values.
				double d1 = globalThreadDataElement.getInclusiveValue(metric);
				double d2 = d1 + functionDataLine.d3;
				globalThreadDataElement.setExclusiveValue(metric, d2);
				globalThreadDataElement.setInclusiveValue(metric, d2);
				
				int i1 = globalThreadDataElement.getNumberOfCalls();
				if(this.firstMetric()){
				    i1 = globalThreadDataElement.getNumberOfCalls();
				    globalThreadDataElement.setNumberOfCalls(i1+functionDataLine.i2);
				}
				i1 = globalThreadDataElement.getNumberOfCalls();
				globalThreadDataElement.setUserSecPerCall(metric, d2/i1);
			    }
			}
		    }
		}
		
		time = (System.currentTimeMillis()) - time;
		System.out.println("Done processing data!");
		System.out.println("Time to process (in milliseconds): " + time);
	    }
	}
        catch(Exception e){
	    ParaProf.systemError(e, null, "SSD01");
	}
    }
    
    //####################################
    //Private Section.
    //####################################

    //######
    //profile.*.*.* string processing methods.
    //######
     private int[] getNCT(String string){
	int[] nct = new int[3];
	StringTokenizer st = new StringTokenizer(string, ".\t\n\r");
	st.nextToken();
	nct[0] = 0;
	nct[1] = Integer.parseInt(st.nextToken());
	if(st.hasMoreTokens())
	    nct[2] = Integer.parseInt(st.nextToken());
	else
	    nct[2] = 0;
	return nct;
    }
  
    private String getCounterName(String inString){
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
      	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD26");
	}
    
	return null;
    }

    private void getFunctionDataLine(String string){
	try{
	    StringTokenizer st = new StringTokenizer(string, ",\t\n\r");
	    functionDataLine.s0 = st.nextToken(); //name
	    functionDataLine.i0 = Integer.parseInt(st.nextToken()); //number_of_children
	    functionDataLine.d0 = Double.parseDouble(st.nextToken()); //excl.total
	    functionDataLine.i1 = Integer.parseInt(st.nextToken()); //excl.calls
	    functionDataLine.d1 = Double.parseDouble(st.nextToken()); //excl.min
	    functionDataLine.d2 = Double.parseDouble(st.nextToken()); //excl.max
	    functionDataLine.d3 = Double.parseDouble(st.nextToken()); //incl.total
	    functionDataLine.i2 = Integer.parseInt(st.nextToken()); //incl.calls
	    functionDataLine.d4 = Double.parseDouble(st.nextToken()); //incl.min
	    functionDataLine.d5 = Double.parseDouble(st.nextToken()); //incl.max
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD08");
	}
    }

    private void getFunctionChildDataLine(String string){
	try{
	    StringTokenizer st = new StringTokenizer(string, ",\t\n\r");
	    functionDataLine.s0 = st.nextToken(); //name
	    functionDataLine.d3 = Double.parseDouble(st.nextToken()); //incl.total
	    functionDataLine.i2 = Integer.parseInt(st.nextToken()); //incl.calls
	    functionDataLine.d4 = Double.parseDouble(st.nextToken()); //incl.min
	    functionDataLine.d5 = Double.parseDouble(st.nextToken()); //incl.max
	}
	catch(Exception e){
	    System.out.println("An error occured!");
	    e.printStackTrace();
	}
    }

    private void setHeaderProcessed(boolean headerProcessed){
	this.headerProcessed = headerProcessed;}

    private boolean headerProcessed(){
	return headerProcessed;}

    //######
    //End - profile.*.*.* string processing methods.
    //######

    //####################################
    //End - Private Section.
    //####################################

    //####################################
    //Instance data.
    //####################################
    private LineData functionDataLine = new LineData();
     private boolean headerProcessed = false;
    //####################################
    //End - Instance data.
    //####################################
}
