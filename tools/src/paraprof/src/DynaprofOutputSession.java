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
		this.increaseVectorStorage();
		
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
		    
		    
		    if(this.firstMetric()){
			//If header is present, its lines will begin with '#'
			inputString = br.readLine();
			if(inputString == null){
			    System.out.println("Error processing file: " + files[i].getName());
			    System.out.println("Unexpected end of file!");
			    return;
			}
			else if((inputString.charAt(0)) == '#'){
			    //Header present.
			    //Do not need second header line at the moment..
			    br.readLine();
			    //Third header line contains the number of metrics.
			    genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
			    genericTokenizer.nextToken();
			    tokenString = genericTokenizer.nextToken();
			    for(int j=(Integer.parseInt(tokenString));j>0;j--){
				inputString = br.readLine();
				genericTokenizer = new StringTokenizer(inputString, " :\t\n\r");
				Metric metricRef = this.addMetric();
				metricRef.setName(genericTokenizer.nextToken());
				
			    }
			    if(this.debug()){
				System.out.println("Header present");
				System.out.println("Number of metrics: " + this.getNumberOfMetrics());
			    }
			}
			else{
			    if(this.debug())
				System.out.println("No header present");
			    Metric metricRef = this.addMetric();
			    metric = metricRef.getID();
			    metricRef.setName("default");
			}
			
			for(int j=this.getNumberOfMetrics();j>0;j--)
			    this.increaseVectorStorage();
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
			inputString = br.readLine();
			this.getFunctionDataLine(inputString);
			//Calculate usec/call
			double usecCall = functionDataLine.d0/functionDataLine.i0;
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
			for(int j=0;j<functionDataLine.i0;j++){
			   this.getFunctionChildDataLine(inputString);
			   if(this.debug()){
			       System.out.println("function child line: " + inputString);
			       System.out.println("name:"+functionDataLine.s0);
			       System.out.println("incl.total:"+functionDataLine.d3);
			       System.out.println("incl.calls:"+functionDataLine.i2);
			       System.out.println("incl.min:"+functionDataLine.d4);
			       System.out.println("incl.max:"+functionDataLine.d5);
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
	    StringTokenizer st1 = new StringTokenizer(string, "\"");
	    functionDataLine.s0 = st1.nextToken(); //Name
	    
	    StringTokenizer st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
	    functionDataLine.i0 = Integer.parseInt(st2.nextToken()); //Calls
	    functionDataLine.i1 = Integer.parseInt(st2.nextToken()); //Subroutines
	    functionDataLine.d0 = Double.parseDouble(st2.nextToken()); //Exclusive
	    functionDataLine.d1 = Double.parseDouble(st2.nextToken()); //Inclusive
	    if(this.profileStatsPresent())
		functionDataLine.d2 = Double.parseDouble(st2.nextToken()); //SumExclSqr
	    functionDataLine.i2 = Integer.parseInt(st2.nextToken()); //ProfileCalls
	}
	catch(Exception e){
	    ParaProf.systemError(e, null, "SSD08");
	}
    }

    private void getFunctionChildDataLine(String string){
	try{
	    StringTokenizer st1 = new StringTokenizer(string, "\"");
	    usereventDataLine.s0 = st1.nextToken();

	    StringTokenizer st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
	    usereventDataLine.i0 = (int) Double.parseDouble(st2.nextToken()); //Number of calls.
	    usereventDataLine.d0 = Double.parseDouble(st2.nextToken()); //Max
	    usereventDataLine.d1 = Double.parseDouble(st2.nextToken()); //Min
	    usereventDataLine.d2 = Double.parseDouble(st2.nextToken()); //Mean
	    usereventDataLine.d3 = Double.parseDouble(st2.nextToken()); //Standard Deviation.
	}
	catch(Exception e){
	    System.out.println("An error occured!");
	    e.printStackTrace();
	}
    }

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
    private LineData  usereventDataLine = new LineData();
    //####################################
    //End - Instance data.
    //####################################
}
