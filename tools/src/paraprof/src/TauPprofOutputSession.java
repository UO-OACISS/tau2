/* 
   TauPprofOutputSession.java
   
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

public class TauPprofOutputSession extends ParaProfDataSession{

    public TauPprofOutputSession(){
	super();
	this.setMetrics(new Vector());
    }

    public void run(){
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
	    
	    //A loop counter.
	    int bSDCounter = 0;

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
		bSDCounter++;
		//####################################
		//End - First Line
		//####################################
      
		//####################################
		//Second  Line
		//####################################
		//This is an important line.
		inputString = br.readLine();
		//Set the metric name.
		String metricName = getMetricName(inputString);
      
		//Need to call increaseVectorStorage() on all objects that require it.
		this.getGlobalMapping().increaseVectorStorage();
		//Only need to call addDefaultToVectors() if not the first run.
		if(!(this.firstMetric())){
		    if(UtilFncs.debug)
			System.out.println("Increasing the storage for the new counter.");
		
		    for(Enumeration e1 = (this.getGlobalMapping().getMapping(0)).elements(); e1.hasMoreElements() ;){
			GlobalMappingElement tmpGME = (GlobalMappingElement) e1.nextElement();
			tmpGME.incrementStorage();
		    }
	  
		    for(Enumeration e2 = (this.getGlobalMapping().getMapping(2)).elements(); e2.hasMoreElements() ;){
			GlobalMappingElement tmpGME = (GlobalMappingElement) e2.nextElement();
			tmpGME.incrementStorage();
		    }
	  
		    for(Enumeration e3 = this.getNCT().getNodes().elements(); e3.hasMoreElements() ;){
			node = (Node) e3.nextElement();
			for(Enumeration e4 = node.getContexts().elements(); e4.hasMoreElements() ;){
			    context = (Context) e4.nextElement();
			    for(Enumeration e5 = context.getThreads().elements(); e5.hasMoreElements() ;){
				thread = (Thread) e5.nextElement();
				thread.incrementStorage();
				for(Enumeration e6 = thread.getFunctionList().elements(); e6.hasMoreElements() ;){
				    GlobalThreadDataElement ref = (GlobalThreadDataElement) e6.nextElement();
				    //Only want to add an element if this mapping existed on this thread.
				    //Check for this.
				    if(ref != null)
					ref.incrementStorage();
				}
			    }
			}
		    }
	  
		    if(UtilFncs.debug)
			System.out.println("Done increasing the storage for the new counter.");
		}
      
		//Now set the metric name.
		if(metricName == null)
		    metricName = new String("Time");

		System.out.println("Metric name is: " + metricName);
      
		metric = this.getNumberOfMetrics();
		this.addMetric(metricName);

		bSDCounter++;
		//####################################
		//End - Second  Line
		//####################################
      
		//####################################
		//Third Line
		//####################################
		//Do not need the third line.
		inputString = br.readLine();
		if(inputString == null)
		    return;
		bSDCounter++;
		//####################################
		//End - Third Line
		//####################################
      
		while((inputString = br.readLine()) != null){
		    genericTokenizer = new StringTokenizer(inputString, " \t\n\r");

		    int lineType = -1;
		    /*
		      (0) t-exclusive
		      (1) t-inclusive
		      (2) m-exclusive
		      (3) m-inclusive
		      (4) exclusive
		      (5) inclusive
		      (6) userevent
		    */
	    
		    //Determine the lineType.
		    if((inputString.charAt(0)) == 't'){
			if(checkForExcInc(inputString, true, false))
			    lineType = 0;
			else
			    lineType = 1;
		    }
		    else if((inputString.charAt(0)) == 'm'){
			if(checkForExcInc(inputString, true, false))
			    lineType = 2;
			else
			    lineType = 3;
		    }
		    else if(checkForExcInc(inputString, true, true))
			lineType = 4;
		    else if(checkForExcInc(inputString, false, true)) 
			lineType = 5;
		    else if(noue(inputString))
			lineType = 6;
	    
		    //Common things to grab
		    if((lineType!=6) && (lineType!=-1)){
			this.getFunctionDataLine1(inputString);
			mappingID = this.getGlobalMapping().addGlobalMapping(functionDataLine1.s0, 0, 1);
			globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);
		    }

		    switch(lineType){
		    case 0:
			if(this.firstMetric()){ 
			    //Grab the group names.
			    groupNamesString = getGroupNames(inputString);
			    if(groupNamesString != null){
				StringTokenizer st = new StringTokenizer(groupNamesString, " |");
				while (st.hasMoreTokens()){
				    String tmpString = st.nextToken();
				    if(tmpString != null){
					//The potential new group is added here.  If the group is already present, the the addGlobalMapping
					//function will just return the already existing group id.  See the GlobalMapping class for more details.
					int tmpInt = this.getGlobalMapping().addGlobalMapping(tmpString, 1, 1);
					//The group is either already present, or has just been added in the above line.  Now, using the addGroup
					//function, update this mapping to be a member of this group.
					this.getGlobalMapping().addGroup(mappingID, tmpInt, 0);
					if((tmpInt != -1) && (UtilFncs.debug))
					    System.out.println("Adding " + tmpString + " group with id: " + tmpInt + " to mapping: " + functionDataLine1.s0);
				    }    
				}    
			    }
			}
			this.getGlobalMapping().setTotalExclusiveValueAt(metric, functionDataLine1.d0, mappingID, 0);
			break;
		    case 1:
			this.getGlobalMapping().setTotalInclusiveValueAt(metric, functionDataLine1.d0, mappingID, 0);
			break;
		    case 2:
			//Now set the values correctly.
			if((this.getGlobalMapping().getMaxMeanExclusiveValue(metric)) < functionDataLine1.d0){
			    this.getGlobalMapping().setMaxMeanExclusiveValue(metric, functionDataLine1.d0);}
			if((this.getGlobalMapping().getMaxMeanExclusivePercentValue(metric)) < functionDataLine1.d1){
			    this.getGlobalMapping().setMaxMeanExclusivePercentValue(metric, functionDataLine1.d1);}
		    
			globalMappingElement.setMeanExclusiveValue(metric, functionDataLine1.d0);
			globalMappingElement.setMeanExclusivePercentValue(metric, functionDataLine1.d1);
			break;
		    case 3:
			//Now set the values correctly.
			if((this.getGlobalMapping().getMaxMeanInclusiveValue(metric)) < functionDataLine1.d0){
			    this.getGlobalMapping().setMaxMeanInclusiveValue(metric, functionDataLine1.d0);}
			if((this.getGlobalMapping().getMaxMeanInclusivePercentValue(metric)) < functionDataLine1.d1){
			    this.getGlobalMapping().setMaxMeanInclusivePercentValue(metric, functionDataLine1.d1);}
		    
			globalMappingElement.setMeanInclusiveValue(metric, functionDataLine1.d0);
			globalMappingElement.setMeanInclusivePercentValue(metric, functionDataLine1.d1);
		    
			//Set number of calls/subroutines/usersec per call.
			inputString = br.readLine();

			this.getFunctionDataLine2(inputString);

			//Set the values.
			globalMappingElement.setMeanNumberOfCalls(functionDataLine2.i0);
			globalMappingElement.setMeanNumberOfSubRoutines(functionDataLine2.i1);
			globalMappingElement.setMeanUserSecPerCall(metric, functionDataLine2.d0);

			//Set the max values.
			if((this.getGlobalMapping().getMaxMeanNumberOfCalls()) < functionDataLine2.i0)
			    this.getGlobalMapping().setMaxMeanNumberOfCalls(functionDataLine2.i0);

			if((this.getGlobalMapping().getMaxMeanNumberOfSubRoutines()) < functionDataLine2.i1)
			    this.getGlobalMapping().setMaxMeanNumberOfSubRoutines(functionDataLine2.i1);

			if((this.getGlobalMapping().getMaxMeanUserSecPerCall(metric)) < functionDataLine2.d0)
			    this.getGlobalMapping().setMaxMeanUserSecPerCall(metric, functionDataLine2.d0);

			globalMappingElement.setMeanValuesSet(true);
			break;
		    case 4:
			if((globalMappingElement.getMaxExclusiveValue(metric)) < functionDataLine1.d0)
			    globalMappingElement.setMaxExclusiveValue(metric, functionDataLine1.d0);
			if((globalMappingElement.getMaxExclusivePercentValue(metric)) < functionDataLine1.d1)
			    globalMappingElement.setMaxExclusivePercentValue(metric, functionDataLine1.d1);

			//Get the node,context,thread.
			int[] array = this.getNCT(inputString);
			nodeID = array[0];
			contextID = array[1];
			threadID = array[2];

			node = this.getNCT().getNode(nodeID);
			if(node==null)
			    node = this.getNCT().addNode(nodeID);
			context = node.getContext(contextID);
			if(context==null)
			    context = node.addContext(contextID);
			thread = context.getThread(threadID);
			if(thread==null){
			    thread = context.addThread(threadID);
			    thread.initializeFunctionList(this.getNumberOfMappings());
			}

			globalThreadDataElement = thread.getFunction(mappingID);
			
			if(globalThreadDataElement == null){
			    globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(mappingID, 0), false);
			    thread.addFunction(globalThreadDataElement, mappingID);
			}
			globalThreadDataElement.setExclusiveValue(metric, functionDataLine1.d0);
			globalThreadDataElement.setExclusivePercentValue(metric, functionDataLine1.d1);
			//Now check the max values on this thread.
			if((thread.getMaxExclusiveValue(metric)) < functionDataLine1.d0)
			    thread.setMaxExclusiveValue(metric, functionDataLine1.d0);
			if((thread.getMaxExclusivePercentValue(metric)) < functionDataLine1.d1)
			    thread.setMaxExclusivePercentValue(metric, functionDataLine1.d1);
			break;
		    case 5:
			if((globalMappingElement.getMaxInclusiveValue(metric)) < functionDataLine1.d0)
			    globalMappingElement.setMaxInclusiveValue(metric, functionDataLine1.d0);
		    
			if((globalMappingElement.getMaxInclusivePercentValue(metric)) < functionDataLine1.d1)
			    globalMappingElement.setMaxInclusivePercentValue(metric, functionDataLine1.d1);
		    
			thread = this.getNCT().getThread(nodeID,contextID,threadID);
			globalThreadDataElement = thread.getFunction(mappingID);
		    
			globalThreadDataElement.setInclusiveValue(metric, functionDataLine1.d0);
			globalThreadDataElement.setInclusivePercentValue(metric, functionDataLine1.d1);
			if((thread.getMaxInclusiveValue(metric)) < functionDataLine1.d0)
			    thread.setMaxInclusiveValue(metric, functionDataLine1.d0);
			if((thread.getMaxInclusivePercentValue(metric)) < functionDataLine1.d1)
			    thread.setMaxInclusivePercentValue(metric, functionDataLine1.d1);
		    
			//Get the number of calls and number of sub routines
			inputString = br.readLine();
			this.getFunctionDataLine2(inputString);

			//Set the values.
			globalThreadDataElement.setNumberOfCalls(functionDataLine2.i0);
			globalThreadDataElement.setNumberOfSubRoutines(functionDataLine2.i1);
			globalThreadDataElement.setUserSecPerCall(metric, functionDataLine2.d0);

			//Set the max values.
			if(globalMappingElement.getMaxNumberOfCalls() < functionDataLine2.i0)
			    globalMappingElement.setMaxNumberOfCalls(functionDataLine2.i0);
			if(thread.getMaxNumberOfCalls() < functionDataLine2.i0)
			    thread.setMaxNumberOfCalls(functionDataLine2.i0);

			if(globalMappingElement.getMaxNumberOfSubRoutines() < functionDataLine2.i1)
			    globalMappingElement.setMaxNumberOfSubRoutines(functionDataLine2.i1);
			if(thread.getMaxNumberOfSubRoutines() < functionDataLine2.i1)
			    thread.setMaxNumberOfSubRoutines(functionDataLine2.i1);
		    
			if(globalMappingElement.getMaxUserSecPerCall(metric) < functionDataLine2.d0)
			    globalMappingElement.setMaxUserSecPerCall(metric, functionDataLine2.d0);
			if(thread.getMaxUserSecPerCall(metric) < functionDataLine2.d0)
			    thread.setMaxUserSecPerCall(metric, functionDataLine2.d0);
			break;
		    case 6:
			//Just ignore the string if this is not the first check.
			//Assuming is that user events do not change for each counter value.
			if(this.firstMetric()){
			    //The first line will be the user event heading ... skip it.
			    br.readLine();
			    //Now that we know how many user events to expect, we can grab that number of lines.
			    //Note that inputString is still set the the line before the heading which is what we want.
			    int numberOfLines = getNumberOfUserEvents(inputString);
			    for(int j=numberOfLines; j<numberOfLines; j++){
				//Initialize the user list for this thread.
				if(j == 0)
				    (this.getNCT().getThread(nodeID,contextID,threadID)).initializeUsereventList(this.getNumberOfUserEvents());
				
				s1 = br.readLine();
				s2 = br.readLine();
				getUserEventData(s1);
				System.out.println("noc:"+usereventDataLine.i0+"min:"+usereventDataLine.d1+"max:"+usereventDataLine.d0+"mean:"+usereventDataLine.d2);
				
				if(usereventDataLine.i0 != 0){
				    mappingID = this.getGlobalMapping().addGlobalMapping(usereventDataLine.s0, 2, 1);
				    globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 2);
				    globalThreadDataElement = thread.getUserevent(mappingID);

				    if(globalThreadDataElement == null){
					globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(mappingID, 2), true);
					thread.addUserevent(globalThreadDataElement, mappingID);
				    }

				    globalThreadDataElement.setUserEventNumberValue(usereventDataLine.i0);
				    globalThreadDataElement.setUserEventMinValue(usereventDataLine.d1);
				    globalThreadDataElement.setUserEventMaxValue(usereventDataLine.d0);
				    globalThreadDataElement.setUserEventMeanValue(usereventDataLine.d2);

				    if((globalMappingElement.getMaxUserEventNumberValue()) < usereventDataLine.i0)
					globalMappingElement.setMaxUserEventNumberValue(usereventDataLine.i0);
				    if((globalMappingElement.getMaxUserEventMaxValue()) < usereventDataLine.d0)
					globalMappingElement.setMaxUserEventMaxValue(usereventDataLine.d0);
				    if((globalMappingElement.getMaxUserEventMinValue()) < usereventDataLine.d1)
					globalMappingElement.setMaxUserEventMinValue(usereventDataLine.d1);
				    if((globalMappingElement.getMaxUserEventMeanValue()) < usereventDataLine.d2)
					globalMappingElement.setMaxUserEventMeanValue(usereventDataLine.d2);
				}
			    }
			    //Now set the userevents flag.
			    setUserEventsPresent(true);
			}
			break;
		    default:
			if(UtilFncs.debug){
			    System.out.println("Skipping line: " + bSDCounter);
			}
			break;
		    }
		   
		    //Increment the loop counter.
		    bSDCounter++;
		}
	    
		//Close the file.
		br.close();
	    
		if(UtilFncs.debug){
		    System.out.println("The total number of threads is: " + this.getNCT().getTotalNumberOfThreads());
		    System.out.println("The number of mappings is: " + this.getNumberOfMappings());
		    System.out.println("The number of user events is: " + this.getNumberOfUserEvents());
		}

		//Set firstRead to false.
		this.setFirstMetric(false);
	    
		time = (System.currentTimeMillis()) - time;
		System.out.println("Done processing data file!");
		System.out.println("Time to process file (in milliseconds): " + time);
	    }
	    System.out.println("Processing callpath data ...");
	    if(CallPathUtilFuncs.isAvailable(getGlobalMapping().getMappingIterator(0))){
		setCallPathDataPresent(true);
		CallPathUtilFuncs.buildRelations(getGlobalMapping());
	    }
	    else
		System.out.println("No callpath data found.");
	    System.out.println("Done - Processing callpath data!");

	    //Need to notify observers that we are done.  Be careful here.
	    //It is likely that they will modify swing elements.  Make sure
	    //to dump request onto the event dispatch thread to ensure
	    //safe update of said swing elements.  Remember, swing is not thread
	    //safe for the most part.
	    EventQueue.invokeLater(new Runnable(){
		    public void run(){
			TauPprofOutputSession.this.notifyObservers();
		    }
		});
	}
        catch(Exception e){
	    UtilFncs.systemError(e, null, "SSD01");
	}
    }
    
    //####################################
    //Private Section.
    //####################################

    //######
    //Pprof.dat string processing methods.
    //######
    private boolean noue(String s){
	int stringPosition = 0;
	char tmpChar = s.charAt(stringPosition);
	while(tmpChar!='\u0020'){
	    stringPosition++;
	    tmpChar = s.charAt(stringPosition);
	}
	stringPosition++;
	tmpChar = s.charAt(stringPosition);
	if(tmpChar=='u')
	    return true;
	else
	    return false;
    }

    private boolean checkForExcInc(String inString, boolean exclusive, boolean checkString){
	boolean result = false;
    
	try{
	    //In this function I need to be careful.  If the mapping name contains "excl", I
	    //might interpret this line as being the exclusive line when in fact it is not.
      
	    if(checkString){
		StringTokenizer checkTokenizer = new StringTokenizer(inString," ");
		String tmpString2 = checkTokenizer.nextToken();
		if((tmpString2.indexOf(",")) == -1)
		    return result;
	    }
      
	    //Now, we want to grab the substring that occurs AFTER the SECOND '"'.
	    //At present, pprof does not seem to allow an '"' in the mapping name.  So
	    //, I can be assured that I will not find more than two before the "excl" or "incl".
	    StringTokenizer checkQuotesTokenizer = new StringTokenizer(inString,"\"");
      
	    //Need to get the third token.  Could do it in a loop, just as quick this way.
	    String tmpString = checkQuotesTokenizer.nextToken();
	    tmpString = checkQuotesTokenizer.nextToken();
	    tmpString = checkQuotesTokenizer.nextToken();
      
	    //Ok, now, the string in tmpString should include at least "excl" or "incl", and
	    //also, the first token should be either "excl" or "incl".
	    StringTokenizer checkForExclusiveTokenizer = new StringTokenizer(tmpString, " \t\n\r");
	    tmpString = checkForExclusiveTokenizer.nextToken();
      
	    //At last, do the check.  
	    if(exclusive){
		if(tmpString.equals("excl"))
		    result = true;
	    }
	    else{
		if(tmpString.equals("incl"))
		    result = true;
	    }
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SSD04");}
	return result;
    }

    private void getFunctionDataLine1(String string){
	try{
	    StringTokenizer st1 = new StringTokenizer(string, "\"");
	    st1.nextToken();
	    functionDataLine1.s0 = st1.nextToken(); //Name
	    
	    StringTokenizer st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
	    st2.nextToken();
	    functionDataLine1.d0 = Double.parseDouble(st2.nextToken()); //Value
	    functionDataLine1.d1 = Double.parseDouble(st2.nextToken()); //Percent value
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SSD08");
	}
    }

    private void getFunctionDataLine2(String string){ 
	try{
	    StringTokenizer getMappingIDTokenizer = new StringTokenizer(string, " \t\n\r");
	    getMappingIDTokenizer.nextToken();
	    getMappingIDTokenizer.nextToken();
	    getMappingIDTokenizer.nextToken();
	    
	    functionDataLine2.i0 = (int) Double.parseDouble(getMappingIDTokenizer.nextToken()); //Number of calls
	    functionDataLine2.i1 = (int) Double.parseDouble(getMappingIDTokenizer.nextToken()); //Number of subroutines
	    functionDataLine2.d0 = Double.parseDouble(getMappingIDTokenizer.nextToken()); //User seconds per call
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SSD10");
	}
    }

    private void getUserEventData(String string){
	try{
	    StringTokenizer st1 = new StringTokenizer(string, "\"");
	    st1.nextToken();
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

    private String getGroupNames(String string){
	try{  
	    StringTokenizer getMappingNameTokenizer = new StringTokenizer(string, "\"");
 	    getMappingNameTokenizer.nextToken();
	    getMappingNameTokenizer.nextToken();
	    String str = getMappingNameTokenizer.nextToken();
        
	    //Just do the group check once.
	    if(!(this.groupCheck())){
		//If present, "GROUP=" will be in this token.
		int tmpInt = str.indexOf("GROUP=");
		if(tmpInt > 0){
		    this.setGroupNamesPresent(true);
		}
		this.setGroupCheck(true);
	    }
	    
	    if(groupNamesPresent()){
		 str = getMappingNameTokenizer.nextToken();
		    return str;
	    }
	    //If here, this profile file does not track the group names.
	    return null;
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SSD12");
	}
	return null;
    }
  
    private int getNumberOfUserEvents(String string){
	try{
	    StringTokenizer st = new StringTokenizer(string, " \t\n\r");
	    return Integer.parseInt(st.nextToken());
  	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "SSD16");
	}
	return -1;
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
      	}
	catch(Exception e){
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
    private LineData functionDataLine1 = new LineData();
    private LineData functionDataLine2 = new LineData();
    private LineData  usereventDataLine = new LineData();
    //####################################
    //End - Instance data.
    //####################################
}
