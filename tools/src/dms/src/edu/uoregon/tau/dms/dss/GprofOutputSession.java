/* 
   Name:        TauPprofOutputSession.java
   Author:      Robert Bell
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

package edu.uoregon.tau.dms.dss;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

import java.io.*;
import java.util.*;

public class GprofOutputSession extends ParaProfDataSession{

    public GprofOutputSession(){
	super();
	this.setMetrics(new Vector());
    }

    public void run(){
	try{
	    //Record time.
	    long time = System.currentTimeMillis();

	    //######
	    //Frequently used items.
	    //######
	    GlobalMappingElement globalMappingElement = null;
	    GlobalThreadDataElement globalThreadDataElement = null;
	    
	    Node node = null;
	    Context context = null;
	    edu.uoregon.tau.dms.dss.Thread thread = null;
	    
	    String inputString = null;
	    String s1 = null;
	    String s2 = null;
	    
	    String tokenString;
	    StringTokenizer genericTokenizer;
	    
	    int mappingID = -1;
	    int callPathMappingID = -1;
	    GlobalMappingElement gme1 = null;
	    GlobalMappingElement gme2 = null;
	    
	    Vector v = null;
	    File[] files = null;
	    //######
	    //End - Frequently used items.
	    //######

	    System.out.println("In the GprofOutputSession");

	    v = (Vector) initializeObject;
	    for(Enumeration e = v.elements(); e.hasMoreElements() ;){
		files = (File[]) e.nextElement();
		System.out.println("Processing data file, please wait ......");

		FileInputStream fileIn = new FileInputStream(files[0]);
		InputStreamReader inReader = new InputStreamReader(fileIn);
		BufferedReader br = new BufferedReader(inReader);

		//Need to call increaseVectorStorage() on all objects that require it.
		this.getGlobalMapping().increaseVectorStorage();

		//Since this is gprof output, there will only be one node,context, and thread.
		node = this.getNCT().addNode(0);
		context = node.addContext(0);
		thread = context.addThread(0);
		thread.setDebug(this.debug());
		if(this.debug())
		    this.outputToFile("n,c,t: " + 0 + "," + 0 + "," + 0);
		thread.initializeFunctionList(this.getGlobalMapping().getNumberOfMappings(0));

		//Time is the only metric tracked with gprof.
		this.addMetric("Time");
		if(this.debug()){
		    System.out.println("metric name: Time");
		    this.outputToFile("metric name: Time");
		}
		

		boolean callPathSection = false;
		boolean parent = true;
		Vector parents = new Vector();
		LineData self = null;
		Vector children = new Vector();

		while((inputString = br.readLine()) != null){
		    int length = inputString.length();
		    if(length!=0){
			//The first time we see g, set the call path setion to be true,
			//and the second time, set it to be false.
			if(inputString.charAt(0)=='g'){
			    if(!callPathSection){
				System.out.println("###### Call path section ######");
				callPathSection = true;
			    }
			    else{
				System.out.println("###### Summary section ######");
				callPathSection = false;
			    }
			}
			
			if(callPathSection){
			    if(inputString.charAt(0)=='['){
				self = getSelfLineData(inputString);
				parent=false;
			    }
			    else if(inputString.charAt(0)=='-'){
				//Add self to the global mapping.
				mappingID = this.getGlobalMapping().addGlobalMapping(self.s0, 0, 1);
				gme1 = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);

				System.out.println("SELF:"+"["+gme1.getMappingID()+ "]   " +self.s0);
				globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(gme1.getMappingID(), 0), false);
				thread.addFunction(globalThreadDataElement, gme1.getMappingID());
				globalThreadDataElement.setInclusiveValue(0,self.d1+self.d2);
				globalThreadDataElement.setExclusiveValue(0,self.d1);
				globalThreadDataElement.setNumberOfCalls(self.i0);
				globalThreadDataElement.setNumberOfSubRoutines(children.size());
				//globalThreadDataElement.setUserSecPerCall(0,self.d1/self.i0); //Check that this is done using inclusive.
				//Set the max values (thread max values are calculated in the edu.uoregon.tau.dms.dss.Thread class).
				
				int size = parents.size();
				for(int i=0;i<size;i++){
				    LineData lineDataParent = (LineData) parents.elementAt(i);
				    mappingID = this.getGlobalMapping().addGlobalMapping(lineDataParent.s0, 0, 1);
				    String s = lineDataParent.s0 + " => " + self.s0 + "  ";
				    callPathMappingID = this.getGlobalMapping().addGlobalMapping(lineDataParent.s0 + " => " + self.s0 + "  ", 0, 1);
				    System.out.println("call path name:"+this.getGlobalMapping().getGlobalMappingElement(callPathMappingID, 0).getMappingName());
				    this.getGlobalMapping().getGlobalMappingElement(callPathMappingID, 0).setCallPathObject(true);

				    System.out.println("PARENT:"+"["+mappingID+ "] "+lineDataParent.s0);
				    System.out.println("CALLPATH:"+"["+callPathMappingID+ "] "+s);

				    globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(callPathMappingID, 0), false);
				    thread.addFunction(globalThreadDataElement, callPathMappingID);
				    globalThreadDataElement.setInclusiveValue(0,lineDataParent.d0+lineDataParent.d1);
				    globalThreadDataElement.setExclusiveValue(0,lineDataParent.d0);
				    globalThreadDataElement.setNumberOfCalls(lineDataParent.i0);

				}
				parents.clear();
				
				size = children.size();
				for(int i=0;i<size;i++){
				    LineData lineDataChild = (LineData) children.elementAt(i);
				    mappingID = this.getGlobalMapping().addGlobalMapping(lineDataChild.s0, 0, 1);
				    String s = self.s0 + " => " + lineDataChild.s0 + "  ";
				    callPathMappingID = this.getGlobalMapping().addGlobalMapping(self.s0 + " => " + lineDataChild.s0 + "  ", 0, 1);
				    this.getGlobalMapping().getGlobalMappingElement(callPathMappingID, 0).setCallPathObject(true);

				    System.out.println("CHILD:"+"["+mappingID+"]  "+lineDataChild.s0);
				    System.out.println("CALLPATH:"+"["+callPathMappingID+ "] "+s);

				    globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(callPathMappingID, 0), false);
				    thread.addFunction(globalThreadDataElement, callPathMappingID);
				    globalThreadDataElement.setInclusiveValue(0,lineDataChild.d0+lineDataChild.d1);
				    globalThreadDataElement.setExclusiveValue(0,lineDataChild.d0);
				    globalThreadDataElement.setNumberOfCalls(lineDataChild.i0);

				}
				children.clear();
				System.out.println(inputString);
				parent=true;
			    }
			    else if(inputString.charAt(length-1)==']'){
				if(parent)
				    parents.add(getParentLineData(inputString));
				else
				    children.add(getChildLineData(inputString));
			    }
			}
			else if(inputString.charAt(length-1)==']'){
			    System.out.println(getSummaryLineData(inputString).s0);
			}
		    }
		    genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
		}
	    }
	    thread.setThreadData(0);
	    this.setMeanDataAllMetrics(0,this.getNumberOfMetrics());

	    if(CallPathUtilFuncs.isAvailable(getGlobalMapping().getMappingIterator(0))){
		setCallPathDataPresent(true);
		CallPathUtilFuncs.buildRelations(getGlobalMapping());
	    }

	    time = (System.currentTimeMillis()) - time;
	    System.out.println("Done processing data!");
	    System.out.println("Time to process (in milliseconds): " + time);

	    //Need to notify observers that we are done.  Be careful here.
	    //It is likely that they will modify swing elements.  Make sure
	    //to dump request onto the event dispatch thread to ensure
	    //safe update of said swing elements.  Remember, swing is not thread
	    //safe for the most part.
	    EventQueue.invokeLater(new Runnable(){
		    public void run(){
			GprofOutputSession.this.notifyObservers();
		    }
		});
	}
        catch(Exception e){
	    UtilFncs.systemError(e, null, "GOS01");
	}
    }
    
    //####################################
    //Private Section.
    //####################################

    //######
    //Gprof.dat string processing methods.
    //######
    private LineData getSelfLineData(String string){
	LineData lineData = new LineData();
	try{
	    StringTokenizer st = new StringTokenizer(string, " \t\n\r");
	    
	    //In some implementations, the self line will not give
	    //the number of calls for the top level function (usually main).
	    //Check the number of tokens to see if we are in this case.  If so,
	    //by default, we assume a number of calls value of 1.
	    int numberOfTokens = st.countTokens();

	    //Skip the first token.
		// Entries are numbered with consecutive integers. 
		// Each function therefore has an index number, which 
		// appears at the beginning of its primary line. Each 
		// cross-reference to a function, as a caller or 
		// subroutine of another, gives its index number as 
		// well as its name. The index number guides you if 
		// you wish to look for the entry for that function.
	    st.nextToken();

		// This is the percentage of the total time that was 
		// spent in this function, including time spent in 
		// subroutines called from this function. The time 
		// spent in this function is counted again for the 
		// callers of this function. Therefore, adding up these 
		// percentages is meaningless.
	    lineData.d0 = Double.parseDouble(st.nextToken());

		// This is the total amount of time spent in this 
		// function. This should be identical to the number 
		// printed in the seconds field for this function in 
		// the flat profile.
	    lineData.d1 = Double.parseDouble(st.nextToken());

		// This is the total amount of time spent in the 
		// subroutine calls made by this function. This should 
		// be equal to the sum of all the self and children 
		// entries of the children listed directly below this 
		// function.
	    lineData.d2 = Double.parseDouble(st.nextToken());

		// This is the number of times the function was called. 
		// If the function called itself recursively, there are 
		// two numbers, separated by a `+'. The first number 
		// counts non-recursive calls, and the second counts 
		// recursive calls. 
	    if(numberOfTokens!=7)
		lineData.i0 = 1;
	    else {
		String tmpStr = st.nextToken();
		if (tmpStr.indexOf("+") >= 0) {
		} else {
	    	StringTokenizer st2 = new StringTokenizer(tmpStr, "+");
			lineData.i0 = Integer.parseInt(st2.nextToken());
			// do this?
			// lineData.i0 += Integer.parseInt(st2.nextToken());
		}
		}
	    
	    lineData.s0 = st.nextToken(); //Name
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "GOS02");
	}
	return lineData;
    }

    private LineData getParentLineData(String string){
	LineData lineData = new LineData();
	try{
	    StringTokenizer st1 = new StringTokenizer(string, " \t\n\r");
	    
		// get the estimate of the amount of time spent in self when 
		// it was called from parent
	    lineData.d0 = Double.parseDouble(st1.nextToken());
		// get the estimate of the amount of time spent in subroutines 
		// of self when self was called from parent. The sum of the 
		// self and children fields is an estimate of the amount of 
		// time spent within calls to self from parent.
	    lineData.d1 = Double.parseDouble(st1.nextToken());
	    
	   	StringTokenizer st2 = new StringTokenizer(st1.nextToken(), "/");
		// the number of times self was called from parent 
	    lineData.i0 = Integer.parseInt(st2.nextToken());
		// the total number of nonrecursive calls to self from all its parents
	    lineData.i1 = Integer.parseInt(st2.nextToken());

	    lineData.s0 = st1.nextToken(); //Name
	}
	catch(Exception e){
		System.out.println("***\n" + string + "\n***");
		e.printStackTrace();
		UtilFncs.systemError(e, null, "GOS03");
	}
	return lineData;
    }

    private LineData getChildLineData(String string){
	LineData lineData = new LineData();
	try{
	    StringTokenizer st1 = new StringTokenizer(string, " \t\n\r");
	    
		// get the estimate of the amount of time spent directly 
		// in child when it was called from self
	    lineData.d0 = Double.parseDouble(st1.nextToken());
		// get the estimate of the amount of time spent in 
		// subroutines of child when child was called from self. 
		// The sum of the self and children fields is an estimate 
		// of the total time spent in calls to child from self
	    lineData.d1 = Double.parseDouble(st1.nextToken());
	    
		// This ratio is used to determine how much of self and 
		// children time gets credited to parent.
	   	StringTokenizer st2 = new StringTokenizer(st1.nextToken(), "/");
		// get the number of calls to child from self
	    lineData.i0 = Integer.parseInt(st2.nextToken());
		// get the total number of nonrecursive calls to report. 
	    lineData.i1 = Integer.parseInt(st2.nextToken());

	    lineData.s0 = st1.nextToken(); //Name
	}
	catch(Exception e){
		System.out.println("***\n" + string + "\n***");
		e.printStackTrace();
		UtilFncs.systemError(e, null, "GOS03");
	}
	return lineData;
    }

    private LineData getSummaryLineData(String string){
	LineData lineData = new LineData();
	try{
	    StringTokenizer st = new StringTokenizer(string, " \t\n\r");
	    
	    lineData.d0 = Double.parseDouble(st.nextToken());
	    lineData.d1 = Double.parseDouble(st.nextToken());
	    lineData.d2 = Double.parseDouble(st.nextToken());
	    lineData.i0 = Integer.parseInt(st.nextToken());
	    lineData.d3 = Double.parseDouble(st.nextToken());
	    lineData.d4 = Double.parseDouble(st.nextToken());
	    
	    lineData.s0 = st.nextToken(); //Name
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "GOS04");
	}
	return lineData;
    }

    
    //######
    //End - Gprof.dat string processing methods.
    //######

    //####################################
    //End - Private Section.
    //####################################

    //####################################
    //Instance data.
    //####################################
    //####################################
    //End - Instance data.
    //####################################
}
