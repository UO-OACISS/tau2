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

package dms.dss;

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
	    //######
	    //Frequently used items.
	    //######
	    GlobalMappingElement globalMappingElement = null;
	    GlobalThreadDataElement globalThreadDataElement = null;
	    
	    Node node = null;
	    Context context = null;
	    dms.dss.Thread thread = null;
	    
	    String inputString = null;
	    String s1 = null;
	    String s2 = null;
	    
	    String tokenString;
	    StringTokenizer genericTokenizer;
	    
	    int mappingID = -1;
	    
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
		long time = System.currentTimeMillis();

		FileInputStream fileIn = new FileInputStream(files[0]);
		InputStreamReader inReader = new InputStreamReader(fileIn);
		BufferedReader br = new BufferedReader(inReader);

		//Since this is gprof output, there will only be one node,context, and thread.
		node = this.getNCT().addNode(0);
		context = node.addContext(0);
		thread = context.addThread(0);
		thread.setDebug(this.debug());
		if(this.debug())
		    this.outputToFile("n,c,t: " + 0 + "," + 0 + "," + 0);

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
				int size = parents.size();
				for(int i=0;i<size;i++){
				    LineData lineDataParent = (LineData) parents.elementAt(i);
				    mappingID = this.getGlobalMapping().addGlobalMapping(lineDataParent.s0, 0, 1);
				    System.out.println("PARENT:"+"["+mappingID + "] " +lineDataParent.s0); 
				}
				parents.clear();
				mappingID = this.getGlobalMapping().addGlobalMapping(self.s0, 0, 1);
				System.out.println("SELF:"+"["+mappingID + "]   " +self.s0);
				size = children.size();
				for(int i=0;i<size;i++){
				    LineData lineDataChild = (LineData) children.elementAt(i);
				    mappingID = this.getGlobalMapping().addGlobalMapping(lineDataChild.s0, 0, 1);
				    System.out.println("CHILD:"+"["+mappingID + "]  " +lineDataChild.s0); 
				}
				children.clear();
				System.out.println(inputString);
				parent=true;
			    }
			    else if(inputString.charAt(length-1)==']'){
				if(parent)
				    parents.add(getParentChildLineData(inputString));
				else
				    children.add(getParentChildLineData(inputString));
			    }
			}
			else if(inputString.charAt(length-1)==']'){
			    System.out.println(getSummaryLineData(inputString).s0);
			}
		    }
		    genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
		}

	    System.exit(0);
	    }
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
	    st.nextToken();
	    lineData.d0 = Double.parseDouble(st.nextToken());
	    lineData.d1 = Double.parseDouble(st.nextToken());
	    lineData.d2 = Double.parseDouble(st.nextToken());

	    if(numberOfTokens!=7)
		lineData.i0 = 1;
	    else
		lineData.i0 = Integer.parseInt(st.nextToken());
	    
	    lineData.s0 = st.nextToken(); //Name
	}
	catch(Exception e){
	    UtilFncs.systemError(e, null, "GOS02");
	}
	return lineData;
    }

    private LineData getParentChildLineData(String string){
	LineData lineData = new LineData();
	try{
	    StringTokenizer st1 = new StringTokenizer(string, " \t\n\r");
	    
	    lineData.d0 = Double.parseDouble(st1.nextToken());
	    lineData.d1 = Double.parseDouble(st1.nextToken());
	    
	    StringTokenizer st2 = new StringTokenizer(st1.nextToken(), "/");
	    lineData.i0 = Integer.parseInt(st2.nextToken());;
	    lineData.i1 = Integer.parseInt(st2.nextToken());;

	    lineData.s0 = st1.nextToken(); //Name
	}
	catch(Exception e){
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
