/* 
   TauOutputSession.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;



import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

import java.io.*;
import java.util.*;
import dms.dss.*;

public class TauOutputSession extends ParaProfDataSession{

    public TauOutputSession(){
	super();
    }

    /**
     * Terminate the DataSession object.
     *
     */
    public void terminate(){}   // formerly "close"

    /**
     * Returns a ListIterator of Application objects.
     *
     * @return	DataSessionIterator object of all Applications.
     * @see	DataSessionIterator
     * @see	Application
     */
    public ListIterator getApplicationList(){
	return null;}

    /**
     * Returns a ListIterator of Experiment objects
     *
     * @return	DataSessionIterator object of all Experiments.  If there is an Application saved in the DataSession, then only the Experiments for that Application are returned.
     * @see	DataSessionIterator
     * @see	Experiment
     * @see	DataSession#setApplication
     */
    public ListIterator getExperimentList(){
	return null;}

    /**
     * Returns a ListIterator of Trial objects
     *
     * @return	DataSessionIterator object of all Trials.  If there is an Application and/or Experiment saved in the DataSession, then only the Trials for that Application and/or Experiment are returned.
     * @see	DataSessionIterator
     * @see	Trial
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     */
    public ListIterator getTrialList(){
	return null;}


    /**
     * Set the Application for this DataSession.  The DataSession object will maintain a reference to the Application referenced by the id.  To clear this reference, call setApplication(Application) with a null reference.
     *
     * @param	id unique id of the Application object to be saved.
     * @see	Application
     * @see	DataSession#setApplication(Application)
     */
    public Application setApplication(int id){
	return null;}
    
    /**
     * Set the Application for this DataSession.  The DataSession object will maintain a reference to the Application referenced by the name and/or version.  To clear this reference, call setApplication(Application) with a null reference.
     *
     * @param	name name of the Application object to be saved.
     * @param	version version of the Application object to be saved.
     * @see	Application
     * @see	DataSession#setApplication(Application)
     */
    public Application setApplication(String name, String version){
	return null;}

    /**
     * Set the Experiment for this DataSession.  The DataSession object will maintain a reference to the Experiment referenced by the id.  To clear this reference, call setExperiment(Experiment) with a null reference.
     *
     * @param	id unique id of the Experiment object to be saved.
     * @see	Experiment
     * @see	DataSession#setExperiment(Experiment)
     */
    public Experiment setExperiment(int id){
	return null;}
    
    /**
     * Set the Trial for this DataSession.  The DataSession object will maintain a reference to the Trial referenced by the id.  To clear this reference, call setTrial(Trial) with a null reference.
     *
     * @param	id unique id of the Trial object to be saved.
     * @see	Trial
     * @see	DataSession#setTrial(Trial)
     */
    public Trial setTrial(int id){
	return null;}

    /**
     * Returns a ListIterator of Function objects.
     *
     * @return	DataSessionIterator object of all Functions.  If there is an Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) saved in the DataSession, then only the Functions for that Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) are returned.
     * @see	DataSessionIterator
     * @see	Function
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     */
    public ListIterator getFunctions(){
	return null;}

    /**
     * Set the Function for this DataSession.  The DataSession object will maintain a reference to the Function referenced by the id.  To clear this reference, call setFunction(Function) with a null reference.
     *
     * @param	id unique id of the Function object to be saved.
     * @see	Function
     * @see	DataSession#setFunction(Function)
     */
    public Function setFunction(int id){
	return null;}

    /**
     * Returns the Function identified by the unique function id.
     *
     * @param	functionID unique id of the Function object.
     * @return	Function object identified by the unique function id.
     * @see	Function
     */
    public Function getFunction(int functionID){
	return null;}
	
    /**
     * Returns a ListIterator of UserEvent objects.
     *
     * @return	DataSessionIterator object of all UserEvents.  If there is an Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) saved in the DataSession, then only the UserEvents for that Application, Experiment, Trial(s), node(s), context(s) and/or thread(s) are returned.
     * @see	DataSessionIterator
     * @see	UserEvent
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     */
    public ListIterator getUserEvents(){
	return null;}

    /**
     * Set the UserEvent for this DataSession.  The DataSession object will maintain a reference to the UserEvent referenced by the id.  To clear this reference, call setUserEvent(UserEvent) with a null reference.
     *
     * @param	id unique id of the UserEvent object to be saved.
     * @see	UserEvent
     * @see	DataSession#setUserEvent(UserEvent)
     */
    public UserEvent setUserEvent(int id){
	return null;}

    /**
     * Returns the UserEvent identified by the unique user event id.
     *
     * @param	userEventID unique id of the UserEvent object to be saved.
     * @return	UserEvent object identified by the unique user event id.
     * @see	UserEvent
     */
    public UserEvent getUserEvent(int userEventID){
	return null;}
	
    /**
     * Returns the FunctionData for this DataSession
     *
     * @return	DataSessionIterator of FunctionData objects.  If there is an Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or Function(s) saved in the DataSession, then only the Functions for that Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or Function(s) are returned.
     * @see	DataSessionIterator
     * @see	FunctionDataObject
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     * @see	DataSession#setFunction
     */
    public ListIterator getFunctionData(){
	return null;}

    /**
     * Returns the UserEventData for this DataSession
     *
     * @return	DataSessionIterator of UserEventData objects.  If there is an Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or Function(s) saved in the DataSession, then only the UserEvents for that Application, Experiment, Trial(s), node(s), context(s), thread(s) and/or Function(s) are returned.
     * @see	DataSessionIterator
     * @see	UserEventDataObject
     * @see	DataSession#setApplication
     * @see	DataSession#setExperiment
     * @see	DataSession#setTrial
     * @see	DataSession#setNode
     * @see	DataSession#setContext
     * @see	DataSession#setThread
     * @see	DataSession#setUserEvent
     */
    public ListIterator getUserEventData(){
	return null;}

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
	    
	    //A loop counter.
	    int bSDCounter = 0;
	    
	    int numberOfUserEvents = 0;

	    Vector v = null;
	    File[] files = null;
	    //######
	    //End - Frequently used items.
	    //######
	    v = (Vector) obj;
	    for(Enumeration e = v.elements(); e.hasMoreElements() ;){
		files = (File[]) e.nextElement();
		for(int i=0;i<files.length;i++){
		    System.out.println("Processing file: " + files[i].getName());
		    int[] nct = this.getNCT(files[i].getName());
		    System.out.println("n,c,t: " + nct[0] + "," + nct[1] + "," + nct[2]);
		    
		    //####################################
		    //Second  Line
		    //####################################
		    //This is an important line.
		    inputString = br.readLine();
		    genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
		    //It's first token will be the number of mappings present.  Get it.
		    tokenString = genericTokenizer.nextToken();
		    
		    if(!(this.firstMetric())){
			if((this.getNumberOfMappings()) != Integer.parseInt(tokenString)){
			    System.out.println("***********************");
			    System.out.println("The number of mappings does not match!!!");
			    System.out.println("");
			    System.out.println("To add to an existing run, you must be choosing from");
			    System.out.println("a list of multiple metrics from that same run!!!");
			    System.out.println("***********************");
			    
			    return;
			}
		    }
		    
		    //Set the counter name.
		    String counterName = getCounterName(inputString);
		    
		    //Ok, we are adding a counter name.  Since nothing much has happened yet, it is a
		    //good place to initialize a few things.
		    
		    //Need to call increaseVectorStorage() on all objects that require it.
		    this.increaseVectorStorage();
		    
		//Only need to call addDefaultToVectors() if not the first run.
		if(!(this.firstMetric())){
		    if(ParaProf.debugIsOn)
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
	  
		    if(ParaProf.debugIsOn)
			System.out.println("Done increasing the storage for the new counter.");
		}
      
		//Now set the counter name.
		if(counterName == null)
		    counterName = new String("Time");

		System.out.println("Counter name is: " + counterName);
      
		Metric metricRef = this.addMetric();
		metricRef.setName(counterName);
		metric = metricRef.getID();
		System.out.println("The number of mappings in the system is: " + tokenString);
      
		bSDCounter++;
		//####################################
		//End - Second  Line
		//####################################




		}
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
	nct[0] = Integer.parseInt(st.nextToken());
	nct[1] = Integer.parseInt(st.nextToken());
	nct[2] = Integer.parseInt(st.nextToken());
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

    //######
    //End - profile.*.*.* string processing methods.
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
