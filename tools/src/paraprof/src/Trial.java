/*
  Trial.java
  
  
  Title:      ParaProf
  Author:     Robert Bell
  Description:  This class is the heart of Racy's static data system.
  This class is rather an ongoing project.  Much work needs
  to be done with respect to data format.
  The use of tokenizers here could impact the performance
  with large data sets, but for now, this will be sufficient.
  The naming and creation of the tokenizers has been done mainly
  to improve the readability of the code.
          
  It must also be noted that the correct funtioning of this
  class is heavily dependent on the format of the pprof -d format.
  It is NOT resistant to change in that format at all.
*/

package paraprof;

import java.io.*;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;

public class Trial{
    //Constructor.
    public Trial(Experiment inParentExp){
	parentExperiment = inParentExp;
	globalMapping = new GlobalMapping(this);
	systemEvents = new SystemEvents();
	nodes = new Vector();
	counterName = null;
	isUserEventHeadingSet = false;
    }
  
    public Experiment getParentExperiment(){
	return parentExperiment;
    }
  
    public void setDMTN(DefaultMutableTreeNode inNode){
	nodeRef = inNode;
    }
  
    public DefaultMutableTreeNode getDMTN(){
	return nodeRef;
    }
  
    public void setTrialName(String inString){
	trialName = inString;
    }
  
    public String getTrialName(){
	return trialName;
    }
  
    public void setTrialID(int inTrialID){
	trialID = inTrialID;
	//Since the parentApplication is set in the constructor,
	//it is not null.  Therefore we can safely set the experimentIDString.
	trialIDString = parentExperiment.getExperimentIDString() + trialID;
    }
  
    public int getTrialID(){
	return trialID;
    }
  
    public String getTrialIDString(){
	return trialIDString;
    }
  
    public Value addValue(){
	Value newValue = new Value(this);
	newValue.setValueID((values.size()));
	values.add(newValue);
	return newValue;
    }
  
    public ColorChooser getColorChooser(){
	return clrChooser;
    }
  
    public Preferences getPreferences(){
	return preferences;
    }
  
    public void setProfilePathName(String inString){
	profilePathName = inString;
    }
  
    public void setProfilePathNameReverse(String inString){
	profilePathNameReverse = inString;
    }
  
    public String getProfilePathName(){
	return profilePathName;
    }
  
    public String getProfilePathNameReverse(){
	return profilePathNameReverse;
    }
  
    public Vector getValues(){
	return values;
    }
  
    public int getValuePosition(String inString){
	int counter = 0;
	for(Enumeration e = values.elements(); e.hasMoreElements() ;){
	    Value tmpValue = (Value) e.nextElement();
	    if((tmpValue.getValueName()).equals(inString))
		return counter;
	    counter++;
	}
	return -1;
    }
  
    public String toString(){
	return trialName;}
  
    public int getNumberOfNodes(){
	return nodes.size();}

    public Vector getNodes(){
	return nodes;}

    public Node getNode(int nodeID){
	return (Node) nodes.elementAt(nodeID);}

    //Returns the total number of contexts in this trial.
    public int getTotalNumberOfContexts(){
	if(totalNumberOfContexts==-1){
	    for(Enumeration e = this.getNodes().elements(); e.hasMoreElements() ;){
	     Node node = (Node) e.nextElement();
	     totalNumberOfContexts+=(node.getNumberOfContexts());
	    }
	}
	return totalNumberOfContexts;
    }

    //Returns the number of contexts on the specified node.
    public int getNumberOfContexts(int nodeID){
	return ((Node) nodes.elementAt(nodeID)).getNumberOfContexts();}

    public Vector getContexts(int nodeID){
	return (this.getNode(nodeID)).getContexts();}

    public Context getContext(int nodeID, int contextID){
	return (this.getNode(nodeID)).getContext(contextID);}

    //Returns the total number of threads in this trial.
    public int getTotalNumberOfThreads(){
	if(totalNumberOfThreads==-1){
	    for(Enumeration e1 = this.getNodes().elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    totalNumberOfThreads+=(context.getNumberOfThreads());
		}
	    }
	}
	return totalNumberOfThreads;
    }

    //Returns the number of threads on the specified node,context.
    public int getNumberOfThreads(int nodeID, int contextID){
	return (this.getContext(nodeID,contextID)).getNumberOfThreads();}

    public Vector getThreads(int nodeID, int contextID){
	return (this.getContext(nodeID,contextID)).getThreads();}

    public Thread getThread(int nodeID, int contextID, int threadID){
	return (this.getContext(nodeID,contextID)).getThread(threadID);}

    //Returns an array of length 3 which contains the maximum number
    //of nodes, contexts and threads reached. This is not a total,
    //except in the case of the number of nodes.
    //This method is useful for determining an upper bound on the
    //string sizes being displayed in ParaProf's different windows.
    //Computation will only occur during the first call to this method.
    //Subsequent calls will return the results obtained from the first
    //call.
    public int[] getMaxNCTNumbers(){
	if(nct==null){
	    nct = new int[3];
	    for(int i=0;i<3;i++){
		nct[i]=0;}
	    nct[0] = this.getNumberOfNodes();
	    for(Enumeration e1 = (this.getNodes()).elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		if(node.getNumberOfContexts()>nct[1])
		    nct[1]=node.getNumberOfContexts();
		for(Enumeration e2 = (node.getContexts()).elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    if(context.getNumberOfThreads()>nct[2])
			nct[2]=context.getNumberOfThreads();
		}
	    }
	}
	return nct;
    }
    

    //####################################
    //Functions that control the obtaining and the openning
    //and closing of the static main window for
    //this trial.
    //####################################
    public StaticMainWindow getStaticMainWindow(){
	return sMW;
    }
  
    public void showMainWindow(){
  
	if(sMW == null){
	    sMW = new StaticMainWindow(this);
	    sMW.setVisible(true);
	    this.getSystemEvents().addObserver(sMW);
	}
	else{
	    this.getSystemEvents().addObserver(sMW);
	    sMW.show();
	}
    }
  
    public void closeStaticMainWindow(){
	if(sMW != null){
	    this.getSystemEvents().deleteObserver(sMW);
	    sMW.setVisible(false);
	}
    }
    //####################################
    //Functions that control the openning and closing of the static main window for
    //this trial.
    //####################################
  
    public SystemEvents getSystemEvents(){
	return systemEvents;
    }
  
    public void setCurValLoc(int inValueLocation){
	currentValueLocation = inValueLocation;
    }
  
    public int getCurValLoc(){
	return currentValueLocation;
    }
  
  
    public void addDefaultToVectors(){
	maxMeanInclusiveValueList.add(new Double(0));
	maxMeanExclusiveValueList.add(new Double(0));
	maxMeanInclusivePercentValueList.add(new Double(0));
	maxMeanExclusivePercentValueList.add(new Double(0));
	maxMeanUserSecPerCallList.add(new Double(0));
	totalMeanInclusiveValueList.add(new Double(0));
	totalMeanExclusiveValueList.add(new Double(0));
    }
  
    //The following funtion initializes the GlobalMapping object.
    //Since we are in the static mode, the number of mappings is known,
    //therefore, the appropriate number of GlobalMappingElements are created.
    void initializeGlobalMapping(int inNumberOfMappings, int mappingSelection){
	for(int i=0; i<inNumberOfMappings; i++){
	    //globalMapping.addGlobalMapping("Error ... the mapping name has not been set!");
	    globalMapping.addGlobalMapping(null, mappingSelection);
	}
    }
  
    //Rest of the public functions.
    GlobalMapping getGlobalMapping(){
	return globalMapping;
    }
  
    public String getCounterName(){
	Value tmpValue = (Value) values.elementAt(currentValueLocation);
	return tmpValue.getValueName();
    }

    public boolean isTimeMetric(){
	String trialName = this.getCounterName();
	trialName = trialName.toUpperCase();
	if(trialName.indexOf("TIME") == -1)
	    return false;
	else
	    return true;
    }
  
    //####################################
    //Useful functions to help the drawing windows.
    //
    //For the most part, these functions just return data
    //items that are easier to calculate whilst building the global
    //lists
    //####################################
    public boolean firstRead(){
	return firstRead;}

    public int getNumberOfMappings(){
	return numberOfMappings;}
  
    public int getNumberOfUserEvents(){
	return numberOfUserEvents;}
  
    public void setMaxMeanInclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanInclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    public double getMaxMeanInclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanInclusiveValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
  
    public void setMaxMeanExclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanExclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    public double getMaxMeanExclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanExclusiveValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
  
    public void setMaxMeanInclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanInclusivePercentValueList.add(dataValueLocation, tmpDouble);}
  
    public double getMaxMeanInclusivePercentValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanInclusivePercentValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
  
    public void setMaxMeanExclusivePercentValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanExclusivePercentValueList.add(dataValueLocation, tmpDouble);}
  
    public double getMaxMeanExclusivePercentValue(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanExclusivePercentValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
  
    public void setMaxMeanNumberOfCalls(double inDouble){
	maxMeanNumberOfCalls = inDouble;}
  
    public double getMaxMeanNumberOfCalls(){
	return maxMeanNumberOfCalls;}
  
    public void setMaxMeanNumberOfSubRoutines(double inDouble){
	maxMeanNumberOfSubRoutines = inDouble;}
  
    public double getMaxMeanNumberOfSubRoutines(){
	return maxMeanNumberOfSubRoutines;}
  
    public void setMaxMeanUserSecPerCall(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	maxMeanUserSecPerCallList.add(dataValueLocation, tmpDouble);}
  
    public double getMaxMeanUserSecPerCall(int dataValueLocation){
	Double tmpDouble = (Double) maxMeanUserSecPerCallList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
    
    
    public void setTotalMeanInclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	totalMeanInclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    public double getTotalMeanInclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) totalMeanInclusiveValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
  
    public void setTotalMeanExclusiveValue(int dataValueLocation, double inDouble){
	Double tmpDouble = new Double(inDouble);
	totalMeanExclusiveValueList.add(dataValueLocation, tmpDouble);}
  
    public double getTotalMeanExclusiveValue(int dataValueLocation){
	Double tmpDouble = (Double) totalMeanExclusiveValueList.elementAt(dataValueLocation);
	return tmpDouble.doubleValue();}
  
    public boolean groupNamesPresent(){
	return groupNamesPresent;
    }
  
    private void setUserEventsPresent(boolean inBoolean){
	userEventsPresent = inBoolean;
    }
  
    public boolean userEventsPresent(){
	return userEventsPresent;
    }

    private void setCallPathDataPresent(boolean inBoolean){
	callPathDataPresent = inBoolean;
    }
  
    public boolean callPathDataPresent(){
	return callPathDataPresent;
    }
    //******************************
    //End - Useful functions to help the drawing windows.
    //******************************
  
  
    //******************************
    //Instance data.
    //******************************
    Experiment parentExperiment = null;
    DefaultMutableTreeNode nodeRef = null;
    private String trialName = null;
    private int trialID = -1;
    private String trialIDString = null;
  
    private Vector values = new Vector();
  
    private SystemEvents systemEvents = null;
    private StaticMainWindow sMW = null;
    private ColorChooser clrChooser = new ColorChooser(this, null);
    private Preferences  preferences = new  Preferences(this, null);
  
    //The file that the is currently being worked on by the
    //running thread.
    private File currentFile = null;
  
    //Flag indicating whether we have already processed a file in
    //this trial.  This cuts down on repeated work.
    private boolean firstRead = true;    
  
    private String profilePathName = null;
    private String profilePathNameReverse = null;
    //private Vector valueNameList = new Vector(); 
    private int currentValueLocation = 0;
    private int currentValueWriteLocation = 0;    
  
    private GlobalMapping globalMapping;
    private Vector nodes;
    private String counterName;
    private boolean isUserEventHeadingSet;
    boolean groupNamesCheck = false;
    boolean groupNamesPresent = false;
    boolean userEventsPresent = false;
    boolean callPathDataPresent = false;
    int bSDCounter;

    private int numberOfMappings = -1;
    private int numberOfUserEvents = -1;
    private int totalNumberOfContexts = -1;
    private int totalNumberOfThreads = -1;
    private int[] nct = null;
  
    //Max mean values.
    private Vector maxMeanInclusiveValueList = new Vector();
    private Vector maxMeanExclusiveValueList = new Vector();
    private Vector maxMeanInclusivePercentValueList = new Vector();
    private Vector maxMeanExclusivePercentValueList = new Vector();
    private double maxMeanNumberOfCalls = 0;
    private double maxMeanNumberOfSubRoutines = 0;
    private Vector maxMeanUserSecPerCallList = new Vector();
  
    private Vector totalMeanInclusiveValueList = new Vector();
    private Vector totalMeanExclusiveValueList = new Vector();
  
  
    //******************************
    //End - Instance data.
    //******************************

}

class UserEventData{
    public int node = -1;
    public int context = -1;
    public int threadID = -1;
    public int id = -1;
    public String name = null;
    public int noc = -1;
    public double max = -1.0;
    public double min = -1.0;
    public double mean = -1.0;
    public double std = -1.0;
}

class UnexpectedStateException extends Exception{
    public UnexpectedStateException(){}
    public UnexpectedStateException(String err){
	super("UnexpectedStateException - message: " + err);
    }
}

