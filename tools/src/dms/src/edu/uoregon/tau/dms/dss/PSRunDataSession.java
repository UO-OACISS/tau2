/* 
   Name:        PSRunDataSession.java
   Author:      Kevin Huck
   Description: Parse an psrun XML data file.
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
import org.xml.sax.*;
import org.xml.sax.helpers.*;

public class PSRunDataSession extends ParaProfDataSession{

    public PSRunDataSession(){
		super();
		this.setMetrics(new Vector());
    }

    public void initialize(Object initializeObject){
		boolean firstFile = true;
		try{
	    	v = (Vector) initializeObject;
			// create our XML parser
        	XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");
        	DefaultHandler handler = new PSRunLoadHandler();
        	xmlreader.setContentHandler(handler);
        	xmlreader.setErrorHandler(handler);
	    	for(Enumeration e = v.elements(); e.hasMoreElements() ;){
				files = (File[]) e.nextElement();
				for (int i = 0 ; i < files.length ; i++) {
					System.out.println("Processing data file, please wait ......");
					long time = System.currentTimeMillis();

	    			StringTokenizer st = new StringTokenizer(files[i].getName(), ".");

					if (st.countTokens() == 3) {
						// increment the node counter - there's a file for each node
						nodeID++;
					} else {
						String prefix = st.nextToken();

						String tid = st.nextToken();
						threadID = Integer.parseInt(tid);

						String nid = st.nextToken();
						Integer tmpID = (Integer)nodeHash.get(nid);
						if (tmpID == null) {
							nodeID = nodeHash.size();
							nodeHash.put(nid, new Integer(nodeID));
						} else {
							nodeID = tmpID.intValue();
						}
					}

      
	  				// parse the next file
		        	xmlreader.parse(new InputSource(new FileInputStream(files[i])));

					// initialize the thread/node
					initializeThread();

					// get the data, and put it in the mapping
					PSRunLoadHandler tmpHandler = (PSRunLoadHandler)handler;
					Hashtable metricHash = tmpHandler.getMetricHash();
					for (Enumeration keys = metricHash.keys(); keys.hasMoreElements(); ) {
						String key = (String)keys.nextElement();
						String value = (String)metricHash.get(key);
						processHardwareCounter(key, value);
					}

					// generate summary statistics
					//Remove after testing is complete.
					//this.setMeanDataAllMetrics(0);
	    
					if(UtilFncs.debug){
			    		System.out.println("The total number of threads is: " + 
							this.getNCT().getTotalNumberOfThreads());
			    		System.out.println("The number of mappings is: " + 
							this.getGlobalMapping().getNumberOfMappings(0));
		    			System.out.println("The number of user events is: " + 
							this.getGlobalMapping().getNumberOfMappings(2));
					}

					time = (System.currentTimeMillis()) - time;
					System.out.println("Done processing data file!");
					System.out.println("Time to process file (in milliseconds): " + time);
				}
	    	}
		//Generate derived data.
		this.generateDerivedData(0);

		} catch(Exception e) {
			e.printStackTrace();
	    	UtilFncs.systemError(e, null, "SSD01");
		}
    }
    
    //####################################
    //Private Section.
    //####################################

	private void initializeThread() {
		// create the mapping, if necessary
		mappingID = this.getGlobalMapping().addGlobalMapping("Entire application", 0, 1);
		globalMappingElement = this.getGlobalMapping().getGlobalMappingElement(mappingID, 0);

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
			thread.setDebug(this.debug());
			thread.initializeFunctionList(this.getGlobalMapping().getNumberOfMappings(0));
			thread.initializeUsereventList(this.getGlobalMapping().getNumberOfMappings(2));
		}

		globalThreadDataElement = thread.getFunction(mappingID);
		if(globalThreadDataElement == null) {
			globalThreadDataElement = new GlobalThreadDataElement(this.getGlobalMapping().getGlobalMappingElement(mappingID, 0), false);
			thread.addFunction(globalThreadDataElement, mappingID);
		}
	}

	private void processHardwareCounter(String key, String value) {
		thread.incrementStorage();
		globalMappingElement.incrementStorage();
		globalThreadDataElement.incrementStorage();
		try{
	    	double eventValue = 0;
	    	eventValue = Double.parseDouble(value);

			metric = this.getNumberOfMetrics();
			//Set the metric name.
			Metric newMetric = this.addMetric(key);
			if (metric < this.getNumberOfMetrics()) {
				//Need to call increaseVectorStorage() on all objects that require it.
				this.getGlobalMapping().increaseVectorStorage();
			}
			metric = newMetric.getID();

			globalThreadDataElement.setExclusiveValue(metric, eventValue);
			globalThreadDataElement.setInclusiveValue(metric, eventValue);
			globalThreadDataElement.setUserSecPerCall(metric, eventValue);
			globalThreadDataElement.setNumberOfCalls(1);
			globalThreadDataElement.setNumberOfSubRoutines(0);

            if((globalMappingElement.getMaxExclusiveValue(metric)) < eventValue) {
				globalMappingElement.setMaxExclusiveValue(metric, eventValue);
				globalMappingElement.setMaxInclusiveValue(metric, eventValue);
			}
			if(globalMappingElement.getMaxUserSecPerCall(metric) < eventValue)
				globalMappingElement.setMaxUserSecPerCall(metric, eventValue);
			globalMappingElement.setMaxNumberOfCalls(1);
			globalMappingElement.setMaxNumberOfSubRoutines(0);

		} catch(Exception e) {
	    	System.out.println("An error occured while parsing the XML data!");
	    	e.printStackTrace();
		}
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
	private Hashtable nodeHash = new Hashtable();
	//######
	//End - Frequently used items.
	//######
}
