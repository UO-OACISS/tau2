package translator;

import java.io.*;
import java.util.*;
import java.net.*;

public class Translator implements Serializable{

    private File readPprof;
    private File writeXml;
    private String metricStr;
    private String trialTime;

    /* This variable connects translator to DB in order to check whether
       the app. and exp. associated with the trial data do exist there. */
    private perfdb.ConnectionManager connector;

    int maxNode;
    int maxContext;
    int maxThread;
    String funAmount;	
    int ueAmount=0;

    // NodeList stores globalnodes
    private Vector NodeList;

    //globalMapping stores total and mean information. 
    GlobalMapping globalMapping;

    private String heading;
    private String userEventHeading;
    private boolean isUserEventHeadingSet;

    //constructor
    public Translator(String configFileName, String sourcename, String targetname) {

	readPprof = new File(sourcename);

	// Get the creation time of pprof.dat

	Date date = new Date(readPprof.lastModified());
        trialTime = date.toString();

	writeXml = new File(targetname);
	
	if (readPprof.exists()){
	    System.out.println("Found "+ sourcename + " ... Loading");
	}
	else {
		System.out.println("Did not find pprof.dat file!"); 
		System.exit(-1);
	}

       	if (!writeXml.exists()){
	    try {
		if (writeXml.createNewFile()){
		    System.out.println("Create pprof.xml!");
		}
	    }
	    catch(Exception e){
		e.printStackTrace();	
	    }
        }    
	
	globalMapping = new GlobalMapping();
       	NodeList = new Vector();

	heading = null;
	userEventHeading = null;
	isUserEventHeadingSet = false;

	maxNode = 0;
	maxContext = 0;
	maxThread = 0;

	connector = new perfdb.ConnectionManager(configFileName);
	connector.connect();
    }

    public Vector getNodeList()
    {
       	return NodeList;
    }


    void initializeGlobalMapping(int inNumberOfFunctions)
    {
       	for(int i=0; i<inNumberOfFunctions; i++)
       	{
       		globalMapping.addGlobalFunction("Name has not been set!");
       	}
    }


    GlobalMapping getGlobalMapping()
    {
       	return globalMapping;
    }


    //the core function of the class: read in pprof -d file

    public void buildPprof(){
	try{

	    BufferedReader preader = new BufferedReader(new FileReader(readPprof));

	    //intermediate strings
	    String inputString;
	    String tokenString;
	    
	    String userEventNameString;

	    StringTokenizer genericTokenizer;
	    
	    int functionID = -1;
	    int userEventID = -1;
	    String functionName;
	    String functionGroup;
	    int node = -1;
	    int context = -1;
	    int thread = -1;

	    double value = -1;
	    double percentValue = -1;
	    double calls = -1;
	    double subrs = -1;
	    double inclpcall = -1;
	    GlobalMappingElement tmpGlobalMappingElement;

	    GlobalNode currentGlobalNode = null;
      	    GlobalContext currentGlobalContext = null;
	    GlobalThread currentGlobalThread = null;
	    GlobalThreadDataElement tmpGlobalThreadDataElement = null;			

	    int lastNode = -1;
	    int lastContext = -1;
	    int lastThread = -1;
	    int numOfFunctions = 0;


	    int counter = 0;

	    //loop counter
	    int i = 0;

	    //read in the pprof file
	    while ((inputString = preader.readLine())!= null){
		//System.out.println(inputString);
		// skip the first line
		if (i>0){
	
		    // Obtain the number of functions.
		    if (i==1) 
			funAmount = getFunAmt(inputString);	   
 
		    genericTokenizer = new StringTokenizer(inputString, " \t\n\r");

		    if(i!=1){
			
			if (i!=2){
			    
			    if(checkForBeginningT(inputString)){

				counter++;
				if(checkForExclusiveWithTOrM(inputString)){
				    
				    functionName = getFunctionName(inputString);

				    functionID = getFunctionID(inputString);
				   				    				    
				    if(!(globalMapping.setFunctionNameAt(functionName, functionID)))
				       	System.out.println("There was an error adding function to the global mapping");

				    if (isAGroupMember(inputString)){
					functionGroup = getFunctionGroup(inputString);
					if(!(globalMapping.setFunctionGroupAt(functionGroup, functionID)))
					    System.out.println("There was an error adding function to the global mapping");
				    }
				    
				    value = getValue(inputString);				    
				    percentValue = getPercentValue(inputString);
				    

				    if(!(globalMapping.setTotalExclusiveValueAt(value, functionID)))
				        System.out.println("There was an error setting Exc/Inc total time"); 
				    else globalMapping.setTotalExclusivePercentValueAt(percentValue, functionID); 
				}
				else if(checkForInclusiveWithTOrM(inputString)){
				    
				    
			       	    functionID = getFunctionID(inputString);

			            //Grab the value.
			            value = getValue(inputString);
				    percentValue = getPercentValue(inputString);

	      			    //Set the value for this function.
				    if(!(globalMapping.setTotalInclusiveValueAt(value, functionID)))
				  	System.out.println("There was an error setting Exc/Inc total time");
				    else{ 
					globalMapping.setTotalInclusivePercentValueAt(percentValue, functionID); 

					//Set the total stat string.
					//The next string in the file should be the correct one.  Assume it.
					//If the file format changes, we are all screwed anyway.
				
					inputString = preader.readLine();
					
					//Set the total stat string.
					//Now extract the other info from this string.******************ADD
					calls = getCall(inputString);
				        subrs = getSubrs(inputString);
				        inclpcall = getInclPCall(inputString);
				        globalMapping.getGlobalMappingElement(functionID).setTotalCall(calls);
				        globalMapping.getGlobalMappingElement(functionID).setTotalSubrs(subrs);
				        globalMapping.getGlobalMappingElement(functionID).setTotalInclPCall(inclpcall);													
				    }
				}
			    }
			    else if(checkForBeginningM(inputString)){
				
				if(checkForExclusiveWithTOrM(inputString))
			        {
					//Grab the function ID.
			      	       	functionID = getFunctionID(inputString);
			      		//Grab the value.
			     		value = getValue(inputString);
					percentValue = getPercentValue(inputString);
									
					//Grab the correct global mapping element.
					tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(functionID);	
				 	tmpGlobalMappingElement.setMeanExclusiveValue(value);
					tmpGlobalMappingElement.setMeanExclusivePercentValue(percentValue);
				}
				else if(checkForInclusiveWithTOrM(inputString))
				{
			       		//Grab the function ID.
					functionID = getFunctionID(inputString);

					//Grab the value.
					value = getValue(inputString);
					percentValue = getPercentValue(inputString);
														
					//Grab the correct global mapping element.
					tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(functionID);	
					tmpGlobalMappingElement.setMeanInclusiveValue(value);
					tmpGlobalMappingElement.setMeanInclusivePercentValue(percentValue);
									
					//Set the total stat string.
		      			//The next string in the file should be the correct one.  Assume it.
					//If the file format changes, we are all screwed anyway.

					inputString = preader.readLine();
					//System.out.println(inputString);

					//Set the total stat string.
					//Now extract the other info from this string.******************ADD

					calls = getCall(inputString);
				        subrs = getSubrs(inputString);
				        inclpcall = getInclPCall(inputString);
				        tmpGlobalMappingElement.setMeanCall(calls);
				        tmpGlobalMappingElement.setMeanSubrs(subrs);
				        tmpGlobalMappingElement.setMeanInclPCall(inclpcall);	
												
				}
			    }
			    else{
			    
				if(checkForExclusive(inputString))
		       		{
				   
				    //Grab the function ID.
				    functionID = getFunctionID(inputString);
				    functionName = getFunctionName(inputString);
				   				   
				    if (isAGroupMember(inputString)){
					functionGroup = getFunctionGroup(inputString);
				    }
				    else functionGroup = null;

				    //Grab the value.
				    value = getValue(inputString);
				    percentValue = getPercentValue(inputString);
									
				    //Update the max values if required.
				    //Grab the correct global mapping element.
				    //tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(functionID);
									
				
				    //Print out the node,context,thread.
				    node = getNode(inputString, false);
				    context = getContext(inputString, false);
				    thread = getThread(inputString, false);

				    //System.out.println("node/context/thread"+node+"/"+context+"/"+thread);

				    if (maxNode < node) maxNode = node;
				    if (maxContext < context) maxContext = context;
				    if (maxThread < thread) maxThread = thread;

				    //Now the complicated part.  Setting up the node,context,thread data.
									
									
				    //These first two if statements force a change if the current node or
				    //current context changes from the last, but without a corresponding change
				    //in the thread number.  For example, if we have the sequence:
				    //0,0,0 - 1,0,0 - 2,0,0 or 0,0,0 - 0,1,0 - 1,0,0.
				    if(lastNode != node)
				    {
					lastContext = -1;
					lastThread = -1;
				    }
									
				    if(lastContext != context)
				    {
					lastThread = -1;
				    }

				    if(lastThread != thread)
				    {
									
					if(thread == 0)
					{
					    //Create a new thread ... and set it to be the current thread.
					    currentGlobalThread = new GlobalThread(0);

					    //Add the correct number of global thread data elements.
					    for(i=0;i<numOfFunctions;i++)
					    {
						GlobalThreadDataElement tmpRef = null;
												
						//Add it to the currentGlobalThreadObject.
						currentGlobalThread.addThreadDataElement(tmpRef);
					    }
											
					    //Update the thread number.
					    lastThread = thread;
					
					   
					    //Set the appropriate global thread data element.
					    Vector tmpVector = currentGlobalThread.getThreadDataList();
					    GlobalThreadDataElement tmpGTDE = null;										   
					    
					    tmpGTDE = (GlobalThreadDataElement) tmpVector.elementAt(functionID);


					    if(tmpGTDE == null)
					    {
						tmpGTDE = new GlobalThreadDataElement();
						tmpGTDE.setFunctionID(functionID);
						tmpGTDE.setFunctionName(functionName);
						tmpGTDE.setFunctionGroup(functionGroup);
						currentGlobalThread.addThreadDataElement(tmpGTDE, functionID);
					    }
					    tmpGTDE.setFunctionExists();
					    tmpGTDE.setExclValue(value);
					    tmpGTDE.setExclPercValue(percentValue);  

					    
					    //Check to see if the context is zero.
					    if(context == 0)
					    {
					
						//Create a new context ... and set it to be the current context.
						currentGlobalContext = new GlobalContext(0);

						//Add the current thread
						currentGlobalContext.addThread(currentGlobalThread);
												
						//Create a new node ... and set it to be the current node.
						currentGlobalNode = new GlobalNode(node);

						//Add the current context.
						currentGlobalNode.addContext(currentGlobalContext);

						//Add the current server.
						NodeList.addElement(currentGlobalNode);
						
						//Update last context and last node.
						lastContext = context;
						lastNode = node;
					    }
					    else
					    {
						//Context number is not zero.  Create a new context ... and set it to be current.
						currentGlobalContext = new GlobalContext(context);
						//Add the current thread
						currentGlobalContext.addThread(currentGlobalThread);
						
						//Add the current context.
						currentGlobalNode.addContext(currentGlobalContext);
												
						//Update last context and last node.
						lastContext = context;
					    }
					}
					else{

					    //Thread number is not zero.  Create a new thread ... and set it to be the current thread.
					    currentGlobalThread = new GlobalThread(thread);

					    //Add the correct number of global thread data elements.
					    for(i=0;i<numOfFunctions;i++)
					    {
						GlobalThreadDataElement tmpRef = null;
						
						//Add it to the currentGlobalThreadObject.
						currentGlobalThread.addThreadDataElement(tmpRef);
					    }
											
					    //Update the thread number.
					    lastThread = thread;
					    
					    //Not thread changes.  Just set the appropriate global thread data element.
					    Vector tmpVector = currentGlobalThread.getThreadDataList();
					    GlobalThreadDataElement tmpGTDE = null;
					    tmpGTDE = (GlobalThreadDataElement) tmpVector.elementAt(functionID);
																	
					    if(tmpGTDE == null)
					    {
						tmpGTDE = new GlobalThreadDataElement();
						tmpGTDE.setFunctionID(functionID);
						tmpGTDE.setFunctionName(functionName);
						tmpGTDE.setFunctionGroup(functionGroup);
						currentGlobalThread.addThreadDataElement(tmpGTDE, functionID);
					    }

					    tmpGTDE.setFunctionExists();
					    tmpGTDE.setExclValue(value);
					    tmpGTDE.setExclPercValue(percentValue);
						
					    //Add the current thread
					    currentGlobalContext.addThread(currentGlobalThread);
					}
				    }
				    else{
					
					//Not thread changes.  Just set the appropriate global thread data element.
					Vector tmpVector = currentGlobalThread.getThreadDataList();
					GlobalThreadDataElement tmpGTDE = null;
					tmpGTDE = (GlobalThreadDataElement) tmpVector.elementAt(functionID);
									
											
					if(tmpGTDE == null)
					{
					    tmpGTDE = new GlobalThreadDataElement();
					    tmpGTDE.setFunctionID(functionID);
					    tmpGTDE.setFunctionName(functionName);
					    tmpGTDE.setFunctionGroup(functionGroup);
					    currentGlobalThread.addThreadDataElement(tmpGTDE, functionID);
					}
										
					tmpGTDE.setFunctionExists();
					tmpGTDE.setExclValue(value);
					tmpGTDE.setExclPercValue(percentValue);
				
				    }
				}  
				else if(checkForInclusive(inputString))
				{
					 //Grab the function ID.
					 functionID = getFunctionID(inputString);
					 functionName = getFunctionName(inputString);
					 if (isAGroupMember(inputString))
					     functionGroup = getFunctionGroup(inputString);
					 else 
					     functionGroup = null;

					 //Grab the value.
					 value = getValue(inputString);
					 percentValue = getPercentValue(inputString);
									
					 //Update the max values if required.
					 //Grab the correct global mapping element.
					 //tmpGlobalMappingElement = globalMapping.getGlobalMappingElement(functionID);

					 //Print out the node,context,thread.
					 node = getNode(inputString, false);
					 context = getContext(inputString, false);
					 thread = getThread(inputString, false);

					 if (maxNode<node) maxNode = node;
					 if (maxContext<context) maxContext = context;
					 if (maxThread<thread) maxThread = thread; 
									
					 //Find the correct global thread data element.
					 GlobalNode tmpGS = (GlobalNode) NodeList.elementAt(node);
					 Vector tmpGlobalContextList = tmpGS.getContextList();
					 GlobalContext tmpGC = (GlobalContext) tmpGlobalContextList.elementAt(context);
					 Vector tmpGlobalThreadList = tmpGC.getThreadList();
					 GlobalThread tmpGT = (GlobalThread) tmpGlobalThreadList.elementAt(thread);
					 Vector tmpGlobalThreadDataElementList = tmpGT.getThreadDataList();
									
					 GlobalThreadDataElement tmpGTDE = (GlobalThreadDataElement) tmpGlobalThreadDataElementList.elementAt(functionID);
					 //Now set the inclusive value!
									
					 if(tmpGTDE == null)
					 {
					     tmpGTDE = new GlobalThreadDataElement();
					     tmpGTDE.setFunctionID(functionID);
					     tmpGTDE.setFunctionName(functionName);
					     tmpGTDE.setFunctionGroup(functionGroup);
					     currentGlobalThread.addThreadDataElement(tmpGTDE, functionID);
					 }

					 tmpGTDE.setInclValue(value);
					 tmpGTDE.setInclPercValue(percentValue);
					 
									
					 //Set the total stat string.
					 //The next string in the file should be the correct one.  Assume it.
					 //If the file format changes, we are all screwed anyway.
					 inputString = preader.readLine();
					 //System.out.println(inputString);

					 //Set the total stat string.
					 //Now extract the other info from this string.
					 calls = getCall(inputString);
					 subrs = getSubrs(inputString);
					 inclpcall = getInclPCall(inputString);
					 tmpGTDE.setCall(calls);
					 tmpGTDE.setSubrs(subrs);
					 tmpGTDE.setInclPCall(inclpcall);
					 
				} // end of checkForIncl 	 					 
				else if(checkForUserEvents(inputString))
				    {
					//Get the number of user events.
					int numberOfUserEvents = getNumberOfUserEvents(inputString);
					ueAmount = numberOfUserEvents;
					System.out.println("The number of user events defined is: " + numberOfUserEvents);
														
					//The first line will be the user event heading ... get it.
					inputString = preader.readLine();
					userEventHeading = inputString;
									
					//Find the correct global thread data element.
					GlobalNode tmpGSUE = null;
					Vector tmpGlobalContextListUE = null;;
					GlobalContext tmpGCUE = null;;
					Vector tmpGlobalThreadListUE = null;;
					GlobalThread tmpGTUE = null;;
					Vector tmpGlobalThreadDataElementListUE = null;
									
					//Now that we know how many user events to expect, we can grab that number of lines.
					for(int j=0; j<numberOfUserEvents; j++)
					    {
						inputString = preader.readLine();
										
						//Initialize the user list for this thread.
						if(j == 0)
						    {
							//Note that this works correctly because we process the user events in a different manner.
							//ALL the user events for each THREAD NODE are processed in the above for-loop.  Therefore,
							//the below for-loop is only run once on each THREAD NODE.  If you do not believe it, uncomment
							//the output line below.
							//System.out.println("Creating the list for node,context,thread: " +node+","+context+","+thread);
											
							//Get the node,context,thread.
							node = getNode(inputString, true);
							context = getContext(inputString, true);
							thread = getThread(inputString, true);
											
							//Find the correct global thread data element.
							tmpGSUE = (GlobalNode) NodeList.elementAt(node);
							tmpGlobalContextListUE = tmpGSUE.getContextList();
							tmpGCUE = (GlobalContext) tmpGlobalContextListUE.elementAt(context);
							tmpGlobalThreadListUE = tmpGCUE.getThreadList();
							tmpGTUE = (GlobalThread) tmpGlobalThreadListUE.elementAt(thread);
											
							for(int k=0; k<numberOfUserEvents; k++)
							    {
								tmpGTUE.addUserThreadDataElement(new GlobalThreadDataElement());
							    }
											
							tmpGlobalThreadDataElementListUE = tmpGTUE.getUserThreadDataList();
						    }
										
										
						//Extract all the information out of the string that I need.
						//Grab the function name.
						userEventNameString = getUserEventName(inputString);

						//System.out.println("The user event name is: " + userEventNameString);

						//Grab the function ID.
						userEventID = getUserEventID(inputString);
						//System.out.println("The user event ID: " + userEventID);
										
						GlobalThreadDataElement tmpGTDEUE = (GlobalThreadDataElement) tmpGlobalThreadDataElementListUE.elementAt(userEventID);
						//Ok, now set the instance data elements.
						tmpGTDEUE.setUserEventID(userEventID);
						tmpGTDEUE.setUserEventName(userEventNameString);
						tmpGTDEUE.setUserEventNumberValue(getUENValue(inputString));
						tmpGTDEUE.setUserEventMinValue(getUEMinValue(inputString));
						tmpGTDEUE.setUserEventMaxValue(getUEMaxValue(inputString));
						tmpGTDEUE.setUserEventMeanValue(getUEMeanValue(inputString));
																
						//Ok, now get the next string as that is the stat string for this event.
						inputString = preader.readLine();
																
					    }
				    }// end of checkForUserEvent    
			    }// enf of if(checkForBeginningT(inputString)).

			}//end of if i!=2.

		    }// end of if i!=1.
		    else {
			//this is the second line, get the number of the functions.
			tokenString = genericTokenizer.nextToken();

			//Set the number of functions.
			numOfFunctions = Integer.parseInt(tokenString);

			//Now initialize the global mapping with the correct number of functions.
			initializeGlobalMapping(numOfFunctions);	

			//get the metric of this trial
			tokenString = genericTokenizer.nextToken();  
			metricStr = getMetric(tokenString);

			if (metricStr.length()==0) metricStr = "time";
		    }
		   
		}//end of if i>0.
		
		i++;
	    }//end of while	    

	}//end of try
	catch(Exception e){
		e.printStackTrace();
	}
	
    }// end of the method.


    public void writeXmlFiles(String probsize, String appid, String expid){

	GlobalNode nodeObject;
	Vector ContextList;

	GlobalContext contextObject;
	Vector ThreadList;

	GlobalThread threadObject;
	Vector ThreadDataList;
	Vector UserThreadDataList;

	GlobalThreadDataElement funObject, ueObject;

	int currentnode;
	int currentcontext;
	int currentthread;

	Vector globalMappingElementList;
	GlobalMappingElement globalmappingObject;

	try{
	    String sys = "";
	    String config = "";
	    String instru = "";
	    String compiler = "";
	    String appname = "";
	    String version = "";
	    String hostname = InetAddress.getLocalHost().getHostName();
	    File ff;

	    BufferedReader ireader = new BufferedReader(new InputStreamReader(System.in));

		if (probsize == null) {
	    	System.out.println("Please tell me the problem size");
	    	probsize = ireader.readLine().trim();
	    	if (probsize.length()==0) probsize = "-1";
		}

		if (appid == null) {
	    	System.out.println("Please tell me the ID of the application this trial belongs to.");
	    	appid  = ireader.readLine().trim();
		}

	    if (appid.length()>0) 
			appid =  checkForApp(appid);
	    else {
			System.out.println("If you don't know the exact application ID, please tell me the application group the trial belongs to:");
			appname = ireader.readLine().trim();
			System.out.println("Please tell me the application version");
			version = ireader.readLine().trim();
			appid = checkForApp(appname, version);
	    }
	
	    if (appid==null){
			System.out.println("Cannot find such an application. Quit translanting.");
			System.exit(-1);
	    }
		
		if (expid == null){
	    	System.out.println("Please tell me the ID of the experiment group this trial belongs to.");
	    	expid  = ireader.readLine().trim();
		}
	    
	    if (expid.length()>0) 
			expid =  checkForEnv(expid);
	    else {
			System.out.println("If you don't know the exact experiment ID number, please tell me the location of environment files,");
			// System.out.println("in the form of machinename:absolutepath, ");

			System.out.println("if they are different from ./Sys_Info.xml, ./Config_Info.xml, ./Instru_Info.xml and ./Compiler_Info.xml respectively.");

			System.out.println("System information:");
			sys = ireader.readLine().trim();
			if (sys.trim().length()==0)
		    	ff = new File("Sys_Info.xml");
			else ff = new File(sys);
			sys = hostname + ":" + ff.getAbsolutePath();

			System.out.println("Configuration information:");
			config = ireader.readLine().trim();
			if (config.trim().length()==0)
		    	ff = new File("Config_Info.xml");
			else ff = new File(config);	
			config = hostname + ":" + ff.getAbsolutePath();
	    
			System.out.println("Instrumentation information:");    
			instru = ireader.readLine().trim();
			if (instru.trim().length()==0)
		    	ff = new File("Instru_Info.xml");
			else ff = new File(instru); 	
			instru = hostname + ":" + ff.getAbsolutePath();

			System.out.println("Compiler information:");
			compiler = ireader.readLine().trim();
			if (compiler.trim().length()==0)
		    	ff = new File("Compiler_Info.xml");
			else ff = new File(compiler);	
			compiler = hostname + ":" + ff.getAbsolutePath();

			expid = checkForEnv(sys, config, compiler, instru, appid);
	    }

		if (expid == null) {
			System.out.println("Cannot find such an experiment that is in accordance with the information you provided above. Quit translanting.");
			System.exit(-1);
		}
	else{
	    //System.out.println("the experiment id is "+expid);

	    // start to write xml file here.	    
	    BufferedWriter xwriter = new BufferedWriter(new FileWriter(writeXml));

	    xwriter.write("<?xml version=\"1.0\"?>", 0, ("<?xml version=\"1.0\"?>").length());
	    xwriter.newLine();
	    xwriter.write("<Trials>", 0, ("<Trials>").length());
	    xwriter.newLine();
	    	    
	    xwriter.write("    <Onetrial Metric='" + metricStr + "'>", 0, ("    <Onetrial Metric='" + metricStr + "'>").length());
	    xwriter.newLine();

	    writeComputationModel(xwriter, maxNode+1, maxContext+1, maxThread+1);
	    
	    xwriter.write("\t<Env>", 0, ("\t<Env>").length());
	    xwriter.newLine();
	  	   	   
	    xwriter.write("\t   <AppID>" + appid + "</AppID>", 0, ("\t   <AppID>" + appid + "</AppID>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t   <ExpID>" + expid + "</ExpID>", 0, ("\t   <ExpID>" + expid + "</ExpID>").length());
	    xwriter.newLine();

	    xwriter.write("\t</Env>", 0, ("\t</Env>").length());
	    xwriter.newLine();

	    xwriter.write("\t<Trialtime>" + trialTime + "</Trialtime>", 0, ("\t<Trialtime>" + trialTime + "</Trialtime>").length());
            xwriter.newLine();

	    xwriter.write("\t<FunAmt>" + funAmount + "</FunAmt>", 0, ("\t<FunAmt>" + funAmount + "</FunAmt>").length());
            xwriter.newLine();

	    xwriter.write("\t<UserEventAmt>" + ueAmount + "</UserEventAmt>", 0, ("\t<UserEventAmt>" + ueAmount + "</UserEventAmt>").length());
            xwriter.newLine();

	    xwriter.write("\t<ProblemSize>" + probsize + "</ProblemSize>", 0, ("\t<ProblemSize>" + probsize + "</ProblemSize>").length());
	    xwriter.newLine();

	    xwriter.write("\t<Pprof>", 0, ("\t<Pprof>").length());
	    xwriter.newLine();

	    for(Enumeration en = NodeList.elements(); en.hasMoreElements() ;)
	    {
		nodeObject = (GlobalNode) en.nextElement();
		currentnode = nodeObject.getNodeName();
	
		ContextList = nodeObject.getContextList();
		for(Enumeration ec = ContextList.elements(); ec.hasMoreElements() ;)
	        {
		    contextObject = (GlobalContext) ec.nextElement();
		    currentcontext = contextObject.getContextName();

		    ThreadList = contextObject.getThreadList();
		    for(Enumeration et = ThreadList.elements(); et.hasMoreElements() ;)
		    {
			threadObject = (GlobalThread) et.nextElement();
			currentthread = threadObject.getThreadName();
			
			ThreadDataList = threadObject.getThreadDataList();
			UserThreadDataList = threadObject.getUserThreadDataList();

			writeIDs(xwriter,currentnode,currentcontext, currentthread);

			for(Enumeration ef = ThreadDataList.elements(); ef.hasMoreElements() ;)
			{
			    funObject = (GlobalThreadDataElement) ef.nextElement();
			    if (funObject!=null){
								
				writeFunName(xwriter,funObject.getFunctionName());
  				
				writeFunID(xwriter, funObject.getFunctionID());
				
				writeFunGroup(xwriter, funObject.getFunctionGroup());
				
				writeInclPerc(xwriter,funObject.getInclPercValue());
				
				writeIncl(xwriter,funObject.getInclValue());
				
				writeExclPerc(xwriter,funObject.getExclPercValue());
				
				writeExcl(xwriter, funObject.getExclValue());
				
				writeCall(xwriter,funObject.getCall());
				
				writeSubrs(xwriter,funObject.getSubrs());
				
				writeInclPCall(xwriter,funObject.getInclPCall());	
			
			    }
			}

			for(Enumeration uef = UserThreadDataList.elements(); uef.hasMoreElements() ;)
			{
			    ueObject = (GlobalThreadDataElement) uef.nextElement();
			    if (ueObject!=null){
				
				writeUEName(xwriter,ueObject.getUserEventName());
				
				writeUEID(xwriter, ueObject.getUserEventID());
				
				writeNumofSamples(xwriter,ueObject.getUserEventNumberValue());
				
				writeMaxValue(xwriter,ueObject.getUserEventMaxValue());
				
				writeMinValue(xwriter,ueObject.getUserEventMinValue());
 
				writeMeanValue(xwriter, ueObject.getUserEventMeanValue());
			    }
			}			
		    }
		}    
	    }

	    xwriter.write("\t</Pprof>", 0, ("\t</Pprof>").length());
	    xwriter.newLine();
	    xwriter.newLine();

	    
	    xwriter.write("\t<totalfunsummary>", 0, ("\t<totalfunsummary>").length());
	    xwriter.newLine();
	    
	    // process total information
	    globalMappingElementList = globalMapping.getNameIDMapping();

	    for(Enumeration et = globalMappingElementList.elements(); et.hasMoreElements() ;){
		
       		globalmappingObject = (GlobalMappingElement) et.nextElement();
		if (globalmappingObject.getFunctionName() != "Name has not been set!"){

		    
		    writeTotalFunName(xwriter, globalmappingObject.getFunctionName());
  
		    
		    writeTotalFunID(xwriter, globalmappingObject.getGlobalID());

		    writeTotalFunGroup(xwriter, globalmappingObject.getFunctionGroup());
		   
		    writeTotalInclPerc(xwriter, globalmappingObject.getTotalInclusivePercentValue());

		    
		    writeTotalIncl(xwriter, globalmappingObject.getTotalInclusiveValue());
		   
		    
		    writeTotalExclPerc(xwriter, globalmappingObject.getTotalExclusivePercentValue());
 
		    
		    writeTotalExcl(xwriter, globalmappingObject.getTotalExclusiveValue());
		    
		    
		    writeTotalCall(xwriter, globalmappingObject.getTotalCall());

		    
		    writeTotalSubrs(xwriter, globalmappingObject.getTotalSubrs());
 
		    
		    writeTotalInclPCall(xwriter, globalmappingObject.getTotalInclPCall());
		    		    
	        }
	    }

	    xwriter.write("\t</totalfunsummary>", 0, ("\t</totalfunsummary>").length());
	    xwriter.newLine();
	    xwriter.newLine();

	    xwriter.write("\t<meanfunsummary>", 0, ("\t<meanfunsummary>").length());
	    xwriter.newLine();

	    for(Enumeration em = globalMappingElementList.elements(); em.hasMoreElements() ;){
		
       		globalmappingObject = (GlobalMappingElement) em.nextElement();
		if (globalmappingObject.getFunctionName() != "Name has not been set!"){
		    
		    // mean stuff 
		    writeMeanFunName(xwriter, globalmappingObject.getFunctionName());

		    writeMeanFunID(xwriter, globalmappingObject.getGlobalID());

		    writeMeanFunGroup(xwriter, globalmappingObject.getFunctionGroup());

		    writeMeanInclPerc(xwriter,globalmappingObject.getMeanInclusivePercentValue());
 
		   
		    writeMeanIncl(xwriter,globalmappingObject.getMeanInclusiveValue());
 		    
		    
		    writeMeanExclPerc(xwriter,globalmappingObject.getMeanExclusivePercentValue());
 
		    
		    writeMeanExcl(xwriter,globalmappingObject.getMeanExclusiveValue());
 		    
		    
		    writeMeanCall(xwriter,globalmappingObject.getMeanCall());
 
		    
		    writeMeanSubrs(xwriter,globalmappingObject.getMeanSubrs());
 
		    
		    writeMeanInclPCall(xwriter, globalmappingObject.getMeanInclPCall());			    		    
	        }
	    }

	    xwriter.write("\t</meanfunsummary>", 0, ("\t</meanfunsummary>").length());
	    xwriter.newLine();	   
	    
	    xwriter.write("    </Onetrial>", 0, ("    </Onetrial>").length());
	    xwriter.newLine();

	    xwriter.write("</Trials>", 0, ("</Trials>").length());
	    xwriter.newLine();
	    xwriter.close();

	    connector.dbclose();
	}    	    		
	}catch(Exception e){	    
		e.printStackTrace();
	}
    }

	//******************************
	//Helper functions for buildStatic data.
	//******************************
    boolean checkForBeginningT(String inString)
    {
       	try{
      		StringTokenizer checkForBeginningTTokenizer = new StringTokenizer(inString, " \t\n\r");
			
       		String tmpString;
       		
       		tmpString = checkForBeginningTTokenizer.nextToken();
       			
       		if(tmpString.equals("t"))
       			return true;
       		else
       			return false;
       	}catch(Exception e)
       	{
       		e.printStackTrace();	
       	}
		
       	return false;
    }
	
    boolean checkForBeginningM(String inString)
    {
		
       	try{
       		StringTokenizer checkForBeginningTTokenizer = new StringTokenizer(inString, " \t\n\r");
       		
       		String tmpString;
			
       		tmpString = checkForBeginningTTokenizer.nextToken();
				
       		if(tmpString.equals("m"))
       			return true;
       		else
       			return false;
       	}catch(Exception e){
       		e.printStackTrace();	
       	}
		
       	return false;
    }
	
    boolean checkForExclusive(String inString)
    {
		
       	try{
			//In this function I need to be careful.  If the function name contains "excl", I
			//might interpret this line as being the exclusive line when in fact it is not.
			
			//Check for the right string.
		//System.out.println(inString);
       		StringTokenizer checkTokenizer = new StringTokenizer(inString," ");
       		String tmpString2 = checkTokenizer.nextToken();
       		if((tmpString2.indexOf(",")) != -1)
       		{
				//Ok, so at least we have the correct string.
				//Now, we want to grab the substring that occurs AFTER the SECOND '"'.
				//At present, pprof does not seem to allow an '"' in the function name.  So
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
		       	if(tmpString.equals("excl"))
		       	{
		       		return true;
		       	}
		 }
			
			//If here, it means that we are not looking at the correct string or that we did not
			//find a match.  Therefore, return false.
	       	 return false;
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
			
       	}
		
       	return false;
    }
	
    boolean checkForExclusiveWithTOrM(String inString)
    {
		
       	try{
			//In this function I need to be careful.  If the function name contains "excl", I
			//might interpret this line as being the exclusive line when in fact it is not.
			
			//Ok, so at least we have the correct string.
			//Now, we want to grab the substring that occurs AFTER the SECOND '"'.
			//At present, pprof does not seem to allow an '"' in the function name.  So
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
       		if(tmpString.equals("excl"))
       		{
       			return true;
       		}
			
			//If here, it means that we are not looking at the correct string or that we did not
			//find a match.  Therefore, return false.
       		return false;
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
			
       	}
		
       	return false;
    }
	
    boolean checkForInclusive(String inString)
    {
		
       	try{
			//In this function I need to be careful.  If the function name contains "incl", I
			//might interpret this line as being the inclusive line when in fact it is not.
			
			
			//Check for the right string.
       		StringTokenizer checkTokenizer = new StringTokenizer(inString," ");
       		String tmpString2 = checkTokenizer.nextToken();
       		if((tmpString2.indexOf(",")) != -1)
       		{
			
				//Now, we want to grab the substring that occurs AFTER the SECOND '"'.
				//At present, pprof does not seem to allow an '"' in the function name.  So
				//, I can be assured that I will not find more than two before the "excl" or "incl".
       			StringTokenizer checkQuotesTokenizer = new StringTokenizer(inString,"\"");
				
				//Need to get the third token.  Could do it in a loop, just as quick this way.
       			String tmpString = checkQuotesTokenizer.nextToken();
       			tmpString = checkQuotesTokenizer.nextToken();
       			tmpString = checkQuotesTokenizer.nextToken();
				
				//Ok, now, the string in tmpString should include at least "excl" or "incl", and
				//also, the first token should be either "excl" or "incl".
       			StringTokenizer checkForInclusiveTokenizer = new StringTokenizer(tmpString, " \t\n\r");
       			tmpString = checkForInclusiveTokenizer.nextToken();
					
				//At last, do the check.	
       			if(tmpString.equals("incl"))
       			{
       				return true;
       			}
       		}
			//If here, it means that we are not looking at the correct string or that we did not
			//find a match.  Therefore, return false.
       		return false;
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
       	
       	}
		
       	return false;
    }
	
    boolean checkForInclusiveWithTOrM(String inString)
    {
		
       	try{
			//In this function I need to be careful.  If the function name contains "incl", I
			//might interpret this line as being the inclusive line when in fact it is not.

			//Ok, so at least we have the correct string.
			//Now, we want to grab the substring that occurs AFTER the SECOND '"'.
			//At present, pprof does not seem to allow an '"' in the function name.  So
			//, I can be assured that I will not find more than two before the "excl" or "incl".
       		StringTokenizer checkQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Need to get the third token.  Could do it in a loop, just as quick this way.
       		String tmpString = checkQuotesTokenizer.nextToken();
       		tmpString = checkQuotesTokenizer.nextToken();
       		tmpString = checkQuotesTokenizer.nextToken();
			
			//Ok, now, the string in tmpString should include at least "excl" or "incl", and
			//also, the first token should be either "excl" or "incl".
       		StringTokenizer checkForInclusiveTokenizer = new StringTokenizer(tmpString, " \t\n\r");
       		tmpString = checkForInclusiveTokenizer.nextToken();
				
			//At last, do the check.	
       		if(tmpString.equals("incl"))
       		{
       			return true;
       		}
			
			//If here, it means that we are not looking at the correct string or that we did not
			//find a match.  Therefore, return false.
       		return false;
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
			
       	}
       	
       	return false;
    }
	
    String getFunctionName(String inString)
    {
       	try{
       		String tmpString;
			
       		
       		StringTokenizer getFunctionNameTokenizer = new StringTokenizer(inString, "\"");
			
			//Since we know that the function name is the only one in the quotes, just ignore the
			//first token, and then grab the next.
			
			//Grab the first token.
       		tmpString = getFunctionNameTokenizer.nextToken();
			
			//Grab the second token.
       		tmpString = getFunctionNameTokenizer.nextToken();
			
			//Now return the second string.
       		return tmpString.trim();
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
		
       	return null;
    }
	
    int getFunctionID(String inString)
    {
       	try{
       		String tmpString;
       		
       		StringTokenizer getFunctionIDTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			//The function id will be the second token on its line.
			
			//Grab the first token.
       		tmpString = getFunctionIDTokenizer.nextToken();
			
			//Grab the second token.
       		tmpString = getFunctionIDTokenizer.nextToken();
			
			
			//Now return the id.
			//Integer tmpInteger = new Integer(tmpString);
			//int tmpInt = tmpInteger.intValue();
       		return Integer.parseInt(tmpString);
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
			
       	}
		
       	return -1;
    }

    boolean isAGroupMember(String inString){
	try{
	    String tmpString;
		       		
	    StringTokenizer getFunctionGroupTokenizer = new StringTokenizer(inString, "\"");
						
	    if (getFunctionGroupTokenizer.countTokens()>3){
		//Grab the first token.
       		tmpString = getFunctionGroupTokenizer.nextToken();
			
		//Grab the second token.
       		tmpString = getFunctionGroupTokenizer.nextToken();
			
		//Grab the third token.
       		tmpString = getFunctionGroupTokenizer.nextToken();

		return tmpString.trim().endsWith("GROUP=");
	    }
       	}
       	catch(Exception e)
	    {
       		e.printStackTrace();	
	    }
		
       	return false;
    }


    String getFunctionGroup(String inString)
    {
       	try{
       		String tmpString;
		       		
       		StringTokenizer getFunctionGroupTokenizer = new StringTokenizer(inString, "\"");
						
		//Grab the first token.
       		tmpString = getFunctionGroupTokenizer.nextToken();
			
		//Grab the second token.
       		tmpString = getFunctionGroupTokenizer.nextToken();
			
		//Grab the third token.
       		tmpString = getFunctionGroupTokenizer.nextToken();

		//Grab the fourth token.
       		tmpString = getFunctionGroupTokenizer.nextToken();

		//Now return the fourth string.
       		return tmpString.trim();
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
	}
		
       	return null;
    }
	
    double getValue(String inString)
    {
       	try{
       		String tmpString;
			
			//First strip away the portion of the string not needed.
       		StringTokenizer valueQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
       		tmpString = valueQuotesTokenizer.nextToken();
       		tmpString = valueQuotesTokenizer.nextToken();
       		tmpString = valueQuotesTokenizer.nextToken();
		
			//Ok, now concentrate on the third token.  The token in question should be the second.
       		StringTokenizer valueTokenizer = new StringTokenizer(tmpString, " \t\n\r");
       		tmpString = valueTokenizer.nextToken();
       		tmpString = valueTokenizer.nextToken();
			//Now return the value obtained as an int.
       		return Double.parseDouble(tmpString);
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
		
       	return -1;
    }
	
    double getPercentValue(String inString)
    {
       	try{
       		String tmpString;
			
			//First strip away the portion of the string not needed.
       		StringTokenizer percentValueQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
       		tmpString = percentValueQuotesTokenizer.nextToken();
       		tmpString = percentValueQuotesTokenizer.nextToken();
       		tmpString = percentValueQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the third.
       		StringTokenizer percentValueTokenizer = new StringTokenizer(tmpString, " \t\n\r");
       		tmpString = percentValueTokenizer.nextToken();
       		tmpString = percentValueTokenizer.nextToken();
       		tmpString = percentValueTokenizer.nextToken();
			
			//Now return the value obtained.
       		return Double.parseDouble(tmpString);
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
		
       	return -1;
    }

    double getCall(String inString){
	try{
		String tmpString;
		StringTokenizer callValueTokenizer = new StringTokenizer(inString, " \t\n\r");
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		return Double.parseDouble(tmpString);
	}
	catch (Exception e){
		e.printStackTrace();
	}
	return -1;
    }

    double getSubrs(String inString){
	try{
		String tmpString;
		StringTokenizer callValueTokenizer = new StringTokenizer(inString, " \t\n\r");
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		return Double.parseDouble(tmpString);
	}
	catch (Exception e){
		e.printStackTrace();
	}
	return -1;
    }

    double getInclPCall(String inString){
	try{
		String tmpString;
		StringTokenizer callValueTokenizer = new StringTokenizer(inString, " \t\n\r");
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		tmpString = callValueTokenizer.nextToken();
		return Double.parseDouble(tmpString);
	}
	catch (Exception e){
		e.printStackTrace();
	}
	return -1;
    }
	
    boolean checkForUserEvents(String inString)
    {
       	try{
       		String tmpString;
			
       		StringTokenizer checkForUserEventsTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			//Looking for the second token ... no danger of conflict here.
			
			//Grab the first token.
       		tmpString = checkForUserEventsTokenizer.nextToken();
			
			//Grab the second token.
       		tmpString = checkForUserEventsTokenizer.nextToken();
			
			//No do the check.
       		if(tmpString.equals("userevents"))
       			return true;
       		else
       			return false;
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
       		
       	}
		
       	return false;	
    }
	
    int getNumberOfUserEvents(String inString)
    {
       	try{
       		StringTokenizer getNumberOfUserEventsTokenizer = new StringTokenizer(inString, " \t\n\r");
       		
       		String tmpString;
			
			//It will be the first token.
       		tmpString = getNumberOfUserEventsTokenizer.nextToken();
			
			//Now return the number of user events number.
       		return Integer.parseInt(tmpString);
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
		
       	return -1;
    }
										
    String getUserEventName(String inString)
    {
       	try{
       		String tmpString;
			
       		StringTokenizer getUserEventNameTokenizer = new StringTokenizer(inString, "\"");
			
			//Since we know that the user event name is the only one in the quotes, just ignore the
			//first token, and then grab the next.
			
			//Grab the first token.
       		tmpString = getUserEventNameTokenizer.nextToken();
			
			//Grab the second token.
       		tmpString = getUserEventNameTokenizer.nextToken();
			
			//Now return the second string.
		
       		return tmpString;
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
       		
       	}
		
       	return null;
    }
	
    int getUserEventID(String inString)
    {
       	try{
       		String tmpString;
			
       		StringTokenizer getUserEventIDTokenizer = new StringTokenizer(inString, " \t\n\r");
			
			//The function id will be the third token on its line.
			
			//Grab the first token.
       		tmpString = getUserEventIDTokenizer.nextToken();
			
			//Grab the second token.
       		tmpString = getUserEventIDTokenizer.nextToken();
			
			//Grab the second token.
       		tmpString = getUserEventIDTokenizer.nextToken();
			
       		return Integer.parseInt(tmpString);
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
		
       	return -1;
    }
	
    int getUENValue(String inString)
    {
		
       	try{
       		String tmpString;
			
			//First strip away the portion of the string not needed.
       		StringTokenizer uEQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
       		tmpString = uEQuotesTokenizer.nextToken();
       		tmpString = uEQuotesTokenizer.nextToken();
       		tmpString = uEQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the first.
       		StringTokenizer uETokenizer = new StringTokenizer(tmpString, " \t\n\r");
       		tmpString = uETokenizer.nextToken();
			
			//Now return the value obtained as an int.
       		return (int)Double.parseDouble(tmpString);
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
       	
       	return -1;
    }
	
    double getUEMinValue(String inString)
    {
       	try{
       		String tmpString;
			
			//First strip away the portion of the string not needed.
       		StringTokenizer uEQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
       		tmpString = uEQuotesTokenizer.nextToken();
       		tmpString = uEQuotesTokenizer.nextToken();
       		tmpString = uEQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the third.
       		StringTokenizer uETokenizer = new StringTokenizer(tmpString, " \t\n\r");
	       	tmpString = uETokenizer.nextToken();
	       	tmpString = uETokenizer.nextToken();
	       	tmpString = uETokenizer.nextToken();
			
			//Now return the value obtained.
	       	return Double.parseDouble(tmpString);
        }
       	catch(Exception e)
       	{
		
       	}
		
       	return -1;
    }
	
    double getUEMaxValue(String inString)
    {
       	try{
       		String tmpString;
			
			//First strip away the portion of the string not needed.
       		StringTokenizer uEQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
       		tmpString = uEQuotesTokenizer.nextToken();
       		tmpString = uEQuotesTokenizer.nextToken();
       		tmpString = uEQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the second.
                StringTokenizer uETokenizer = new StringTokenizer(tmpString, " \t\n\r");
               	tmpString = uETokenizer.nextToken();
       		tmpString = uETokenizer.nextToken();
			
			//Now return the value obtained.
       		return Double.parseDouble(tmpString);
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
		
       	return -1;
    }
	
    double getUEMeanValue(String inString)
    {
       	try{
       		String tmpString;
			
			//First strip away the portion of the string not needed.
       		StringTokenizer uEQuotesTokenizer = new StringTokenizer(inString,"\"");
			
			//Grab the third token.
       		tmpString = uEQuotesTokenizer.nextToken();
       		tmpString = uEQuotesTokenizer.nextToken();
       		tmpString = uEQuotesTokenizer.nextToken();
			
			//Ok, now concentrate on the third token.  The token in question should be the forth.
       		StringTokenizer uETokenizer = new StringTokenizer(tmpString, " \t\n\r");
       		tmpString = uETokenizer.nextToken();
       		tmpString = uETokenizer.nextToken();
       		tmpString = uETokenizer.nextToken();
       		tmpString = uETokenizer.nextToken();
			
			//Now return the value obtained.
       		return Double.parseDouble(tmpString);
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
       	
       	}
		
       	return -1;
    }
	
    int getNode(String inString, boolean UEvent)
    {
       	try{
       		StringTokenizer getNodeTokenizer = new StringTokenizer(inString, ", \t\n\r");
			
       		String tmpString;
			
       		if(UEvent)
       		{
				//Need to strip off the first token.
       			tmpString = getNodeTokenizer.nextToken();
       		}
			
			//Get the first token.
       		tmpString = getNodeTokenizer.nextToken();
			
			//Now return the node number.
       		Integer tmpInteger = new Integer(tmpString);
       		int tmpInt = tmpInteger.intValue();
       		return tmpInt;
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
		
       	return -1;
		
    }
	
    int getContext(String inString, boolean UEvent)
    {
       	try{
       		StringTokenizer getContextTokenizer = new StringTokenizer(inString, ", \t\n\r");
			
       		String tmpString;
			
       		if(UEvent)
       		{
				//Need to strip off the first token.
       			tmpString = getContextTokenizer.nextToken();
       		}
			
			//Get the first token.
       		tmpString = getContextTokenizer.nextToken();
			
			//Get the second.
       		tmpString = getContextTokenizer.nextToken();
			
			//Now return the context number.
       		Integer tmpInteger = new Integer(tmpString);
       		int tmpInt = tmpInteger.intValue();
       		return tmpInt;
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
       	
       	}
		
       	return -1;
    }
	
    int getThread(String inString, boolean UEvent)
    {		
       	try{
       		StringTokenizer getThreadTokenizer = new StringTokenizer(inString, ", \t\n\r");
			
       		String tmpString;
			
       		if(UEvent)
       		{
				//Need to strip off the first token.
       			tmpString = getThreadTokenizer.nextToken();
       		}
			
			//Get the first token.
       		tmpString = getThreadTokenizer.nextToken();
			
			//Get the second.
       		tmpString = getThreadTokenizer.nextToken();
			
			//Get the third token.
       		tmpString = getThreadTokenizer.nextToken();
			
			//Now return the context number.
       		Integer tmpInteger = new Integer(tmpString);
       		int tmpInt = tmpInteger.intValue();
       		return tmpInt;
       	}
       	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
		
       	return -1;
    }

    public void writeComputationModel(BufferedWriter writer, int node, int context, int thread){

	try{
	    writer.write("\t<ComputationModel>", 0, ("\t<ComputationModel>").length());
	    writer.newLine();

	    String tmpString;
	    tmpString = "\t   <node level=\"Top\" statis_info=\"sum\">"+node+"</node>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    tmpString = "\t   <context level=\"Secondary\" statis_info=\"contextPnode\">"+context+"</context>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    tmpString = "\t   <thread level=\"Lowest\" statis_info=\"threadPcontext\">"+thread+"</thread>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    writer.write("\t</ComputationModel>", 0, "\t</ComputationModel>".length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeIDs(BufferedWriter writer, int node, int context, int thread){
	
	try{
	    String tmpString;
	    tmpString = "\t   <nodeID>"+node+"</nodeID>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    tmpString = "\t   <contextID>"+context+"</contextID>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    tmpString = "\t   <threadID>"+thread+"</threadID>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}

    }

    public void writeFunName(BufferedWriter writer, String funname){
	try{

	    writer.write("\t   <instrumentedobj>", 0, ("\t   <instrumentedobj>").length());
	    writer.newLine();
	    
	    String tmpString;
	    funname = replace(funname, "&", "&amp;");
	    funname = replace(funname, "<", "&lt;");
	    funname = replace(funname, ">", "&gt;");
	    tmpString = "\t\t<funname>"+funname+"</funname>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}

    }
    
    public void writeFunID(BufferedWriter writer, int funID){
	try{

	    String tmpString;
	    tmpString = "\t\t<funID>"+funID+"</funID>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();

	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeFunGroup(BufferedWriter writer, String funGroup){
	try{

	    String tmpString;
	    tmpString = "\t\t<fungroup>"+funGroup+"</fungroup>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();

	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeInclPerc(BufferedWriter writer, double InclPercValue){
	try{
	    String tmpString;
	    tmpString = "\t\t<inclperc>"+InclPercValue+"</inclperc>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}

    }

    public void writeIncl(BufferedWriter writer, double InclValue){
	try{
	    String tmpString;
	    tmpString = "\t\t<inclutime>"+InclValue+"</inclutime>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }
    
    public void writeExclPerc(BufferedWriter writer, double ExclPercValue){
	try{
	    String tmpString;
	    tmpString = "\t\t<exclperc>"+ExclPercValue+"</exclperc>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeExcl(BufferedWriter writer, double ExclValue){
	try{
	    String tmpString;
	    tmpString = "\t\t<exclutime>"+ExclValue+"</exclutime>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }


    public void writeCall(BufferedWriter writer, double calls){
	try{
	    String tmpString;
	    tmpString = "\t\t<call>"+calls+"</call>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    
    }

    public void writeSubrs(BufferedWriter writer,double subrs){
	try{
	    String tmpString;
	    tmpString = "\t\t<subrs>"+subrs+"</subrs>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeInclPCall(BufferedWriter writer, double InclPCall){
	try{
	    String tmpString;
	    tmpString = "\t\t<inclutimePcall>"+InclPCall+"</inclutimePcall>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    writer.write("\t   </instrumentedobj>", 0, ("\t   </instrumentedobj>").length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeUEName(BufferedWriter writer, String name){
	try{

	    writer.write("\t   <userevent>", 0, ("\t   <userevent>").length());
	    writer.newLine();

	    String tmpString;	    
	    tmpString = "\t\t<uename>"+name+"</uename>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();	    
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeUEID(BufferedWriter writer, int ueID){
	try{

	    String tmpString;
	    tmpString = "\t\t<ueID>"+ueID+"</ueID>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();

	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeNumofSamples(BufferedWriter writer, int numofsamples){
	try{
	    String tmpString;
	    tmpString = "\t\t<numofsamples>"+numofsamples+"</numofsamples>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeMaxValue(BufferedWriter writer, double maxValue){
	try{
	    String tmpString;
	    tmpString = "\t\t<maxvalue>"+maxValue+"</maxvalue>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeMinValue(BufferedWriter writer, double minValue){
	try{
	    String tmpString;
	    tmpString = "\t\t<minvalue>"+minValue+"</minvalue>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeMeanValue(BufferedWriter writer, double meanValue){
	try{
	    String tmpString;
	    tmpString = "\t\t<meanvalue>"+meanValue+"</meanvalue>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    writer.write("\t   </userevent>", 0, ("\t   </userevent>").length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeTotalFunName(BufferedWriter writer, String funname){
	try{
	    writer.write("\t   <totalfunction>", 0, ("\t   <totalfunction>").length());
	    writer.newLine();

	    String tmpString;
	    funname = replace(funname, "&", "&amp;");
	    funname = replace(funname, "<", "&lt;");
	    funname = replace(funname, ">", "&gt;");
	    tmpString = "\t\t<funname>"+funname+"</funname>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeTotalFunID(BufferedWriter writer, int funID ){
	try{

	    String tmpString;
	    tmpString = "\t\t<funID>"+funID+"</funID>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
        catch(Exception e) {
       		e.printStackTrace();	
		
       	}
    }

    public void writeTotalFunGroup(BufferedWriter writer, String funGroup ){
	try{

	    String tmpString;
	    tmpString = "\t\t<fungroup>"+funGroup+"</fungroup>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
        catch(Exception e) {
       		e.printStackTrace();	
		
       	}
    }

    public void writeTotalInclPerc(BufferedWriter writer, double inclperc){
	try{
	    
	    String tmpString;
	    tmpString = "\t\t<inclperc>"+inclperc+"</inclperc>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();

	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }


    public void writeTotalIncl(BufferedWriter writer, double incl){
	try{

	    String tmpString;
	    tmpString = "\t\t<inclutime>"+incl+"</inclutime>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }


    public void writeTotalExclPerc(BufferedWriter writer, double exclperc){
	try{

	    String tmpString;
	    tmpString = "\t\t<exclperc>"+exclperc+"</exclperc>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine(); 
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }


    public void writeTotalExcl(BufferedWriter writer, double excl){

	try{

	    String tmpString;
	    tmpString = "\t\t<exclutime>"+excl+"</exclutime>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine(); 
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    
    
    public void writeTotalCall(BufferedWriter writer, double calls){
	
	try{

	    String tmpString;
	    tmpString = "\t\t<call>"+calls+"</call>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine(); 
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
	
    }


    public void writeTotalSubrs(BufferedWriter writer, double subrs){
	try{
	    String tmpString;
	    tmpString = "\t\t<subrs>"+subrs+"</subrs>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine(); 

	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
	
    }

    public void writeTotalInclPCall(BufferedWriter writer, double InclPCall){
	try{

	    String tmpString;
	    tmpString = "\t\t<inclutimePcall>"+InclPCall+"</inclutimePcall>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine(); 
	    writer.write("\t   </totalfunction>", 0, ("\t   </totalfunction>").length());
	    writer.newLine();	    

	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
	
    }

    public void writeMeanFunID(BufferedWriter writer, int funID){

	try{
	    String tmpString;
	    tmpString = "\t\t<funID>"+funID+"</funID>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeMeanFunName(BufferedWriter writer, String funname){
	try{
	    writer.write("\t   <meanfunction>", 0, ("\t   <meanfunction>").length());
	    writer.newLine();

	    String tmpString;
	    funname = replace(funname, "&", "&amp;");
	    funname = replace(funname, "<", "&lt;");
	    funname = replace(funname, ">", "&gt;");
	    tmpString = "\t\t<funname>"+funname+"</funname>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeMeanFunGroup(BufferedWriter writer, String funGroup ){
	try{

	    String tmpString;
	    tmpString = "\t\t<fungroup>"+funGroup+"</fungroup>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
        catch(Exception e) {
       		e.printStackTrace();	
		
       	}
    }

    public void writeMeanInclPerc(BufferedWriter writer, double inclperc){

	try{
	    String tmpString;
	    tmpString = "\t\t<inclperc>"+inclperc+"</inclperc>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeMeanIncl(BufferedWriter writer, double incl){
	try{
	    String tmpString;
	    tmpString = "\t\t<inclutime>"+incl+"</inclutime>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeMeanExclPerc(BufferedWriter writer, double exclperc){

	try{
	    String tmpString;
	    tmpString = "\t\t<exclperc>"+exclperc+"</exclperc>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine(); 
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}	
    }

    public void writeMeanExcl(BufferedWriter writer, double excl){

	try{
	    String tmpString;
	    tmpString = "\t\t<exclutime>"+excl+"</exclutime>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine(); 
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeMeanCall(BufferedWriter writer, double calls){

	try{
	    String tmpString;
	    tmpString = "\t\t<call>"+calls+"</call>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine(); 
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeMeanSubrs(BufferedWriter writer, double subrs){

	try{
	    String tmpString;
	    tmpString = "\t\t<subrs>"+subrs+"</subrs>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine(); 
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public void writeMeanInclPCall(BufferedWriter writer, double inclpcall){

	try{
	    String tmpString;
	    tmpString = "\t\t<inclutimePcall>"+inclpcall+"</inclutimePcall>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine(); 
	    writer.write("\t   </meanfunction>", 0, ("\t   </meanfunction>").length());
	    writer.newLine();	
	}
	catch(Exception e)
       	{
       		e.printStackTrace();	
		
       	}
    }

    public String getMetric(String str){
	if (str.length() > 26)
		return(str.substring(26, str.length()));
	else return new String("");
    }

    public String replace(String str, String lstr, String rstr){
	String tempStr = "";
        int i;
        while ( (i=str.indexOf(lstr)) != -1) {
                if (i>0)
                        tempStr += str.substring(0,i);
                tempStr += rstr;
                str = str.substring(i+1);
        }
        tempStr += str;
        return tempStr;
    } 	
 
    public String getFunAmt(String inString){
	try{
		String tmpString;
		StringTokenizer funAmtTokenizer = new StringTokenizer(inString, " \t\n\r");
		tmpString = funAmtTokenizer.nextToken();
		return tmpString;
	}
	catch (Exception e){
		e.printStackTrace();
	}
	return null;
    }
 
    public String checkForApp(String appid){
	String returnVal = null;
	
	StringBuffer buf = new StringBuffer();
        buf.append("select a.appid from ");
	buf.append("applications a ");
	buf.append("where a.appid=" + appid + "; ");
	try {
              //  System.out.println(buf.toString());
                returnVal = connector.getDB().getDataItem(buf.toString());               
        }catch (Exception ex) {
                ex.printStackTrace();
        }
        
        return returnVal;
    }

    public String checkForApp(String appname, String appversion ){
	String appid = null;
	
	StringBuffer buf = new StringBuffer();
        buf.append("select a.appid from ");
        buf.append("applications a ");
        buf.append("where a.appname='" + appname.trim()                   
                   + "' and a.version='" + appversion.trim() + "'; ");
	try {
                //System.out.println(buf.toString());
                appid = connector.getDB().getDataItem(buf.toString());
                
        }catch (Exception ex) {
                ex.printStackTrace();
        }
    	
	return appid;
    }
	
    public String checkForEnv(String expid){
	String returnVal = null;
	
	StringBuffer buf = new StringBuffer();
        buf.append("select e.expid from ");
	buf.append("experiments e ");
	buf.append("where e.expid=" + expid + "; ");
	try {
              //  System.out.println(buf.toString());
                returnVal = connector.getDB().getDataItem(buf.toString());
                if (returnVal == null) {
                        System.out.println("No proper experiment group found");
                }
        }catch (Exception ex) {
                ex.printStackTrace();
        }
        
        return returnVal;
    }

    public String checkForEnv(String sys, String config, String compiler, String instru, String appid ){
	String expid = null;
	
	StringBuffer buf = new StringBuffer();
        buf.append("select e.expid from ");
        buf.append("experiments e ");
        buf.append("where e.appid='" + appid.trim()
                   + "' and e.sysinfo like '%" + sys
                   + "%' and e.configinfo like '%" + config
                   + "%' and e.instruinfo like '%" + instru
                   + "%' and e.compilerinfo like '%" + compiler + "%'; ");
	try {
                //System.out.println(buf.toString());
                expid = connector.getDB().getDataItem(buf.toString());
                if (expid == null) 
                        System.out.println("Error: No proper experiment group found");
        }catch (Exception ex) {
                ex.printStackTrace();
        }
    	
	return expid;
    }

	//******************************
	//End - Helper functions for buildStatic data.
	//******************************

    static public void main(String[] args){
	String USAGE = "USAGE: Translator configfilename sourcefilename destinationname [problem_size] [application_id] [experiment_id]";
	if (args.length == 0) {
                System.err.println(USAGE);
                System.exit(-1);
        }

	Translator trans = new Translator(args[0], args[1], args[2]);
	trans.buildPprof(); 
	int ctr = 3;
	String problemSize = null, appid = null, expid = null;
	if (ctr < args.length)
		problemSize = args[ctr++];
	if (ctr < args.length)
		appid = args[ctr++];
	if (ctr < args.length)
		expid = args[ctr++];
	trans.writeXmlFiles(problemSize, appid, expid);
	System.out.println("Done - Translating pprof.dat into pprof.xml!");
    }
} 
