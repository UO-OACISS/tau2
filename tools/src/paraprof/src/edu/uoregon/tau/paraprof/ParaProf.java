/* 
   ParaProf.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import jargs.gnu.CmdLineParser;
import edu.uoregon.tau.dms.dss.*;

public class ParaProf implements ParaProfObserver, ActionListener{
    
    //####################################
    //Class declarations
    //####################################

    //######
    //System wide stuff.
    //######
    static String homeDirectory = null;
    static File paraProfHomeDirectory = null;
    static String profilePathName = null;       //This contains the path to the currently loaded profile data.
    static int defaultNumberPrecision = 4;
    static boolean dbSupport = false;
    static ParaProfLisp paraProfLisp = null;
    static SavedPreferences savedPreferences = null;
    static ParaProfManager paraProfManager = null;
    static ApplicationManager applicationManager = null;
    static HelpWindow helpWindow = null;
    static Runtime runtime = null;
    static boolean runHasBeenOpened = false;
    //######
    //End - System wide stuff.
    //######

    //######
    //Command line options related.
    //######
    public static String USAGE = "USAGE: paraprof [{-f, --filetype} file_type] [{-s,--sourcefile} sourcefilename] [{-p,--filenameprefix} filenameprefix] [{-i --fixnames}] [{-d,--debug} debug]\n\tWhere:\n\t\tfile_type = profiles (TAU), pprof (TAU), dynaprof, mpip, hpm, gprof, psrun, sddf (svpablo)\n";
    private static int fileType = -1; //0:pprof, 1:profile, 2:dynaprof, 3:mpip, 4:hpmtoolkit, 5:gprof, 6:psrun 
    private static boolean dump = false;
    private static int dumptype = -1;
    private static String sourceFile = null;
    private static String filePrefix = null;
    private static boolean fixNames = false;
    //######
    //End - Command line options related.
    //######

    //####################################
    //End - Class declarations
    //####################################
    
    //####################################
    //Instance declarations
    //####################################
    private ParaProfTrial trial = null;
    //####################################
    //End - Instance declarations
    //####################################
    
    public ParaProf(){
	/*try {
	    UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
	} 
	catch (Exception e) { 
	}*/
	//End uncomment!
    }
  
    public void startSystem(){
	try{

	    //######
	    //Initialization of static objects takes place on a need basis. This helps prevent
	    //the creation of a graphical system unless it is absolutly necessary. Static initializations
	    //are marked with "Static Initialization" to make them easy to find.
	    //######

	    //######
	    //Static Initialization
	    //######
	    ParaProf.savedPreferences = new SavedPreferences();
	    //######
	    //End - Static Initialization
	    //######
	    
	    //Establish the presence of a .ParaProf directory.  This is located by default in the user's home
	    //directory.
	    ParaProf.paraProfHomeDirectory = new File(homeDirectory+"/.ParaProf");
	    if(paraProfHomeDirectory.exists()){
		System.out.println("Found ParaProf home directory!");
		System.out.println("Looking for preferences ...");
		//Try and load a preference file ... ParaProfPreferences.dat
		try{
		    FileInputStream savedPreferenceFIS = new FileInputStream(ParaProf.paraProfHomeDirectory.getPath()+"/ParaProf.dat");
		    
		    //If here, means that no exception was thrown, and there is a preference file present.
		    //Create ObjectInputStream and try to read it in.
		    ObjectInputStream inSavedPreferencesOIS = new ObjectInputStream(savedPreferenceFIS);
		    ParaProf.savedPreferences = (SavedPreferences) inSavedPreferencesOIS.readObject();
		}
		catch(Exception e){
		    if(e instanceof FileNotFoundException){
			System.out.println("No preference file present, using defaults!");
		    }
		    else{
			//Print some kind of error message, and quit the system.
			System.out.println("There was an internal error whilst trying to read the ParaProf preference");
			System.out.println("file.  Please delete this file, or replace it with a valid one!");
			System.out.println("Note: Deleting the file will cause ParaProf to restore the default preferences");
		    }
		}

		//Try and find perfdmf.cfg.
		File perfDMFcfg = new File(ParaProf.paraProfHomeDirectory.getPath()+"/perfdmf.cfg");
		if(perfDMFcfg.exists()){
		    System.out.println("Found db configuration file: " + ParaProf.paraProfHomeDirectory.getPath()+"/perfdmf.cfg");
		    ParaProf.savedPreferences.setDatabaseConfigurationFile(ParaProf.paraProfHomeDirectory.getPath()+"/perfdmf.cfg");
		}
		else
		    System.out.println("Did not find db configuration file ... load manually"); 
	    }
	    else{
		System.out.println("Did not find ParaProf home directory ... creating ...");
		paraProfHomeDirectory.mkdir();
		System.out.println("Done creating ParaProf home directory!");
	    }
		
	    //######
	    //Static Initialization
	    //######
	    ParaProf.paraProfLisp = new ParaProfLisp(UtilFncs.debug);
	    //######
	    //End - Static Initialization
	    //######
	    
            //Register lisp primatives in ParaProfLisp.
	    ParaProf.paraProfLisp.registerParaProfPrimitives();

	    //See if the user has defined any lisp code to run.
	    try{
		FileInputStream file = new FileInputStream("ParaProfLisp.lp");
		//If here, means that no exception was thrown, and there is a lisp file present.
		InputStreamReader isr = new InputStreamReader(file);
		BufferedReader br = new BufferedReader(isr);
		
		String inputString = null;
		
		while((inputString = br.readLine()) != null){
		    System.out.println("Expression: " + inputString);
		    System.out.println(ParaProf.paraProfLisp.eval(inputString));
		}
		
	    }
	    catch(Exception e){
		if(e instanceof FileNotFoundException){
		    System.out.println("No ParaProfLisp.lp file present!");
		}
		else{
		    //Print some kind of error message, and quit the system.
		    System.out.println("There was an internal error whilst trying to read the ParaProfLisp.pl");
		    System.out.println("Please delete this file, or replace it with a valid one!");
		}
	    }



	    //######
	    //Static Initialization
	    //######
	    ParaProf.applicationManager = new ApplicationManager();
	    //######
	    //End - Static Initialization
	    //######

            //Create a default application.
	    ParaProfApplication app = ParaProf.applicationManager.addApplication();
	    app.setName("Default App");
	    
	    //Create a default experiment.
	    ParaProfExperiment experiment = app.addExperiment();
	    experiment.setName("Default Exp");
	    ParaProfDataSession dataSession = null;
	    FileList fl = new FileList();
	    Vector v = null;

	    //######
	    //Static Initialization
	    //######
	    ParaProf.helpWindow = new HelpWindow(UtilFncs.debug);
	    ParaProf.paraProfManager = new ParaProfManager();
	    //######
	    //End - Static Initialization
	    //######
	    System.out.println("fileType:"+fileType );
	    if(fileType!=-1){
		switch(fileType){
		    case 0:
			dataSession = new TauPprofOutputSession();
			if(sourceFile==null){
			    if(filePrefix==null)
				v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, "pprof", UtilFncs.debug);
			    else
				v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, filePrefix, UtilFncs.debug);
			}
			else{
			    v = new Vector();
			    File file = new File(sourceFile);
			    if(file.exists()){
				File[] files = new File[1];
				files[0] = file;
				v.add(files);
				fl.setFileList(v);
				fl.setPath(file.getPath());
			    }
			}
			break;
		    case 1:
			dataSession = new TauOutputSession();
			if(sourceFile==null){
			    if(filePrefix==null)
				v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, "profile", UtilFncs.debug);
			    else
				v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, filePrefix, UtilFncs.debug);
			}
			else{
			    v = new Vector();
			    File file = new File(sourceFile);
			    if(file.exists()){
				File[] files = new File[1];
				files[0] = file;
				v.add(files);
				fl.setFileList(v);
				fl.setPath(file.getPath());
			    }
			}
			break;
		    case 2:
			dataSession = new DynaprofOutputSession();
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, filePrefix, UtilFncs.debug);
			break;
		    case 3:
			dataSession = new MpiPOutputSession();
			v = new Vector();
			if(sourceFile!=null){
			    File file = new File(sourceFile);
			    if(file.exists()){
				File[] files = new File[1];
				files[0] = file;
				v.add(files);
				fl.setFileList(v);
				fl.setPath(file.getPath());
			    }
			}
			break;
		    case 4:
			dataSession = new HPMToolkitDataSession();
			v = new Vector();
			if(sourceFile==null){
			    if(filePrefix!=null)
				v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, filePrefix, UtilFncs.debug);
			}
			else{
			    File file = new File(sourceFile);
			    if(file.exists()){
				File[] files = new File[1];
				files[0] = file;
				v.add(files);
				fl.setFileList(v);
				fl.setPath(file.getPath());
			    }
			}
			break;
		    case 5:
			dataSession = new GprofOutputSession(fixNames);
			v = new Vector();
			if(sourceFile==null){
			    if(filePrefix!=null)
				v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, filePrefix, UtilFncs.debug);
			}
			else{
			    File file = new File(sourceFile);
			    if(file.exists()){
				File[] files = new File[1];
				files[0] = file;
				v.add(files);
				fl.setFileList(v);
				fl.setPath(file.getPath());
			    }
			}
			break;
		    case 6:
			dataSession = new PSRunDataSession();
			v = new Vector();
			if(sourceFile==null){
			    if(filePrefix!=null)
				v = fl.getFileList(new File(System.getProperty("user.dir")), null, fileType, filePrefix, UtilFncs.debug);
			}
			else{
			    File file = new File(sourceFile);
			    if(file.exists()){
				File[] files = new File[1];
				files[0] = file;
				v.add(files);
				fl.setFileList(v);
				fl.setPath(file.getPath());
			    }
			}
			break;
		    default:
			v = new Vector();
			System.out.println("Unrecognized file type.");
			System.out.println("Use ParaProf's manager window to load them manually.");
			break;
		}
		if(v.size()>0){
		    trial = new ParaProfTrial(fileType);
		    trial.setApplicationID(0);
		    trial.setExperimentID(0);
		    trial.setID(0);
		    trial.setName("Default Trial");
		    trial.setDefaultTrial(true);
		    trial.setPaths(fl.getPath());
		    experiment.addTrial(trial);
		    trial.setLoading(true);
		    dataSession.setDebug(UtilFncs.debug);
		    DataSessionThreadControl dataSessionThreadControl = new DataSessionThreadControl();
		    dataSessionThreadControl.setDebug(UtilFncs.debug);
		    dataSessionThreadControl.addObserver(this);
		    dataSessionThreadControl.initialize(dataSession,v,true);
		}
		else{
		    System.out.println("No profile files found in the current directory.");
		    System.out.println("Use ParaProf's manager window to load them manually.");
		}
	    }
	    else{
		if(sourceFile==null){
		    if(filePrefix==null)
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, 0 , "pprof", UtilFncs.debug);
		    else
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, 0 , filePrefix, UtilFncs.debug);
		}
		else{
		    v = new Vector();
		    File file = new File(sourceFile);
		    if(file.exists()){
			File[] files = new File[1];
			files[0] = file;
			v.add(files);
			fl.setFileList(v);
			fl.setPath(file.getPath());
		    }
		}
		if(v.size()>0){
		    dataSession = new TauPprofOutputSession();
		    trial = new ParaProfTrial(0);
		    trial.setApplicationID(0);
		    trial.setExperimentID(0);
		    trial.setID(0);
		    trial.setName("Default Trial");
		    trial.setDefaultTrial(true);
		    trial.setPaths(fl.getPath());
		    experiment.addTrial(trial);
		    trial.setLoading(true);
		    dataSession.setDebug(UtilFncs.debug);
		    DataSessionThreadControl dataSessionThreadControl = new DataSessionThreadControl();
		    dataSessionThreadControl.setDebug(UtilFncs.debug);
		    dataSessionThreadControl.addObserver(this);
		    dataSessionThreadControl.initialize(dataSession,v,true);
		}
		else{
		    //Try finding profile.*.*.* files.
		    if(sourceFile==null){
			if(filePrefix==null)
			    v = fl.getFileList(new File(System.getProperty("user.dir")), null, 1 , "profile", UtilFncs.debug);
			else
			    v = fl.getFileList(new File(System.getProperty("user.dir")), null, 1 , filePrefix, UtilFncs.debug);
		    }
		    else{
			v = new Vector();
			File file = new File(sourceFile);
			if(file.exists()){
			    File[] files = new File[1];
			    files[0] = file;
			    v.add(files);
			    fl.setFileList(v);
			    fl.setPath(file.getPath());
			}
		    }
		    if(v.size()>0){
			dataSession = new TauOutputSession();
			trial = new ParaProfTrial(1);
			trial.setApplicationID(0);
			trial.setExperimentID(0);
			trial.setID(0);
			trial.setName("Default Trial");
			trial.setDefaultTrial(true);
			trial.setPaths(fl.getPath());
			experiment.addTrial(trial);
			trial.setLoading(true);
			dataSession.setDebug(UtilFncs.debug);
			DataSessionThreadControl dataSessionThreadControl = new DataSessionThreadControl();
			dataSessionThreadControl.setDebug(UtilFncs.debug);
			dataSessionThreadControl.addObserver(this);
			dataSessionThreadControl.initialize(dataSession,v,true);
		    }
		    else{
			System.out.println("No profile files found in the current directory.");
			System.out.println("Use ParaProf's manager window to load them manually.");
		    }
		}
	    }
	}
	catch (Exception e) {
  	    System.out.println("An un-caught exception has occurred within the program!");
	    System.out.println("The details of this execption has been stored in a file named: exception.err");
	    System.out.println("Please email this file to Robert Bell at bertie@cs.uoregon.edu ");
	    e.printStackTrace();
	}
    }

    public void actionPerformed(ActionEvent evt){
	Object EventSrc = evt.getSource();
    	if(EventSrc instanceof javax.swing.Timer){
	    System.out.println("------------------------");
	    System.out.println("The amount of memory used by the system is: " + runtime.totalMemory());
	    System.out.println("The amount of memory free to the system is: " + runtime.freeMemory());
	}
    }

    //######
    //ParaProfObserver interface.
    //######
    public void update(Object obj){
	//We are only ever watching an instance of DataSession.
	DataSession dataSession = (DataSession)obj;

	//Data session has finished loading. Call its terminate method to
	//ensure proper cleanup.
	dataSession.terminate();

	//Need to update setup the DataSession object for correct usage in
	//in a ParaProfTrial object.
	//Set the colours.
	trial.getColorChooser().setColors(dataSession.getGlobalMapping(), -1);
	//The dataSession has accumulated edu.uoregon.tau.dms.dss.Metrics. Inside ParaProf,
	//these need to be paraprof.Metrics.
	int numberOfMetrics = dataSession.getNumberOfMetrics();
	Vector metrics = new Vector();
	for(int i=0;i<numberOfMetrics;i++){
	    Metric metric = new Metric();
	    metric.setName(dataSession.getMetricName(i));
	    metric.setID(i);
	    metric.setTrial(trial);
	    metrics.add(metric);
	}
	//Now set the data session metrics.
	dataSession.setMetrics(metrics);

	//Now set the trial's DataSession object to be this one.
	trial.setDataSession(dataSession);
	//Set loading to be false, and indicate to ParaProfManager that it should display
	//the path to this (default) trial.
	trial.setLoading(false);
	ParaProf.paraProfManager.populateTrialMetrics(trial);
	trial.showMainWindow();

	//See if the user has defined any lisp code to run.
	try{
	    FileInputStream file = new FileInputStream("ParaProfLisp.lp");
	    //If here, means that no exception was thrown, and there is a lisp file present.
	    InputStreamReader isr = new InputStreamReader(file);
	    BufferedReader br = new BufferedReader(isr);
	    
	    String inputString = null;
	    
	    while((inputString = br.readLine()) != null){
		System.out.println("Expression: " + inputString);
		System.out.println(ParaProf.paraProfLisp.eval(inputString));
	    }
	    
	}
	catch(Exception e){
	    if(e instanceof FileNotFoundException){
		System.out.println("No ParaProfLisp.lp file present!");
	    }
	    else{
		//Print some kind of error message, and quit the system.
		System.out.println("There was an internal error whilst trying to read the ParaProfLisp.pl");
		System.out.println("Please delete this file, or replace it with a valid one!");
	    }
	}
    }
    public void update(){}
    //######
    //End - Observer.
    //######

    public static String getInfoString(){
	return new String("ParaProf Version 2.0 ... The Tau Group!");}

    //This method is reponsible for any cleanup required in ParaProf before an exit takes place.
    public static void exitParaProf(int exitValue){
	if(UtilFncs.objectDebug != null){
	    UtilFncs.objectDebug.outputToFile("ParaProf exiting!!");
	    UtilFncs.objectDebug.flushDebugFileBuffer();
	    UtilFncs.objectDebug.closeDebugFile();
	}
	System.exit(exitValue);
    }
  
    // Main entry point
    static public void main(String[] args){

	//######
	//Static Initialization
	//######
	UtilFncs.objectDebug = new Debug();
	//######
	//End - Static Initialization
	//######
	
	//Make sure we drop a line before beginning any output.
	System.out.println("");

	/*
	System.out.println("------");
	System.out.println("Available properties:");
	Properties p = System.getProperties();
	for(Enumeration e = p.propertyNames();e.hasMoreElements();){
	    System.out.println(e.nextElement());
	}
	System.out.println("------");
	*/

	ParaProf.homeDirectory = System.getProperty("user.home");

	/*
	if(System.getProperty("user.name").equals("sameer")){
	    JOptionPane.showMessageDialog(null,"Sorry, user \"sameer\" detected. We no longer support this user!", "ParaProf Error", JOptionPane.ERROR_MESSAGE);
	    exitParaProf(-1);
	    }*/

	//Bring ParaProf into being!
	ParaProf paraProf = new ParaProf();

	//######
	//Process command line arguments.
	//ParaProf has numerous modes of operation. A number of these mode
	//can be specified on the command line.
	//######
	CmdLineParser parser = new CmdLineParser();
        CmdLineParser.Option helpOpt = parser.addBooleanOption('h', "help");
	CmdLineParser.Option debugOpt = parser.addBooleanOption('d', "debug");
        CmdLineParser.Option configfileOpt = parser.addStringOption('g', "configfile");
        CmdLineParser.Option sourcefileOpt = parser.addStringOption('s', "sourcefile");
	CmdLineParser.Option prefixOpt = parser.addStringOption('p', "--filenameprefix");
	CmdLineParser.Option typeOpt = parser.addStringOption('f', "filetype");
	CmdLineParser.Option fixOpt = parser.addBooleanOption('i', "fixnames");
	try {
            parser.parse(args);
        }
        catch ( CmdLineParser.OptionException e ) {
            System.err.println(e.getMessage());
	    System.err.println(ParaProf.USAGE);
	    exitParaProf(-1);
        }

	Boolean help = (Boolean)parser.getOptionValue(helpOpt);
	Boolean debug = (Boolean)parser.getOptionValue(debugOpt);
        ParaProf.sourceFile = (String)parser.getOptionValue(sourcefileOpt);
        String fileTypeString = (String)parser.getOptionValue(typeOpt);
	ParaProf.filePrefix = (String)parser.getOptionValue(prefixOpt);
	Boolean fixNames = (Boolean)parser.getOptionValue(fixOpt);
	
	if(help!=null && help.booleanValue()){
	    System.err.println(ParaProf.USAGE);
	    exitParaProf(-1);
    	}

	if(fixNames!=null)
	    ParaProf.fixNames = fixNames.booleanValue();

	if(debug!=null){
	    UtilFncs.debug = debug.booleanValue();
	    UtilFncs.objectDebug.debug = debug.booleanValue();
	}

	if(fileTypeString != null){
	    if(fileTypeString.equals("pprof")) {
		ParaProf.fileType = 0;
	    }else if (fileTypeString.equals("profiles")) {
		ParaProf.fileType = 1;
	    }else if (fileTypeString.equals("dynaprof")) {
		ParaProf.fileType = 2;
	    }else if (fileTypeString.equals("mpip")) {
		ParaProf.fileType = 3;
	    }else if (fileTypeString.equals("hpm")) {
		ParaProf.fileType = 4;
	    }else if (fileTypeString.equals("gprof")) {
		ParaProf.fileType = 5;
	    }else if (fileTypeString.equals("psrun")) {
		ParaProf.fileType = 6;
/*
  } else if (fileTypeString.equals("sppm")) {
  ParaProf.fileType = 101;
  } else if (fileTypeString.equals("xprof")) {
  ParaProf.fileType = 0;
  } else if (fileTypeString.equals("sddf")) {
  ParaProf.fileType = 0;
*/
	    }else{
		System.err.println("Please enter a valid file type.");
	    	System.err.println(USAGE);
	    	exitParaProf(-1);
	    }
	}

	if(((sourceFile!=null)||(filePrefix!=null))&&(fileType==-1)){
	    System.out.println("Error: If you specify either a source file or a prefix, you must specify a file type as well!");
	    System.err.println(USAGE);
	    exitParaProf(-1);
	}
	if((fileType==2||fileType==3)&&sourceFile==null){
	    System.out.println("Error: If you specify either dynaprof or mpip, you must specify a source file as well!");
	    System.err.println(USAGE);
	    exitParaProf(-1);
	}
	//######
	//End - Process command line arguments.
	//######
	if(paraProf.dump){
	    System.out.println("ParaProf will dump to the standard out with dump type: " + paraProf.dumptype);
	    exitParaProf(0);
	}
	
	ParaProf.runtime = Runtime.getRuntime();
	
	if(UtilFncs.debug){
	    //Create and start the a timer, and then add racy to it.
	    javax.swing.Timer jTimer = new javax.swing.Timer(8000, paraProf);
	    jTimer.start();
	}
	
	paraProf.startSystem();
    }
}
