/* 
   ParaProf.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import paraprof.*;
import dms.dss.*;

public class ParaProf implements ParaProfObserver, ActionListener{
    //**********
    //Some system wide state variables.
    static String profilePathName = null;       //This contains the path to the currently loaded profile data.
    static int defaultNumberPrecision = 4;
    static boolean dbSupport = false;
    //End - Some system wide state variables.
    //**********
    
    //**********
    //Start or define all the persistant objects.
    static ParaProfLisp paraProfLisp = null;
    static SavedPreferences savedPreferences = new SavedPreferences();
    static ParaProfManager paraProfManager = new ParaProfManager();
    static ApplicationManager applicationManager = new ApplicationManager();
    static HelpWindow helpWindow = new HelpWindow(UtilFncs.debug);
    //End start of persistant objects.
    
    //Useful in the system.
    private static String USAGE = "ParaProf/ParaProf (help | debug)";
    static Runtime runtime;
    static boolean runHasBeenOpened = false;
     //**********
    
    private int type = -1;
    private boolean dump = false;
    private int dumptype = -1;
    String filePrefix = null;
    
    public ParaProf(){
	try {
	    UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
	} 
	catch (Exception e) { 
	}
	//End uncomment!
    }
  
    public void startSystem(){
	try{
	    //Try and load a preference file ... ParaProfPreferences.dat
	    try{
		FileInputStream savedPreferenceFIS = new FileInputStream("ParaProfPreferences.dat");
        
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

	    paraProfLisp = new ParaProfLisp(UtilFncs.debug);
	    //Register lisp primatives in ParaProfLisp.
	    ParaProf.paraProfLisp.registerParaProfPrimitives();

	    //Create a default application.
	    ParaProfApplication app = ParaProf.applicationManager.addApplication();
	    app.setName("Default App");
	    
	    //Create a default experiment.
	    ParaProfExperiment experiment = app.addExperiment();
	    experiment.setName("Default Exp");
	    ParaProfTrial trial = null;
	    FileList fl = new FileList();
	    Vector v = null;
	    if(type!=-1){
		switch(type){
		case 0:
		    if(filePrefix==null)
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, type, "pprof", UtilFncs.debug);
		    else
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, type, filePrefix, UtilFncs.debug);
		    break;
		case 1:
		    if(filePrefix==null)
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, type, "profile", UtilFncs.debug);
		    else
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, type, filePrefix, UtilFncs.debug);
		    break;
		case 2:
		    v = fl.getFileList(new File(System.getProperty("user.dir")), null, type, filePrefix, UtilFncs.debug);
		    break;
		case 5:
		    if(filePrefix==null)
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, type, "gprof", UtilFncs.debug);
		    else
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, type, filePrefix, UtilFncs.debug);
		    break;
		default:
		    v = new Vector();
		    System.out.println("Unrecognized file type.");
		    System.out.println("Use ParaProf's manager window to load them manually.");
		    break;
		}
		if(v.size()>0){
		    trial = new ParaProfTrial(null, type);
		    trial.addObserver(this);
		    trial.setName("Default Trial");
		    trial.setDefaultTrial(true);
		    trial.setPaths(fl.getPath());
		    experiment.addTrial(trial);
		    trial.setLoading(true);
		    ParaProf.paraProfManager.populateTrialMetrics(trial, true);
		    trial.initialize(v);
		}
		else{
		    System.out.println("No profile files found in the current directory.");
		    System.out.println("Use ParaProf's manager window to load them manually.");
		}
	    }
	    else{
		if(filePrefix==null)
		    v = fl.getFileList(new File(System.getProperty("user.dir")), null, 0 , "pprof", UtilFncs.debug);
		else
		    v = fl.getFileList(new File(System.getProperty("user.dir")), null, 0 , filePrefix, UtilFncs.debug);
		if(v.size()>0){
		    trial = new ParaProfTrial(null, 0);
		    trial.addObserver(this);
		    trial.setName("Default Trial");
		    trial.setDefaultTrial(true);
		    trial.setPaths(fl.getPath());
		    experiment.addTrial(trial);
		    trial.setLoading(true);
		    ParaProf.paraProfManager.populateTrialMetrics(trial, true);
		    trial.initialize(v);
		}
		else{
		    //Try finding profile.*.*.* files.
		    if(filePrefix==null) 
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, 1 , "profile", UtilFncs.debug);
		    else
			v = fl.getFileList(new File(System.getProperty("user.dir")), null, 1 , filePrefix, UtilFncs.debug);
		    if(v.size()>0){
			trial = new ParaProfTrial(null, 1);
			trial.addObserver(this);
			trial.setName("Default Trial");
			trial.setDefaultTrial(true);
			trial.setPaths(fl.getPath());
			experiment.addTrial(trial);
			trial.setLoading(true);
			ParaProf.paraProfManager.populateTrialMetrics(trial, true);
			trial.initialize(v);
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
	//We are only ever watching an instance of ParaProfTrial.
	ParaProfTrial trial = (ParaProfTrial) obj;
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
	return new String("ParaProf Version 1.2 ... The Paraducks Group!");}
  
    // Main entry point
    static public void main(String[] args){
	//Bring ParaProf into being!
	ParaProf paraProf = new ParaProf();

	//######
	//Process command line arguments.
	//ParaProf has numerous modes of operation. A number of these mode
	//can be specified on the command line.
	//######
	int position = 0;
	String argument = null;
	//Deal with help and debug individually, then the rest.
	//Help
	while (position < args.length) {
	    argument = args[position++];
	    if (argument.equalsIgnoreCase("HELP")) {
		System.out.println("-----------------------------------");
		System.out.println("ParaProf accepts the arguments below.");
		System.out.println("If an incorrect combination is given, an error will be generated.");
		System.out.println("For any assitance, please email tau-bugs@cs.uoregon.edu");
		System.out.println("Thank you!");
		System.out.println("------");
		System.out.println("help - prints this message.");
		System.out.println("debug - Causes ParaProf to output debugging information (some to file, and some to the standard out).");
		System.out.println("prefix  - prefix path for ParaProf to look for profile data (the default is the current directory).");
		System.out.println("filetype [pprof|profile|dynaprof] - the type of profile data to look for.");
		System.out.println("dump [pprof|standard] - Data is dumped to the standard out.");
		System.out.println("------");
		System.out.println("Some examples:");
		System.out.println("paraprof/ParaProf debug");
		System.out.println("paraprof/ParaProf prefix /tmp/data debug");
		System.out.println("paraprof/ParaProf prefix /tmp/data filetype dynaprof");
		System.out.println("-----------------------------------");
		System.exit(0);
	    }
	}
	//Debug
	position = 0;
	while (position < args.length) {
	    argument = args[position++];
	    if (argument.equalsIgnoreCase("DEBUG")) {
		UtilFncs.debug = true;
	    }
	}
	//Now the rest.
	position = 0;
	while (position < args.length) {
	    argument = args[position++];
	    if (argument.equalsIgnoreCase("FILETYPE")){
		if(args.length==position){
		    System.out.println("No file type given!");
		    System.exit(0);
		}
		argument = args[position++];
		if(argument.equalsIgnoreCase("pprof"))
		    paraProf.type = 0;
		else if(argument.equalsIgnoreCase("profile"))
		    paraProf.type = 1;
		else if(argument.equalsIgnoreCase("dynaprof"))
		    paraProf.type = 2;
		else if(argument.equalsIgnoreCase("gprof"))
		    paraProf.type = 5;
		else{
		    System.out.println("Unrecognized file type: " + argument);
		    System.exit(0);
		}
	    }
	    else if (argument.equalsIgnoreCase("PREFIX")){
		argument = args[position++];
		paraProf.filePrefix = argument;
	    }
	    if (argument.equalsIgnoreCase("DUMP")){
		paraProf.dump = true;
		if(args.length==position){
		    System.out.println("No dump type given!");
		    System.exit(0);
		}
		argument = args[position++];
		if(argument.equalsIgnoreCase("pprof"))
		    paraProf.dumptype = 0;
		else if(argument.equalsIgnoreCase("standard"))
		    paraProf.dumptype = 1;
		else{
		    System.out.println("Unrecognized dump type: " + argument);
		    System.exit(0);
		}
	    }
	}
	//######
	//End - Process command line arguments.
	//######

	if(paraProf.dump){
	    System.out.println("ParaProf will dump to the standard out with dump type: " + paraProf.dumptype);
	    System.exit(0);
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
