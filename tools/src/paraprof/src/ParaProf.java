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
    static SavedPreferences savedPreferences = new SavedPreferences();
    static ParaProfManager paraProfManager = new ParaProfManager();
    static ApplicationManager applicationManager = new ApplicationManager();
    static HelpWindow helpWindow = new HelpWindow();
    //End start of persistant objects.
    
    //Useful in the system.
    private static String USAGE = "ParaProf/ParaProf (help | debug)";
    static Runtime runtime;
    static boolean runHasBeenOpened = false;
     //**********
    
    private int type = -1;
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
		    System.out.println("There was an internal error whilst trying to read the Racy preference");
		    System.out.println("file.  Please delete this file, or replace it with a valid one!");
		    System.out.println("Note: Deleting the file will cause Racy to restore the default preferences");
		}
	    }

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
	//	ParaProf.paraProfManager.show();
    }
    public void update(){}
    //######
    //End - Observer.
    //######

    public static String getInfoString(){
	return new String("ParaProf Version 1.2 ... The Paraducks Group!");}
  
    // Main entry point
    static public void main(String[] args){

	ParaProf paraProf = new ParaProf();

	int position = 0;
	String argument = null;
	//Deal with help and debug individually, then the rest.
	//Help
	while (position < args.length) {
	    argument = args[position++];
	    if (argument.equalsIgnoreCase("HELP")) {
		System.out.println("paraprof/FileList filetype [pprof|profile|dynaprof] | prefix [filename prefix] | help | debug");
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
		argument = args[position++];
		if(argument.equalsIgnoreCase("pprof"))
		    paraProf.type = 0;
		else if(argument.equalsIgnoreCase("profile"))
		    paraProf.type = 1;
		else if(argument.equalsIgnoreCase("dynaprof"))
		    paraProf.type = 2;
	    }
	    else if (argument.equalsIgnoreCase("PREFIX")){
		argument = args[position++];
		paraProf.filePrefix = argument;
	    }
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
