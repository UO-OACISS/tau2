
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

public class ParaProf implements ActionListener{
    //**********
    //Some system wide state variables.
    static boolean debugIsOn = false;         //Flip this if debugging output is required.
    static String profilePathName = null;       //This contains the path to the currently loaded profile data.
    static int defaultNumberPrecision = 4;
    static boolean dbSupport = false;
    //End - Some system wide state variables.
    //**********
    
    //**********
    //Start or define all the persistant objects.
    static SavedPreferences savedPreferences = new SavedPreferences();
    static ApplicationManager applicationManager = new ApplicationManager();
    static HelpWindow helpWindow = new HelpWindow();
    //End start of persistant objects.
    
    //Useful in the system.
    private static String USAGE = "ParaProf/ParaProf (help | debug)";
    static Runtime runtime;
    static boolean runHasBeenOpened = false;
    //**********
    
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

	    ParaProfTrial trial = new ParaProfTrial(0);
	    trial.setName("Default Trial");
	    FileList fl = new FileList();
	    Vector v = fl.getFileList(new File(System.getProperty("user.dir")), null, 0 ,ParaProf.debugIsOn);
	    if(v.size()>0){
		trial.initialize(v);
		experiment.addTrial(trial);
		trial.showMainWindow();
	    }
	    else{
		//Try finding profile.*.*.* files.
		trial = new ParaProfTrial(1);
		trial.setName("Default Trial");
		 v = fl.getFileList(new File(System.getProperty("user.dir")), null, 1 ,ParaProf.debugIsOn);
		 if(v.size()>0){
		     trial.initialize(v);
		     //experiment.addTrial(trial);
		     //trial.showMainWindow();
		 }
		 else{
		     System.out.println("No profile files found in the current directory.");
		     System.out.println("Use ParaProf's manager window to load them manually.");
		 }
	    }

	    ParaProfManager paraProfManager = new ParaProfManager();
	    paraProfManager.expandDefaultParaProfTrialNode();
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

    public static String getInfoString(){
	return new String("ParaProf Version 1.2 ... The Paraducks Group!");}
  
    //Handles system errors.
    public static void systemError(Object obj, Component component, String string){ 
	System.out.println("####################################");
	boolean quit = true; //Quit by default.
	if(obj != null){
	    if(obj instanceof Exception){
		Exception exception = (Exception) obj;
		if(ParaProf.debugIsOn){
		    System.out.println(exception.toString());
		    exception.printStackTrace();
		    System.out.println("\n");
		}
		System.out.println("An error was detected: " + string);
		System.out.println(ParaProfError.contactString);
	    }
	    if(obj instanceof ParaProfError){
		ParaProfError paraProfError = (ParaProfError) obj;
		if(ParaProf.debugIsOn){
		    if((paraProfError.showPopup)&&(paraProfError.popupString!=null))
			JOptionPane.showMessageDialog(paraProfError.component,
						      "ParaProf Error", paraProfError.popupString, JOptionPane.ERROR_MESSAGE);
		    if(paraProfError.exp!=null){
			System.out.println(paraProfError.exp.toString());
			paraProfError.exp.printStackTrace();
			System.out.println("\n");
		    }
		    if(paraProfError.location!=null)
			System.out.println("Location: " + paraProfError.location);
		    if(paraProfError.s0!=null)
			System.out.println(paraProfError.s0);
		    if(paraProfError.s1!=null)
			System.out.println(paraProfError.s1);
		    if(paraProfError.showContactString)
			System.out.println(ParaProfError.contactString);
		}
		else{
		    if((paraProfError.showPopup)&&(paraProfError.popupString!=null))
			JOptionPane.showMessageDialog(paraProfError.component,
						      "ParaProf Error", paraProfError.popupString, JOptionPane.ERROR_MESSAGE);
		    if(paraProfError.location!=null)
			System.out.println("Location: " + paraProfError.location);
		    if(paraProfError.s0!=null)
			System.out.println(paraProfError.s0);
		    if(paraProfError.s1!=null)
			System.out.println(paraProfError.s1);
		    if(paraProfError.showContactString)
			System.out.println(ParaProfError.contactString);
		}
		quit = paraProfError.quit;
	    }
	    else{
		System.out.println("An error has been detected: " + string);
	    }
	}
	else{
	    System.out.println("An error was detected at " + string);
	}
	System.out.println("####################################");
	if(quit)
	    System.exit(0);
    }

    // Main entry point
    static public void main(String[] args){
	int numberOfArguments = 0;
	String argument;
	
	while (numberOfArguments < args.length) {
	    argument = args[numberOfArguments++];
	    if (argument.equalsIgnoreCase("HELP")) {
		System.err.println(USAGE);
		System.exit(-1);
	    }
	    if (argument.equalsIgnoreCase("DEBUG")) {
		ParaProf.debugIsOn = true;
		continue;
	    }
	}
    
	ParaProf.runtime = Runtime.getRuntime();
    
	//Start Racy.
	ParaProf paraProf = new ParaProf();
    
	if(debugIsOn){
	    //Create and start the a timer, and then add racy to it.
	    javax.swing.Timer jTimer = new javax.swing.Timer(8000, paraProf);
	    jTimer.start();
	}
	
	paraProf.startSystem();
    }
}
