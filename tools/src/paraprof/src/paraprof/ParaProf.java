/* 
  ParaProf.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package ParaProf;
import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import javax.swing.colorchooser.*;

public class ParaProf implements ActionListener
{
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
  
  public ParaProf() 
  {
    try {
      UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
    } 
    catch (Exception e) { 
    }
    //End uncomment!
  }
  
  public void startSystem(){
  try{
      //Try and load the Racy preference file ... racyPreferences.dat
      try{
        FileInputStream savedPreferenceFIS = new FileInputStream("ParaProfPreferences.dat");
        
        //If here, means that no exception was thrown, and there is a preference file present.
        //Create ObjectInputStream and try to read it in.
        ObjectInputStream inSavedPreferencesOIS = new ObjectInputStream(savedPreferenceFIS);
        ParaProf.savedPreferences = (SavedPreferences) inSavedPreferencesOIS.readObject();
      }
      catch(Exception e)
      {
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
      
      //Ok, now try to add the default experiment.
      
      //Check to see if a "pprof.dat" file exists.  If it does, load it.
      File testForPprofDat = new File("pprof.dat");
      
      if(testForPprofDat.exists())
      {
        System.out.println("Found pprof.dat!");
        
        //setTitle("ParaProf: " + ParaProf.profilePathName);
        
        //Create a default application.
        Application app = ParaProf.applicationManager.addApplication();
        app.setApplicationName("Default App");
        
        //Create a default experiment.
        Experiment exp = app.addExperiment();
        exp.setExperimentName("Default Exp");
        
        //Add the trial for this pprof.dat file to the experiment.
        Trial trial = null;
        String tmpString1 = null;
        String tmpString2 = null;
        String tmpString3 = null;
        
        tmpString1 = testForPprofDat.getCanonicalPath();
        tmpString2 = ParaProf.applicationManager.getPathReverse(tmpString1);
        tmpString3 = "Default Trial" + " : " + tmpString2;
                                    
        trial = exp.addTrial();
            
        trial.setProfilePathName(tmpString1);
        trial.setProfilePathName(tmpString2);
        trial.setTrialName(tmpString3);
        
        trial.buildStaticData(testForPprofDat);
      }
      else
      {
        boolean foundSomething = false;
        
        File file = new File(".");
        Experiment exp = null;
        Trial trial = null;
      
        String filePath = file.getCanonicalPath();
        File [] list = file.listFiles();
        for(int i = 0; i < list.length; i++)
        {
          File tmpFile = (File) list[i];
          if(tmpFile != null){
            String tmpString = tmpFile.getName();
            
            if(tmpString.indexOf("MULTI__") != -1){
              String newString = filePath + "/" + tmpString + "/pprof.dat";
              File testFile = new File(newString);
              
              if(testFile.exists()){
                if(!foundSomething){
                  System.out.println("Found pprof.dat ... loading");
                  
                  //setTitle("ParaProf: " + ParaProf.profilePathName);
                  
                  //Create a default application.
                  Application app = ParaProf.applicationManager.addApplication();
                  app.setApplicationName("Default App");
                  
                  //Create a default experiment.
                  exp = app.addExperiment();
                  exp.setExperimentName("Default Exp");
                  
                  //Add the experiment run for this pprof.dat file to the experiment.
                  String tmpString1 = null;
                  String tmpString2 = null;
                  String tmpString3 = null;
                  
                  tmpString1 = filePath;
                  tmpString2 = ParaProf.applicationManager.getPathReverse(tmpString1);
                  tmpString3 = "Default Trial" + " : " + tmpString2;
                                              
                  trial = exp.addTrial();
                      
                  trial.setProfilePathName(tmpString1);
                  trial.setProfilePathName(tmpString2);
                  trial.setTrialName(tmpString3);
                  
                  trial.buildStaticData(testFile);
                  
                  System.out.println("Found: " + newString);
                  
                  foundSomething = true;
                }
                else{
                  trial.buildStaticData(testFile);
                } 
              }
            }
          }     
        }
        
        if(!foundSomething)
          System.out.println("Did not find pprof.dat!");
      }
      
      ParaProfManager jRM = new ParaProfManager();
      jRM.expandDefaultTrialNode();
      
      //Bring up the main window.
      //staticMainWindow = new StaticMainWindow();
      //ParaProf.systemEvents.addObserver(staticMainWindow);
      //staticMainWindow.setVisible(true);
      /*
      //Now show the welcome window.  
      RacyWelcomeWindow test = new RacyWelcomeWindow();
      test.setVisible(true);
      */
    }
    catch (Exception e) {
    
      System.out.println("An un-caught exception has occurred within the program!");
      System.out.println("The details of this execption has been stored in a file named: exception.err");
      System.out.println("Please email this file to Robert Bell at bertie@cs.uoregon.edu ");
      e.printStackTrace();
    }
  }
    
  
  public void actionPerformed(ActionEvent evt)
  {
    Object EventSrc = evt.getSource();
    
    if(EventSrc instanceof javax.swing.Timer)
    {
      System.out.println("------------------------");
      System.out.println("The amount of memory used by the system is: " + runtime.totalMemory());
      System.out.println("The amount of memory free to the system is: " + runtime.freeMemory());
    }
  }
  
  //The about ParaProf info. string.
  public static String getInfoString()
  {
    return new String("ParaProf Version 1.2 ... The Paraducks Group!");
  }
  
  //Handles system errors.
  public static void systemError(Object inObject, Component inComponent, String inString)
  { 
    JOptionPane.showMessageDialog(inComponent, "ParaProf Error", "Internal System Error ... ParaProf will now close!", JOptionPane.ERROR_MESSAGE);
    
    
    if(inObject != null){
      if(inObject instanceof Exception){
        if(ParaProf.debugIsOn){
          System.out.println(((Exception) inObject).toString());
          System.out.println("");
          System.out.println("");
        }
        
        System.out.println("An exception was caught at " + inString);
      }
      else{
        System.out.println("An error was detected at " + inString);
      }
    }
    else{
      System.out.println("An error was detected at " + inString);
    }
    System.out.println("Please email us at: tau-bugs@cs.uoregon.edu");
    System.out.println("");
    System.out.println("If possible, include the profile files that caused this error,");
    System.out.println("and a brief desciption your sequence of operation.");
    System.out.println("");
    System.out.println("Also email this error message,as it will tell us where the error occured.");
    System.out.println("");
    System.out.println("Thank you for your help!");
    
    System.exit(0);
  }
      

  // Main entry point
  static public void main(String[] args) 
  {
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
    ParaProf racy = new ParaProf();
    
    if(debugIsOn)
    {
      //Create and start the a timer, and then add racy to it.
      javax.swing.Timer jTimer = new javax.swing.Timer(8000, racy);
      jTimer.start();
    }
    
    racy.startSystem();
  }
  
}
