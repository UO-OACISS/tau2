/* 
	jRacy.java

	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;
import java.util.*;
import java.lang.*;
import java.io.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.*;
import javax.swing.event.*;
import javax.swing.colorchooser.*;

public class jRacy implements ActionListener
{
	//**********
	//Some system wide state variables.
	static boolean debugIsOn = false;					//Flip this if debugging output is required.
	static String profilePathName = null;				//This contains the path to the currently loaded profile data.
	//End - Some system wide state variables.
	//**********
	
	//**********
	//Start or define all the persistant objects.
	static SavedPreferences savedPreferences = new SavedPreferences();
	static ExperimentManager experimentManager = new ExperimentManager();
	static HelpWindow helpWindow = new HelpWindow();
	//End start of persistant objects.
	
	//Useful in the system.
	private static String USAGE = "jRacy/jRacy (help | debug)";
	static Runtime runtime;
	static boolean runHasBeenOpened = false;
	//**********

	public jRacy() 
	{
		try {
			// For native Look and Feel, uncomment the following code.
			
			try {
				UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
			} 
			catch (Exception e) { 
			}
			//End uncomment!
			
			//Add some observers.
			//jRacy.systemEvents.addObserver(helpWindow);
			
			//Try and load the Racy preference file ... racyPreferences.dat
			try
			{
				FileInputStream savedPreferenceFIS = new FileInputStream("jRacyPreferences.dat");
				
				//If here, means that no exception was thrown, and there is a preference file present.
				//Create ObjectInputStream and try to read it in.
				ObjectInputStream inSavedPreferencesOIS = new ObjectInputStream(savedPreferenceFIS);
				jRacy.savedPreferences = (SavedPreferences) inSavedPreferencesOIS.readObject();
				
				//jRacy.clrChooser = new ColorChooser(savedPreferences);
				//jRacy.jRacyPreferences = new Preferences(savedPreferences);
				
				//jRacy.systemEvents.addObserver(jRacyPreferences);
			}
			catch(Exception e)
			{
				if(e instanceof FileNotFoundException)
				{
					//There was no preference file found, therefore, just create a default preference object.
					System.out.println("No preference file present, using defaults!");
					//jRacy.clrChooser = new ColorChooser(null);
					//jRacy.jRacyPreferences = new Preferences(null);
					//jRacy.systemEvents.addObserver(jRacyPreferences);
				}
				else
				{
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
				System.out.println("Found pprof.dat ... loading");
				
				//setTitle("jRacy: " + jRacy.profilePathName);
				
				//Create a default experiment.
				Experiment exp = new Experiment("default");
				experimentManager.addExperiment(exp);
				
				//Add the experiment run for this pprof.dat file to the experiment.
				ExperimentRun expRun = null;
				String tmpString1 = null;
				String tmpString2 = null;
				String tmpString3 = null;
				
				tmpString1 = testForPprofDat.getCanonicalPath();
				tmpString2 = jRacy.experimentManager.getPathReverse(tmpString1);
				tmpString3 = "defaultRun" + " : " + tmpString2;
																	  
				expRun = new ExperimentRun();
						
				expRun.setProfilePathName(tmpString1);
				expRun.setProfilePathName(tmpString2);
				expRun.setRunName(tmpString3);
				
				exp.addExperimentRun(expRun);
				expRun.buildStaticData(true, testForPprofDat);
				
				expRun.showStaticMainWindow();
			}
			else
			{
				boolean foundSomething = false;
				
				File file = new File(".");
				Experiment exp = null;
				ExperimentRun expRun = null;
			
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
									
									//setTitle("jRacy: " + jRacy.profilePathName);
									
									//Create a default experiment.
									exp = new Experiment("default");
									experimentManager.addExperiment(exp);
									
									//Add the experiment run for this pprof.dat file to the experiment.
									String tmpString1 = null;
									String tmpString2 = null;
									String tmpString3 = null;
									
									tmpString1 = filePath;
									tmpString2 = jRacy.experimentManager.getPathReverse(tmpString1);
									tmpString3 = "defaultRun" + " : " + tmpString2;
																						  
									expRun = new ExperimentRun();
											
									expRun.setProfilePathName(tmpString1);
									expRun.setProfilePathName(tmpString2);
									expRun.setRunName(tmpString3);
									
									exp.addExperimentRun(expRun);
									expRun.buildStaticData(true, testFile);
									
									System.out.println("Found: " + newString);
									
									foundSomething = true;
									
									//expRun.showStaticMainWindow();
								}
								else{
									expRun.buildStaticData(false, testFile);
								}	
							}
						}
					}			
				}
				
				if(!foundSomething)
					System.out.println("Did not find pprof.dat!");
				else
					expRun.showStaticMainWindow();
				
				
				jRacy.experimentManager.displayExperimentListManager();
			}
			
			
			//Bring up the main window.
			//staticMainWindow = new StaticMainWindow();
			//jRacy.systemEvents.addObserver(staticMainWindow);
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
	
	//The about jRacy info. string.
	public static String getInfoString()
	{
		return new String("jRacy Version 1.2 ... The Paraducks Group!");
	}
	
	//Handles system errors.
	public static void systemError(Component inComponent, String inString)
	{	
		JOptionPane.showMessageDialog(inComponent, "jRacy Error", "Internal System Error ... Closing jRacy!", JOptionPane.ERROR_MESSAGE);
		
		System.out.println("An exception was caught at " + inString);
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
	
		//At the moment, command line arguments are optional.  Keep this here though if
		//they become mandatory.
		/*if (args.length == 0) {
                System.err.println(USAGE);
		System.exit(-1);
        }*/


		/*try{
			File file = new File(".");
			
			String filePath = file.getCanonicalPath();
			File [] list = file.listFiles();
			for(int i = 0; i < list.length; i++)
			{
				File tmpFile = (File) list[i];
				if(tmpFile != null){
					String tmpString = tmpFile.getName();
					
					if(tmpString.indexOf("MULTI__") != -1){
						System.out.println(filePath + "/" + tmpString);
					}
				}
								
			}
		}
		catch(Exception e){
			System.out.println(e);
		}*/



		int numberOfArguments = 0;
		String argument;

		while (numberOfArguments < args.length) {
           argument = args[numberOfArguments++];
           if (argument.equalsIgnoreCase("HELP")) {
                   System.err.println(USAGE);
                   System.exit(-1);
           }
           if (argument.equalsIgnoreCase("DEBUG")) {
                   jRacy.debugIsOn = true;
                   continue;
           }
		}
		
		jRacy.runtime = Runtime.getRuntime();
		
		//Start Racy.
		jRacy racy = new jRacy();
		
		if(debugIsOn)
		{
			//Create and start the a timer, and then add racy to it.
			javax.swing.Timer jTimer = new javax.swing.Timer(8000, racy);
			jTimer.start();
		}
	}
	
}