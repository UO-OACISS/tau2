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
	//End - Some system wide state variables.
	//**********
	
	//**********
	//Start or define all the persistant objects.
	static SystemEvents systemEvents = new SystemEvents();
	static SavedPreferences savedPreferences = new SavedPreferences();
	static Preferences jRacyPreferences = null;
	static StaticSystemData staticSystemData = null;
	static ColorChooser clrChooser;
	static HelpWindow helpWindow = new HelpWindow();
	static StaticMainWindow staticMainWindow = null;
	//End start of persistant objects.
	
	//Useful in the system.
	static Runtime runtime;
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
			jRacy.systemEvents.addObserver(helpWindow);
			
			//Try and load the Racy preference file ... racyPreferences.dat
			try
			{
				FileInputStream savedPreferenceFIS = new FileInputStream("jRacyPreferences.dat");
				
				//If here, means that no exception was thrown, and there is a preference file present.
				//Create ObjectInputStream and try to read it in.
				ObjectInputStream inSavedPreferencesOIS = new ObjectInputStream(savedPreferenceFIS);
				jRacy.savedPreferences = (SavedPreferences) inSavedPreferencesOIS.readObject();
				
				jRacy.clrChooser = new ColorChooser(savedPreferences);
				jRacy.jRacyPreferences = new Preferences(savedPreferences);
				
				jRacy.systemEvents.addObserver(jRacyPreferences);
			}
			catch(Exception e)
			{
				if(e instanceof FileNotFoundException)
				{
					//There was no preference file found, therefore, just create a default preference object.
					System.out.println("No preference file present, using defaults!");
					jRacy.clrChooser = new ColorChooser(null);
					jRacy.jRacyPreferences = new Preferences(null);
					jRacy.systemEvents.addObserver(jRacyPreferences);
				}
				else
				{
					//Print some kind of error message, and quit the system.
					System.out.println("There was an internal error whilst trying to read the Racy preference");
					System.out.println("file.  Please delete this file, or replace it with a valid one!");
					System.out.println("Note: Deleting the file will cause Racy to restore the default preferences");
				}
			}
			
			//Bring up the main window.
			staticMainWindow = new StaticMainWindow();
			jRacy.systemEvents.addObserver(staticMainWindow);
			staticMainWindow.setVisible(true);
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
		return new String("jRacy Version 1.0 ... The Paraducks Group!");
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