/* 
	GlobalMapping.java

	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;


public class GlobalMapping implements WindowListener, Serializable 
{
	//Constructors.
	public GlobalMapping()
	{
		nameIDMapping = new Vector();
		numberOfGlobalFunctions = 0;
	}
	
	public void addGlobalFunction(String inFunctionName)
	{
		//Just adds to the end of the list.  Its position becomes
		//the value of its function ID.
		GlobalMappingElement tmpGME = new GlobalMappingElement();
		tmpGME.setFunctionName(inFunctionName);
		tmpGME.setGlobalID(numberOfGlobalFunctions);
		tmpGME.setGenericColor(jRacy.clrChooser.getColorInLocation(numberOfGlobalFunctions % (jRacy.clrChooser.getNumberOfColors())));
		nameIDMapping.addElement(tmpGME);
		
		//Update the number of global functions present.  (Example ... first time
		//round, numberOfGlobalFunctions = 0, and thus the new function name gets an
		//ID of 0.  The numberOfGlobalFunctions is now updated to 1 and thus returns
		//the correct amount should it be asked for.
		numberOfGlobalFunctions++;
	}
	
	public boolean setFunctionNameAt(String inFunctionName, int inPosition)
	{
		//First check to make sure that inPosition is not greater than the number of
		//functions present (minus one of course for vector numbering).
		if(inPosition > (this.getNumberOfFunctions() - 1))
		{
			return false;
		}
		
		//If here, the size of inPosition is ok.
		//Therefore grab the element at that position.
		GlobalMappingElement tmpGME = (GlobalMappingElement) nameIDMapping.elementAt(inPosition);
		
		//Set the name.
		tmpGME.setFunctionName(inFunctionName);
		
		//Successful ... return true.
		return true;
	}
	
	public boolean setMeanExclusiveValueAt(double inMeanExclusiveValue, int inPosition)
	{
		//First check to make sure that inPosition is not greater than the number of
		//functions present (minus one of course for vector numbering).
		if(inPosition > (this.getNumberOfFunctions() - 1))
		{
			return false;
		}
		
		//If here, the size of inPosition is ok.
		//Therefore grab the element at that position.
		GlobalMappingElement tmpGME = (GlobalMappingElement) nameIDMapping.elementAt(inPosition);
		
		//Set the mean value.
		tmpGME.setMeanExclusiveValue(inMeanExclusiveValue);
		
		//Successful ... return true.
		return true;
	}
	
	public boolean setMeanInclusiveValueAt(double inMeanInclusiveValue, int inPosition)
	{
		//First check to make sure that inPosition is not greater than the number of
		//functions present (minus one of course for vector numbering).
		if(inPosition > (this.getNumberOfFunctions() - 1))
		{
			return false;
		}
		
		//If here, the size of inPosition is ok.
		//Therefore grab the element at that position.
		GlobalMappingElement tmpGME = (GlobalMappingElement) nameIDMapping.elementAt(inPosition);
		
		//Set the mean value.
		tmpGME.setMeanInclusiveValue(inMeanInclusiveValue);
		
		//Successful ... return true.
		return true;
	}
	
	public boolean setTotalExclusiveValueAt(double inTotalExclusiveValue, int inPosition)
	{
		//First check to make sure that inPosition is not greater than the number of
		//functions present (minus one of course for vector numbering).
		if(inPosition > (this.getNumberOfFunctions() - 1))
		{
			return false;
		}
		
		//If here, the size of inPosition is ok.
		//Therefore grab the element at that position.
		GlobalMappingElement tmpGME = (GlobalMappingElement) nameIDMapping.elementAt(inPosition);
		
		//Set the mean value.
		tmpGME.setTotalExclusiveValue(inTotalExclusiveValue);
		
		//Successful ... return true.
		return true;
	}
	
	public boolean setTotalInclusiveValueAt(double inTotalInclusiveValue, int inPosition)
	{
		//First check to make sure that inPosition is not greater than the number of
		//functions present (minus one of course for vector numbering).
		if(inPosition > (this.getNumberOfFunctions() - 1))
		{
			return false;
		}
		
		//If here, the size of inPosition is ok.
		//Therefore grab the element at that position.
		GlobalMappingElement tmpGME = (GlobalMappingElement) nameIDMapping.elementAt(inPosition);
		
		//Set the mean value.
		tmpGME.setTotalInclusiveValue(inTotalInclusiveValue);
		
		//Successful ... return true.
		return true;
	}
		
	
	public boolean isFunctionPresent(String inFunctionName)
	{
		GlobalMappingElement tmpElement;
		String tmpString;
		
		for(Enumeration e = nameIDMapping.elements(); e.hasMoreElements() ;)
		{
			tmpElement = (GlobalMappingElement) e.nextElement();
			tmpString = tmpElement.getFunctionName();
			if(inFunctionName.equals(tmpString))
				return true;
		}
		
		//If here, it means that the function was not found.
		return false;
	}
	
	public int getNumberOfFunctions()
	{
		return numberOfGlobalFunctions;
	}
	
	public GlobalMappingElement getGlobalMappingElement(int functionID)
	{
		//Note that by default the elments in nameIDMapping are in functionID order.
		
		//First check to make sure that functionID is not greater than the number of
		//functions present (minus one of course for vector numbering).
		if(functionID > (this.getNumberOfFunctions() - 1))
		{
			return null;
		}
		
		//We are ok, therefore, grab the element at that position.
		return (GlobalMappingElement) nameIDMapping.elementAt(functionID);
	}
	
	public int getFunctionId(String inFunctionName)
	{
		//Cycle through the list to obtain the function id.  Return -1
		//if we cannot find the name.
		
		int count = 0;
		GlobalMappingElement tmpGlobalMappingElement = null;
		for(Enumeration e1 = nameIDMapping.elements(); e1.hasMoreElements() ;)
		{
			tmpGlobalMappingElement = (GlobalMappingElement) e1.nextElement();
			if((tmpGlobalMappingElement.getFunctionName()).equals(inFunctionName))
				return count;
				
			count++;
		}
		
		//If here,  means that we did not find the function name.
		return -1;
	}
	
	public GlobalMappingElement getGlobalMappingElement(String inFunctionName)
	{
		//Cycle through the list to obtain the function id.  Return null
		//if we cannot find the name.
		
		GlobalMappingElement tmpGlobalMappingElement = null;
		for(Enumeration e1 = nameIDMapping.elements(); e1.hasMoreElements() ;)
		{
			tmpGlobalMappingElement = (GlobalMappingElement) e1.nextElement();
			if((tmpGlobalMappingElement.getFunctionName()).equals(inFunctionName))
				return tmpGlobalMappingElement;
		}
		
		//If here,  means that we did not find the function name.
		return null;
	}
	
	public Vector getNameIDMapping()
	{
		return nameIDMapping;
	}
	
	public void updateGenericColors()
	{
		for(Enumeration e = nameIDMapping.elements(); e.hasMoreElements() ;)
		{
			
			GlobalMappingElement tmpGME = (GlobalMappingElement) e.nextElement();
			int functionID = tmpGME.getGlobalID();
			tmpGME.setGenericColor(jRacy.clrChooser.getColorInLocation(functionID % (jRacy.clrChooser.getNumberOfColors())));
		}
	}
	
	public void displayFunctionLedger()
	{
		if(!FunctionLedgerWindowShowing)
		{
			//Bring up the Servers and Contexts frame.
			funLedgerWindow = new FunctionLedgerWindow(nameIDMapping);
			//Add the main window as a listener as it needs
			//to know when this window closes.
			funLedgerWindow.addWindowListener(this);
			jRacy.systemEvents.addObserver(funLedgerWindow);
			funLedgerWindow.show();
			FunctionLedgerWindowShowing = true;
		}
		else
		{
			//Just bring it to the foreground.
			funLedgerWindow.show();
		}
	}
	
	public void closeFunctionLedger()
	{
		jRacy.systemEvents.deleteObserver(funLedgerWindow); 
		
		funLedgerWindow.setVisible(false);
		funLedgerWindow.dispose();
	
		FunctionLedgerWindowShowing = false;
	}
	//Window Listener code.
	public void windowClosed(WindowEvent winevt){}
	public void windowIconified(WindowEvent winevt){}
	public void windowOpened(WindowEvent winevt){}
	public void windowClosing(WindowEvent winevt)
	{
		if(winevt.getSource() == funLedgerWindow)
		{
			FunctionLedgerWindowShowing = false;
		}
	}
	
	public void windowDeiconified(WindowEvent winevt){}
	public void windowActivated(WindowEvent winevt){}
	public void windowDeactivated(WindowEvent winevt){}
	
	
	//Instance element.
	Vector nameIDMapping;	//Elements in this vector for the GlobalMapping class
							//will be GlobalMappingElements.
	int numberOfGlobalFunctions;
	
	private FunctionLedgerWindow funLedgerWindow;
	private boolean FunctionLedgerWindowShowing = false;
}