/* 
	SMWContext.java

	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

public class SMWContext
{
	//Constructors.	
	public SMWContext()
	{
		threadList = new Vector();
		yDrawCoord = -1;
	}
	
	public void addThread(SMWThread inSMWThread)
	{
		//Add the thread to the end of the list ... the default
		//for addElement in a Vector.
		threadList.addElement(inSMWThread);
	}
	
	public Vector getThreadList()
	{
		return threadList;
	}
		
	public void setYDrawCoord(int inYDrawCoord)
	{
		yDrawCoord = inYDrawCoord;
	}
	
	public int getYDrawCoord()
	{
		return yDrawCoord;
	}
	
	//Instance data.
	Vector threadList;
	//To aid with drawing searches.
	int yDrawCoord;
	
}