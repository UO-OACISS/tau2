/* 
	SMWThread.java

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

public class SMWThread
{
	//Constructor.
	public SMWThread()
	{
		threadDataList = new Vector();
		userThreadDataList = new Vector();
	}
	
	//The following function adds a thread data element to
	//the threadDataList
	void addThreadDataElement(SMWThreadDataElement inSMWTDE)
	{
		threadDataList.addElement(inSMWTDE);
	}
	
	//The following function adds a thread data element to
	//the userThreadDataList
	void addUserThreadDataElement(SMWThreadDataElement inSMWTDE)
	{
		userThreadDataList.addElement(inSMWTDE);
	}
	
	Vector getThreadDataList()
	{
		return threadDataList;
	}
	
	Vector getUserThreadDataList()
	{
		return userThreadDataList;
	}
	
	//Rest of the public functions.
	public void setYDrawCoord(int inYDrawCoord)
	{
		yDrawCoord = inYDrawCoord;
	}
	
	public int getYDrawCoord()
	{
		return yDrawCoord;
	}
		
	//Instance data.
	Vector threadDataList;
	Vector userThreadDataList;
	//To aid with drawing searches.
	int yDrawCoord;
	
}