/* 
	GlobalContext.java

	Title:			Racy
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


//Instance of this class are really only meant for the ServCon class to
//keep track of the servers and contexts.
//Real thread work goes on in the derived class, ContextObject.

public class GlobalContext implements Serializable 
{
	//Constructors.
	public GlobalContext()
	{
		contextName = null;
		threadList = new Vector();
		numberOfThreads = 0;
	}
	
	public GlobalContext(String inContextName)
	{
		contextName = inContextName;
		threadList = new Vector();
		numberOfThreads = 0;
	}
	
	//Rest of the public functions.
	public void setContextName(String inContextName)
	{
		contextName = inContextName;
	}
	
	public String getContextName()
	{
		return contextName;
	}
	
	public void addThread(GlobalThread inGlobalThread)
	{
		//When a thread is added, since threads do not vanish
		//from the TAU record, we can just add to the end of the list.
		//The threads are thus ordered for this context correctly.
		
		//Keeping track of the number of threads in this context.
		numberOfThreads++;
		//Now add the thread to the end of the list ... the default
		//for addElement in a Vector.
		threadList.addElement(inGlobalThread);
	}
	
	public Vector getThreadList()
	{
		return threadList;
	}
	
	//Instance data.
	Vector threadList;
	int numberOfThreads;
	
	String contextName;
	
}