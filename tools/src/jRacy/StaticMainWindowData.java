/*	
	StaticMainWindowData.java


	Title:			jRacy
	Author:			Robert Bell
	Description:	This object file controls the ordering of all data relating to the
					StaticMainWindow, and its subwindows.  It obtains its data from the
					global StaticSystemData object.  Thus, all requests for data by the
					StaticMainWindow, and its subwindows, are handled by this object only.	
*/

package jRacy;

import java.util.*;
import java.lang.*;

public class StaticMainWindowData
{

	//A variable to check to make sure that the user has loaded system data.
	boolean dataLoaded = false;
	 		
	//The sorted system data lists.  It makes more sense to sort at the beginning.
	//F:Function;N:Name;M:Millisecond;D:Descending;A:Ascending;E:Exclusive;I:Inclusive;Id:Function ID
	
	private Vector sMWGeneralData = new Vector();
	private Vector sMWMeanData = new Vector();
	
	private String currentlySortedAsMeanTotalStatWindow;
	
	public StaticMainWindowData()
	{
	}
	
	//Setting and getting the loaded system data boolean.
	public void setDataLoaded(boolean inBoolean)
	{
		dataLoaded = inBoolean;
	}
	
	public boolean isDataLoaded()
	{
		return dataLoaded;
	}
	
	
	
	//********************************
	//
	//Functions that create the StaticMainWindowData lists.
	//
	//********************************
	
	public void buildStaticMainWindowDataLists()
	{
		buildSMWGeneralData();
		buildSMWMeanList();
	}
	
	
	private void buildSMWGeneralData()
	{
		
		//********************************
		//This function builds the server, context, and thread list for
		//the default static main window displays.
		//
		//
		//Note:
		//The extensions of the global server, global context, and global thread
		//objects are specific to this display structure.  Unless you are happy
		//with how the drawing data is stored in these extensions, you should
		//use your own custom ones.
		//********************************
		
		
		//Copy data to the appropriate list with the appropriate sorting.
		GlobalServer tmpGlobalServer;
		GlobalContext tmpGlobalContext;
		GlobalThread tmpGlobalThread;
		GlobalThreadDataElement tmpGlobalThreadDataElement;
		
		SMWServer tmpSMWServer;
		SMWContext tmpSMWContext;
		SMWThread tmpSMWThread;
		SMWThreadDataElement tmpSMWThreadDataElement;
		SMWThreadDataElement tmpSMWUserThreadDataElement;
		
		
		Vector tmpContextList;
		Vector tmpThreadList;
		Vector tmpThreadDataList;
		
		
		//Get a reference to the global data.
		Vector tmpVector = jRacy.staticSystemData.getStaticServerList();
		
		//Clear the sMWGeneralData list for safety.
		sMWGeneralData.removeAllElements();
		
		for(Enumeration e1 = tmpVector.elements(); e1.hasMoreElements() ;)
		{
			tmpGlobalServer = (GlobalServer) e1.nextElement();
			//Create a new sMWServer object and set the name properly.
			tmpSMWServer = new SMWServer();
			//Add the server.
			sMWGeneralData.addElement(tmpSMWServer);
			
			//Enter the context loop for this server.
			tmpContextList = tmpGlobalServer.getContextList();
				
			for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
			{
				tmpGlobalContext = (GlobalContext) e2.nextElement();
				
				//Create a new context object and set the name properly.
				tmpSMWContext = new SMWContext();
				//Add to the server.
				tmpSMWServer.addContext(tmpSMWContext);
				
					
				//Enter the thread loop for this context.
				tmpThreadList = tmpGlobalContext.getThreadList();
				for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
				{
					tmpGlobalThread = (GlobalThread) e3.nextElement();
					
					//Create a new thread object.
					tmpSMWThread = new SMWThread();
					//Add to the context.
					tmpSMWContext.addThread(tmpSMWThread);
					
					//Now enter the thread loop for this thread.
					tmpThreadDataList = tmpGlobalThread.getThreadDataList();
					for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
					{
						tmpGlobalThreadDataElement = (GlobalThreadDataElement) e4.nextElement();
						
						//Only want to add an element if this function existed on this thread.
						//Check for this.
						if(tmpGlobalThreadDataElement != null)
						{
							//Create a new thread data object.
							tmpSMWThreadDataElement = new SMWThreadDataElement(tmpGlobalThreadDataElement);
							
							tmpSMWThreadDataElement.setFunctionID(tmpGlobalThreadDataElement.getFunctionID());
							
							//Add to the thread data object.
							tmpSMWThread.addThreadDataElement(tmpSMWThreadDataElement);
						}
					}
				}
			}
		}
		
		//Set the dataLoaded boolean to true.
		dataLoaded = true;
	}
	
	private void buildSMWMeanList()
	{
		//First, grab the global mapping element list.
		GlobalMapping tmpGlobalMapping = jRacy.staticSystemData.getGlobalMapping();
		
		Vector tmpVector = tmpGlobalMapping.getNameIDMapping();
		
		//Clear the sMWMeanData for safety.
		sMWMeanData.removeAllElements();
		
		//Now cycle through, building our new list.
		for(Enumeration e1 = tmpVector.elements(); e1.hasMoreElements() ;)
		{
			GlobalMappingElement tmpGME = (GlobalMappingElement) e1.nextElement();
			
			if(tmpGME.getMeanValuesSet())
			{
				//Create a new mean data element.
				SMWMeanDataElement tmpSMWMeanDataElement = new SMWMeanDataElement();

				tmpSMWMeanDataElement.setFunctionID(tmpGME.getGlobalID());
				tmpSMWMeanDataElement.setValue(tmpGME.getMeanExclusiveValue());
				tmpSMWMeanDataElement.setSortByValue();
				tmpSMWMeanDataElement.setSortByReverse(true);
				
				sMWMeanData.addElement(tmpSMWMeanDataElement);
			}		
		}
		
		//Now sort it.
		Collections.sort(sMWMeanData);
	
	}
	
	//********************************
	//
	//End - Functions that create the StaticMainWindowData lists.
	//
	//********************************
	
	
	//********************************
	//
	//Functions that return various sorted version of the data lists.
	//
	//********************************
	public Vector getSMWThreadData(int inServer, int inContext, int inThread, String inString)
	{
		//Return a copy of the requested data, sorted in the appropriate manner.
		
		//First, obtain the appropriate server.
		SMWServer tmpSMWServer = (SMWServer) sMWGeneralData.elementAt(inServer);
		Vector tmpContextList = tmpSMWServer.getContextList();
		SMWContext tmpSMWContext = (SMWContext) tmpContextList.elementAt(inContext);
		Vector tmpThreadList = tmpSMWContext.getThreadList();
		SMWThread tmpSMWThread = (SMWThread) tmpThreadList.elementAt(inThread);
		Vector tmpThreadDataList = tmpSMWThread.getThreadDataList();
		
		//Ok, now that I have the appropriate thread, copy it and then sort the copy in the appropriate manner.
		Vector tmpVector = new Vector();
		SMWThreadDataElement tmpSMWThreadDataElement;
		SMWThreadDataElement tmpSMWThreadDataElementCopy;
		
			
		if(inString.equals("FIdDE"))
		{
			
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
				
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getExclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByFunctionID();
				tmpSMWThreadDataElementCopy.setSortByReverse(true);
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		else if(inString.equals("FIdDI"))
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
				
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getInclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByFunctionID();
				tmpSMWThreadDataElementCopy.setSortByReverse(true);
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		else if(inString.equals("FIdAE"))
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
				
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getExclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByFunctionID();
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		else if(inString.equals("FIdAI"))
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
				
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getInclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByFunctionID();
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		else if(inString.equals("NDE"))
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());

				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getExclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByName();
				tmpSMWThreadDataElementCopy.setSortByReverse(true);
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);			
			}
		}
		else if(inString.equals("NDI"))
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
				
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getInclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByName();
				tmpSMWThreadDataElementCopy.setSortByReverse(true);
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		else if(inString.equals("NAE"))
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
				
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getExclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByName();
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		else if(inString.equals("NAI"))
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
			
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getInclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByName();
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		else if(inString.equals("MDE"))
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
			
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getExclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByValue();
				tmpSMWThreadDataElementCopy.setSortByReverse(true);
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		else if((inString.equals("MDI")))
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
			
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getInclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByValue();
				tmpSMWThreadDataElementCopy.setSortByReverse(true);
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		else if((inString.equals("MAE")))
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
			
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getExclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByValue();
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		else
		{
			for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
				//Create a new thread data object.
				tmpSMWThreadDataElementCopy = new SMWThreadDataElement(tmpSMWThreadDataElement.getGTDE());
			
			
				tmpSMWThreadDataElementCopy.setFunctionID(tmpSMWThreadDataElement.getFunctionID());
				tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getInclusiveValue());
				tmpSMWThreadDataElementCopy.setSortByValue();
				
				tmpVector.addElement(tmpSMWThreadDataElementCopy);
			}
		}
		
		Collections.sort(tmpVector);
		return tmpVector;
	}
	
	public Vector getSMWGeneralData(String inString)
	{	
		SMWServer tmpSMWServer;
		SMWContext tmpSMWContext;
		SMWThread tmpSMWThread;
		SMWThreadDataElement tmpSMWThreadDataElement;
		
		Vector tmpContextList;
		Vector tmpThreadList;
		Vector tmpThreadDataList;
		
		if(inString == null)
		{
			//Just return the current list as the caller does not care as to the order.
			return sMWGeneralData;
		}
		
		//In this function, the entire data structure is sorted according to
		//the specified string.
		if(inString.equals("FIdDE"))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						 tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getExclusiveValue());
							tmpSMWThreadDataElement.setSortByFunctionID();
							tmpSMWThreadDataElement.setSortByReverse(true);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
					
			return sMWGeneralData;
		}
		else if(inString.equals("FIdDI"))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						 tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getInclusiveValue());
							tmpSMWThreadDataElement.setSortByFunctionID();
							tmpSMWThreadDataElement.setSortByReverse(true);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
				
			return sMWGeneralData;
		}
		else if(inString.equals("FIdAE"))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						 tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getExclusiveValue());
							tmpSMWThreadDataElement.setSortByFunctionID();
							tmpSMWThreadDataElement.setSortByReverse(false);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
			
			return sMWGeneralData;
		}
		else if(inString.equals("FIdAI"))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						 tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getInclusiveValue());
							tmpSMWThreadDataElement.setSortByFunctionID();
							tmpSMWThreadDataElement.setSortByReverse(false);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
			
			return sMWGeneralData;
		}
		else if(inString.equals("NDE"))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getExclusiveValue());
							tmpSMWThreadDataElement.setSortByName();
							tmpSMWThreadDataElement.setSortByReverse(true);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
				
			return sMWGeneralData;
		}
		else if(inString.equals("NDI"))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						 tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getInclusiveValue());
							tmpSMWThreadDataElement.setSortByName();
							tmpSMWThreadDataElement.setSortByReverse(true);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
			
			return sMWGeneralData;
		}
		else if(inString.equals("NAE"))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						 tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getExclusiveValue());
							tmpSMWThreadDataElement.setSortByName();
							tmpSMWThreadDataElement.setSortByReverse(false);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
			
			return sMWGeneralData;
		}
		else if(inString.equals("NAI"))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						 tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getInclusiveValue());
							tmpSMWThreadDataElement.setSortByName();
							tmpSMWThreadDataElement.setSortByReverse(false);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
				
			return sMWGeneralData;
		}
		else if(inString.equals("MDE"))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						 tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getExclusiveValue());
							tmpSMWThreadDataElement.setSortByValue();
							tmpSMWThreadDataElement.setSortByReverse(true);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
				
			return sMWGeneralData;
		}
		else if((inString.equals("MDI")))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						 tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getInclusiveValue());
							tmpSMWThreadDataElement.setSortByValue();
							tmpSMWThreadDataElement.setSortByReverse(true);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
				
			return sMWGeneralData;
		}
		else if((inString.equals("MAE")))
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						 tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getExclusiveValue());
							tmpSMWThreadDataElement.setSortByValue();
							tmpSMWThreadDataElement.setSortByReverse(false);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
			
			return sMWGeneralData;
		}
		else
		{
			for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;)
			{
				tmpSMWServer = (SMWServer) e1.nextElement();
				
				//Enter the context loop for this server.
				tmpContextList = tmpSMWServer.getContextList();
					
				for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
				{
					tmpSMWContext = (SMWContext) e2.nextElement();
						
					//Enter the thread loop for this context.
					tmpThreadList = tmpSMWContext.getThreadList();
					
					for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
					{
						tmpSMWThread = (SMWThread) e3.nextElement();
						 
						//Now enter the thread loop for this thread.
						tmpThreadDataList = tmpSMWThread.getThreadDataList();
						
						for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
						{
							tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
							tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getInclusiveValue());
							tmpSMWThreadDataElement.setSortByValue();
							tmpSMWThreadDataElement.setSortByReverse(false);
						}
						
						//Now, sort this thread list.
						Collections.sort(tmpThreadDataList);
					}
				}
			}
			
			return sMWGeneralData;
		}
	}
	
	
	public Vector getSMWMeanData(String inString)
	{	
		SMWServer tmpSMWServer;
		SMWContext tmpSMWContext;
		SMWThread tmpSMWThread;
		SMWThreadDataElement tmpSMWThreadDataElement;
		
		Vector tmpContextList;
		Vector tmpThreadList;
		Vector tmpThreadDataList;
		
		if(inString == null)
		{
			//Just return the current list as the caller does not care as to the order.
			return sMWMeanData;
		}
		
		//In this function, the entire data structure is sorted according to
		//the specified string.
		if(inString.equals("FIdDE"))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanExclusiveValue());
				tmpSMWMeanDataElement.setSortByFunctionID();
				tmpSMWMeanDataElement.setSortByReverse(true);
				
			}
			
			Collections.sort(sMWMeanData);
			
			currentlySortedAsMeanTotalStatWindow = new String("FIdDE");
			
			return sMWMeanData;
			
		}
		else if(inString.equals("FIdDI"))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanInclusiveValue());
				tmpSMWMeanDataElement.setSortByFunctionID();
				tmpSMWMeanDataElement.setSortByReverse(true);
				
			}
		
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
		else if(inString.equals("FIdAE"))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanExclusiveValue());
				tmpSMWMeanDataElement.setSortByFunctionID();
				tmpSMWMeanDataElement.setSortByReverse(false);
				
			}
			
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
		else if(inString.equals("FIdAI"))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanInclusiveValue());
				tmpSMWMeanDataElement.setSortByFunctionID();
				tmpSMWMeanDataElement.setSortByReverse(false);
				
			}
			
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
		else if(inString.equals("NDE"))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanExclusiveValue());
				tmpSMWMeanDataElement.setSortByName();
				tmpSMWMeanDataElement.setSortByReverse(true);
				
			}
			
			
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
		else if(inString.equals("NDI"))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanInclusiveValue());
				tmpSMWMeanDataElement.setSortByName();
				tmpSMWMeanDataElement.setSortByReverse(true);
				
			}
			
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
		else if(inString.equals("NAE"))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanExclusiveValue());
				tmpSMWMeanDataElement.setSortByName();
				tmpSMWMeanDataElement.setSortByReverse(false);
				
			}
			
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
		else if(inString.equals("NAI"))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanInclusiveValue());
				tmpSMWMeanDataElement.setSortByName();
				tmpSMWMeanDataElement.setSortByReverse(false);
				
			}
			
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
		else if(inString.equals("MDE"))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanExclusiveValue());
				tmpSMWMeanDataElement.setSortByValue();
				tmpSMWMeanDataElement.setSortByReverse(true);
				
			}
			
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
		else if((inString.equals("MDI")))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanInclusiveValue());
				tmpSMWMeanDataElement.setSortByValue();
				tmpSMWMeanDataElement.setSortByReverse(true);
				
			}
			
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
		else if((inString.equals("MAE")))
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanExclusiveValue());
				tmpSMWMeanDataElement.setSortByValue();
				tmpSMWMeanDataElement.setSortByReverse(false);
				
			}
			
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
		else
		{
			for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
			{
				SMWMeanDataElement tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
				//Set the sorting method.
				tmpSMWMeanDataElement.setValue(tmpSMWMeanDataElement.getMeanInclusiveValue());
				tmpSMWMeanDataElement.setSortByValue();
				tmpSMWMeanDataElement.setSortByReverse(false);
				
			}
			
			Collections.sort(sMWMeanData);
			return sMWMeanData;
		}
	}
	
	//********************************
	//
	//End - Functions that return various sorted version of the data lists.
	//
	//********************************
}