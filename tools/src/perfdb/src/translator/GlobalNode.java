package translator;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Vector;


public class GlobalNode implements Serializable 
{

       	//Instance data.
	int nodeName;
	Vector contextList;
	int numberOfContexts;
 
	//Constructor.
	public GlobalNode()
	{
		nodeName = 0;
		contextList = null;
		numberOfContexts = 0;
	}
	
	public GlobalNode(int inNodeName)
	{
		nodeName = inNodeName;
		contextList = new Vector();
		numberOfContexts = 0;
	}
	
	//Rest of the public functions.
	public void setNodeName(int inNodeName)
	{
		nodeName = inNodeName;
	}
	
	public int getNodeName()
	{
		return nodeName;
	}
	
	public void addContext(GlobalContext inGlobalContextObject)
	{
		//Keeping track of the number of contexts on this server.
		numberOfContexts++;

		//Now add the context to the end of the list ... the default
		//for addElement in a Vector.
		contextList.addElement(inGlobalContextObject);
	}
	
	public boolean isContextPresent(int inContextName)
	{
		GlobalContext contextObject;
		int tmpString;
		
		for(Enumeration e = contextList.elements(); e.hasMoreElements() ;)
		{
			contextObject = (GlobalContext) e.nextElement();
			tmpString = contextObject.getContextName();
			if(inContextName == tmpString)
				return true;
		}
		//If here, it means that the context name was not in the list.
		return false;
	}
	
	public Vector getContextList() 
	{
		return contextList;
	}	
	
}
