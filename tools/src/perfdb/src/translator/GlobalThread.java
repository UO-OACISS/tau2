package translator;

import java.io.Serializable;
import java.util.Vector;

public class GlobalThread implements Serializable 
{

        	
    //Instance data.
    Vector threadDataList;
    int threadName; 
    Vector userThreadDataList;	

    //Constructor.
    public GlobalThread()
    {
	threadName = 0;
	threadDataList = new Vector();
	userThreadDataList = new Vector();
    }
	
    public GlobalThread(int inThreadName)
    {
	threadName = inThreadName;
	threadDataList = new Vector();
	userThreadDataList = new Vector();
    }	

    //The following function adds a thread data element to
    //the threadDataList
    void addThreadDataElement(GlobalThreadDataElement inGTDE)
    {
	threadDataList.addElement(inGTDE);
    }
	
    void addThreadDataElement(GlobalThreadDataElement inGTDE, int inPosition)
    {
	threadDataList.setElementAt(inGTDE, inPosition);
    }
	
    public void setThreadName(int inThreadName)
    {
	threadName = inThreadName;
    }
	
    public int getThreadName()
    {
	return threadName;
    }
	
    Vector getThreadDataList()
    {
	return threadDataList;
    }

    // The following function adds a thread data element to the userThreadDataList
    void addUserThreadDataElement(GlobalThreadDataElement inGTDE)
    {
	userThreadDataList.addElement(inGTDE);
    }
	
    Vector getUserThreadDataList()
    {
	return userThreadDataList;
    }
	
}
