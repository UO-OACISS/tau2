package translator;

import java.util.*;
import java.io.*;


public class GlobalThreadDataElement implements Serializable 
{
        		
	//Set if function exists on this thread.
	boolean functionExists = false;
	
	//Function ID
	int functionID;
        String functionName;
        String functionGroup = null;
	
	//Named data values.
	private double inclValue;
	private double exclValue;
	private double inclPercValue;
	private double exclPercValue;
        private double call;
        private double subrs;
        private double inclPCall;
			
    // User event section.
	private String userEventName;
	private int userEventID;
	private int userEventNumberValue;
	private double userEventMinValue;
	private double userEventMaxValue;
	private double userEventMeanValue;
        private double userEventStdDevValue;
	String userEventStatString;

	//Constructor.
	public GlobalThreadDataElement()
	{	
				
		inclValue = 0;
		exclValue = 0;
		inclPercValue = 0;
		exclPercValue = 0;
		call = 0;
		subrs = 0;
		inclPCall = 0;
		
		functionID = -1;

		
		userEventName = null;
		userEventNumberValue = 0;
		userEventMinValue = 0;
		userEventMaxValue = 0;
		userEventMeanValue = 0;
		userEventStatString = null;
	}
	
	//Rest of the public functions.
	/*public String getFunctionName()
	{
		tmpGME = (GlobalMappingElement) globalMappingReference.getGlobalMappingElement(functionID);
		
		return tmpGME.getFunctionName();
		}*/
	
	public void setFunctionID(int inFunctionID)
	{
		functionID = inFunctionID;
	}
	
	public void setFunctionExists()
	{
		functionExists = true;
	}
	
	public boolean getFunctionExists()
	{
		return functionExists;
	}
	
	public int getFunctionID()
	{
		return functionID;
	}

        public void setFunctionName(String inFunctionName)
	{
		functionName = inFunctionName;
	}

        public String getFunctionName()
	{
		return functionName;
	}

        public void setFunctionGroup(String funcGroup){
	    functionGroup = funcGroup;
        }

    public String getFunctionGroup(){
	if (functionGroup == null)
	    return "";
	else return functionGroup;
    }
	
	public void setInclValue(double inInclValue)
	{
		inclValue = inInclValue;
	}
	
	public double getInclValue()
	{
		return inclValue;
	}
	
	public void setExclValue(double inExclValue)
	{
		exclValue = inExclValue;
	}
	
	public double getExclValue()
	{
		return exclValue;
	}
	
	
	public void setInclPercValue(double inInclPercValue)
	{
		inclPercValue = inInclPercValue;
	}
	
	public double getInclPercValue()
	{
		return inclPercValue;
	}
	
	public void setExclPercValue(double inExclPercValue)
	{
		exclPercValue = inExclPercValue;
	}
	
	public double getExclPercValue()
	{
		return exclPercValue;
	}
	
    /*	public void setTStatString(String inString)
	{
		tStatString = inString;
	}
	
	public String getTStatString()
	{
		return tStatString;
		}*/

	public void setCall(double callnum)
	{
		call = callnum;
	}
	
	public double getCall()
	{
		return call;
	}

        public void setSubrs(double subrsnum)
	{
		subrs = subrsnum;
	}
	
	public double getSubrs()
	{
		return subrs;
	}

        public void setInclPCall(double inclpcallnum)
	{
		inclPCall = inclpcallnum;
	}
	
	public double getInclPCall()
	{
		return inclPCall;
	}
    
	
    // User event interface.
	public void setUserEventName(String inUserEventName)
	{
		userEventName = inUserEventName;
	}
	
	public String getUserEventName()
	{
		return userEventName;
	}
	
	public void setUserEventID(int inUserEventID)
	{
		userEventID = inUserEventID;
	}
	
	public int getUserEventID()
	{
		return userEventID;
	}
	
	public void setUserEventNumberValue(int inUserEventNumberValue)
	{
		userEventNumberValue = inUserEventNumberValue;
	}
	
	public int getUserEventNumberValue()
	{
		return userEventNumberValue;
	}
	
	public void setUserEventMinValue(double inUserEventMinValue)
	{
		userEventMinValue = inUserEventMinValue;
	}
	
	public double getUserEventMinValue()
	{
		return userEventMinValue;
	}
	
	public void setUserEventMaxValue(double inUserEventMaxValue)
	{
		userEventMaxValue = inUserEventMaxValue;
	}
	
	public double getUserEventMaxValue()
	{
		return userEventMaxValue;
	}
	
	public void setUserEventMeanValue(double inUserEventMeanValue)
	{
		userEventMeanValue = inUserEventMeanValue;
	}
	
	public double getUserEventMeanValue()
	{
		return userEventMeanValue;
	}
	
        public void setUserEventStdDevValue(double inUserEventStdDevValue)
	{
		userEventStdDevValue = inUserEventStdDevValue;
	}
	
	public double getUserEventStdDevValue()
	{
		return userEventStdDevValue;
	}

	public void setUserEventStatString(String inString)
	{
		userEventStatString = inString;
	}
	
	public String getUserEventStatString()
	{
		return userEventStatString;
	}


}




