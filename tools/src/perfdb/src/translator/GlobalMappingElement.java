package translator;

import java.util.*;
import java.io.*;


public class GlobalMappingElement implements Serializable 
{
 
	//Global Mapping reference.
	String functionName;
	int globalID;			//Global ID for this function name.
        String functionGroup=null;
	
	double meanExclusiveValue;
	double totalExclusiveValue;
	double meanExclusivePercentValue;
	double totalExclusivePercentValue;
	
	double meanInclusiveValue;
	double totalInclusiveValue;
	double meanInclusivePercentValue;
	double totalInclusivePercentValue;

    double meanCall;
    double totalCall;

    double meanSubrs;
    double totalSubrs;
    
    double meanInclPCall;
    double totalInclPCall;
	
	String meanTotalStatString;
	String totalTotalStatString; 
	

	//Constructors.
	public GlobalMappingElement()
	{
		functionName = null;
		globalID = -1;
			
		meanExclusiveValue = 0.0;
		totalExclusiveValue = 0.0;
		meanExclusivePercentValue = 0.0;
		totalExclusivePercentValue = 0.0;
		
		meanInclusiveValue = 0.0;
		totalInclusiveValue = 0.0;
		meanInclusivePercentValue = 0.0;
		totalInclusivePercentValue = 0.0;

		meanCall = 0.0;
		totalCall = 0.0;
    
		meanSubrs = 0.0;
		totalSubrs = 0.0;

		meanInclPCall = 0.0;
		totalInclPCall = 0.0;
		
		meanTotalStatString = null;
		totalTotalStatString = null;
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

	public void setGlobalID(int inGlobalID)
	{
		globalID = inGlobalID;
	}
	
	public int getGlobalID()
	{
		return globalID;
	}
		
	//Exclusive part.	
	public void setMeanExclusiveValue(double inMeanExclusiveValue)
	{
		meanExclusiveValue = inMeanExclusiveValue;
	}
	
	public double getMeanExclusiveValue()
	{
		return meanExclusiveValue;
	}
	
	public void setMeanExclusivePercentValue(double inMeanExclusivePercentValue)
	{
		meanExclusivePercentValue = inMeanExclusivePercentValue;
	}
	
	public double getMeanExclusivePercentValue()
	{
		return meanExclusivePercentValue;
	}
	
	public void setTotalExclusiveValue(double inTotalExclusiveValue)
	{
		totalExclusiveValue = inTotalExclusiveValue;
	}
	
	public double getTotalExclusiveValue()
	{
		return totalExclusiveValue;
	}
	
	public void setTotalExclusivePercentValue(double inTotalExclusivePercentValue)
	{
		totalExclusivePercentValue = inTotalExclusivePercentValue;
	}
	
	public double getTotalExclusivePercentValue()
	{
		return totalExclusivePercentValue;
	}
	
	
	//Inclusive part.
	
	public void setMeanInclusiveValue(double inMeanInclusiveValue)
	{
		meanInclusiveValue = inMeanInclusiveValue;
	}
	
	public double getMeanInclusiveValue()
	{
		return meanInclusiveValue;
	}
	
	public void setMeanInclusivePercentValue(double inMeanInclusivePercentValue)
	{
		meanInclusivePercentValue = inMeanInclusivePercentValue;
	}
	
	public double getMeanInclusivePercentValue()
	{
		return meanInclusivePercentValue;
	}
	
	public void setTotalInclusiveValue(double inTotalInclusiveValue)
	{
		totalInclusiveValue = inTotalInclusiveValue;
	}
	
	public double getTotalInclusiveValue()
	{
		return totalInclusiveValue;
	}
	
	public void setTotalInclusivePercentValue(double inTotalInclusivePercentValue)
	{
		totalInclusivePercentValue = inTotalInclusivePercentValue;
	}
	
	public double getTotalInclusivePercentValue()
	{
		return totalInclusivePercentValue;
	}
	
    public void setMeanCall(double call){
	meanCall = call;
    }

    public double getMeanCall(){
	return meanCall;
    }

    public void setTotalCall(double call){
	totalCall = call;
    }

    public double getTotalCall(){
	return totalCall;
    }

    public void setMeanSubrs(double subrs){
	meanSubrs = subrs;
    }

    public double getMeanSubrs(){
	return meanSubrs;
    }

    public void setTotalSubrs(double subrs){
	totalSubrs = subrs;
    }

    public double getTotalSubrs(){
	return totalSubrs;
    }

    public void setMeanInclPCall(double inclpcall){
	meanInclPCall = inclpcall;
    }

    public double getMeanInclPCall(){
	return meanInclPCall;
    }

    public void setTotalInclPCall(double inclpcall){
	totalInclPCall = inclpcall;
    }

    public double getTotalInclPCall(){
	return totalInclPCall;
    }

	//Total stat strings.
	
	public void setMeanTotalStatString(String inMeanTotalStatString)
	{
		meanTotalStatString = inMeanTotalStatString;
	}
	
	public String getMeanTotalStatString()
	{
		return meanTotalStatString;
	}
	
	public void setTotalTotalStatString(String inTotalTotalStatString)
	{
		totalTotalStatString = inTotalTotalStatString;
	}
	
	public String getTotalTotalStatString()
	{
		return totalTotalStatString;
	}

}
