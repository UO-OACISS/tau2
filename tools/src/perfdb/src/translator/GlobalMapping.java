package translator;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Vector;


public class GlobalMapping implements Serializable 
{
        
	Vector nameIDMapping;	//Elements in this vector for the GlobalMapping class
							//will be GlobalMappingElements.
	int numberOfGlobalFunctions;

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

       
       public boolean setFunctionGroupAt(String functionGroup, int inPosition)
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
		tmpGME.setFunctionGroup(functionGroup);
		
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

	public boolean setMeanExclusivePercentValueAt(double inMeanExclusivePercentValue, int inPosition)
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
		tmpGME.setMeanExclusivePercentValue(inMeanExclusivePercentValue);
		
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

	public boolean setMeanInclusivePercentValueAt(double inMeanInclusivePercentValue, int inPosition)
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
		tmpGME.setMeanInclusivePercentValue(inMeanInclusivePercentValue);
		
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

        public boolean setTotalExclusivePercentValueAt(double inTotalExclusivePercentValue, int inPosition)
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
		tmpGME.setTotalExclusivePercentValue(inTotalExclusivePercentValue);
		
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

        public boolean setTotalInclusivePercentValueAt(double inTotalInclusivePercentValue, int inPosition)
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
		tmpGME.setTotalInclusivePercentValue(inTotalInclusivePercentValue);
		
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
	
}
