/* 
	ColorPair.java

	Title:			jRacy
	Author:			Robert Bell
	Description:	
*/

package jRacy;

import java.util.*;
import java.awt.*;

public class ColorPair implements Comparable
{
	//Constructors.	
	public ColorPair(){}
	
	public ColorPair(String inMappingName, Color inColor)
	{
		mappingName = inMappingName;
		color = inColor;
	}
	
	
	public void setMappingName(String inMappingName)
	{
		mappingName = inMappingName;
	}
	
	public void setMappingColor(Color inColor)
	{
		color = inColor;
	}
	
	public String getMappingName()
	{
		return mappingName;
	}
	
	public Color getMappingColor()
	{
		return color;
	}
	
	public int compareTo(Object inObject)
	{
		
		try
		{
			String tmpString = ((ColorPair) inObject).getMappingName();
			return mappingName.compareTo(tmpString);
		}
		catch(Exception e)
		{
			jRacy.systemError(null, "CP01");
		}
		
		return 0;
		
	}
	
	
	//Data section.
	private String mappingName = null;
	private Color color = null;
	
}