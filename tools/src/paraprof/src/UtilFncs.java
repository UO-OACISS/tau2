/* 
   UtilFncs.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  Some useful functions for the system.
*/

package paraprof;
import java.util.*;
import java.lang.*;
import java.io.*;
import java.text.*;

public class UtilFncs{
  
    public static double adjustDoublePresision(double inDouble, int precision){
	String result = null;
    
	try{
	    String formatString = "##0.0";
	    for(int i=0;i<(precision-1);i++){
		formatString = formatString+"0";
	    }
	    if(inDouble < 0.001){
		for(int i=0;i<4;i++){
		    formatString = formatString+"0";
		}
	    }
        
	    DecimalFormat dF = new DecimalFormat(formatString);
	    result = dF.format(inDouble);
	}
	catch(Exception e)
	    {
		ParaProf.systemError(e, null, "UF01");
	    }
	return Double.parseDouble(result);
    }
    
    public static String getTestString(double inDouble, int precision){
      
	//This method comes up with a rough estimation.  The drawing windows do not
	//need to be absolutely accurate.
      
	String returnString = "";
	for(int i=0;i<precision;i++){
	    returnString = returnString + " ";
	}
      
	long tmpLong = Math.round(inDouble);
	returnString = Long.toString(tmpLong) + returnString;
      
	return returnString;
    }

    //This method is used in a number of windows to determine the actual output string
    //displayed. Current types are:
    //0 - microseconds
    //1 - milliseconds
    //2 - seconds
    //At present, the passed in double value is assumed to be in microseconds.
    public static String getOutputString(int type, double d){
	switch(type){
	case 0:
	    return (Double.toString(UtilFncs.adjustDoublePresision(d, ParaProf.defaultNumberPrecision)));
	case 1:
	    return (Double.toString(UtilFncs.adjustDoublePresision((d/1000), ParaProf.defaultNumberPrecision)));
	case 2:
	    return (Double.toString(UtilFncs.adjustDoublePresision((d/1000000), ParaProf.defaultNumberPrecision)));
	default:
	    ParaProf.systemError(null, null, "Unexpected string type - UF02 value: " + type);
	}
	return null;
    }

    public static int exists(int[] ref, int i){
	if(ref == null)
	    return -1;
	int test = ref.length;
	for(int j=0;j<test;j++){
	    if(ref[j]==i)
		return j;
	}
	return -1;
    }

    public static int exists(Vector ref, int i){
	//Assuming a vector of Integers.
	if(ref == null)
	    return -1;
	Integer current = null;
	int test = ref.size();
	for(int j=0;j<test;j++){
	    current = (Integer) ref.elementAt(j);
	    if((current.intValue())==i)
		return j;
	}
	return -1;
    }
}
