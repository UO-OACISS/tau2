 /* 
   Name:        UtilFncs.java
   Author:      Robert Bell
   Description: Some useful functions for the system.
*/

package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.lang.*;
import java.io.*;
import java.text.*;

public class UtilFncs{


    public static String pad (String s, int plen) {
	int len = plen - s.length();
	if (len <= 0) 
	    return s;
	char padchars[] = new char[len];
	for (int i = 0; i < len; i++) 
	    padchars[i] = ' ';
	return s.concat(new String (padchars, 0, len));
    }

  
    public static double adjustDoublePresision(double d, int precision){
	String result = null;
 	try{
	    String formatString = "##0.0";
	    for(int i=0;i<(precision-1);i++){
		formatString = formatString+"0";
	    }
	    if(d < 0.001){
		for(int i=0;i<4;i++){
		    formatString = formatString+"0";
		}
	    }
        
	    DecimalFormat dF = new DecimalFormat(formatString);
	    result = dF.format(d);
	}
	catch(Exception e){
		UtilFncs.systemError(e, null, "UF01");
	}
	return Double.parseDouble(result);
    }
    
    //This method is used in a number of windows to determine the actual output string
    //displayed. Current types are:
    //0 - microseconds
    //1 - milliseconds
    //2 - seconds
    //3 - hr:min:sec
    //At present, the passed in double value is assumed to be in microseconds.
    public static String getOutputString(int type, double d, int precision){
	switch(type){
	case 0:
	    return (Double.toString(UtilFncs.adjustDoublePresision(d, precision)));
	case 1:
	    return (Double.toString(UtilFncs.adjustDoublePresision((d/1000), precision)));
	case 2:
	    return (Double.toString(UtilFncs.adjustDoublePresision((d/1000000), precision)));
	case 3:
	    int hr = 0;
	    int min = 0;
	    hr = (int) (d/3600000000.00);
	    //Calculate the number of microseconds left after hours are subtracted.
	    d = d-hr*3600000000.00;
	    min = (int) (d/60000000.00);
	    //Calculate the number of microseconds left after minutess are subtracted.
	    d = d-min*60000000.00;
	    return (Integer.toString(hr)+":"+Integer.toString(min)+":"+Double.toString(UtilFncs.adjustDoublePresision((d/1000000), precision)));
	default:
	    UtilFncs.systemError(null, null, "Unexpected string type - UF02 value: " + type);
	}
	return null;
    }

    public static String getUnitsString(int type, boolean time, boolean derived){
	if(derived){
	   if(!time)
	       return "counts";
	   switch(type){
	   case 0:
	       return "Derived metric shown in microseconds format";
	   case 1:
	       return "Derived metric shown in milliseconds format";
	   case 2:
	       return "Derived metric shown in seconds format";
	   case 3:
	       return "Derived metric shown in hour:minute:seconds format"; 
	   }
	}
	else{
	    if(!time)
		return "counts";
	    switch(type){
	    case 0:
		return "microseconds";
	    case 1:
		return "milliseconds";
	    case 2:
		return "seconds";
	    case 3:
		return "hour:minute:seconds";
	    default:
		UtilFncs.systemError(null, null, "Unexpected string type - UF03 value: " + type);
	    }
	}
	return null;
    }

    public static String getValueTypeString(int type){
	switch(type){
	case 2:
	    return "exclusive";
	case 4:
	    return "inclusive";
	case 6:
	    return "number of calls";
	case 8:
	    return "number of subroutines";
	case 10:
	    return "per call value";
	case 12:
	    return "number of userevents";
	case 14:
	    return "minimum number of userevents";
	case 16:
	    return "maximum number of userevents";
	case 18:
	    return "mean number of userevents";
	default:
	    UtilFncs.systemError(null, null, "Unexpected string type - UF04 value: " + type);
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

    //####################################
    //Error handling.
    //####################################
    public static boolean debug = false;
    public static Debug objectDebug = null;


    public static void systemError(Object obj, Component component, String string){ 
	System.out.println("####################################");
	boolean quit = true; //Quit by default.
	if(obj != null){
	    if(obj instanceof Exception){
		Exception exception = (Exception) obj;
		if(UtilFncs.debug){
		    System.out.println(exception.toString());
		    exception.printStackTrace();
		    System.out.println("\n");
		}
		System.out.println("An error was detected: " + string);
		System.out.println(ParaProfError.contactString);
	    }
	    if(obj instanceof ParaProfError){
		ParaProfError paraProfError = (ParaProfError) obj;
		if(UtilFncs.debug){
		    if((paraProfError.showPopup)&&(paraProfError.popupString!=null))
			JOptionPane.showMessageDialog(paraProfError.component,
						      "ParaProf Error", paraProfError.popupString, JOptionPane.ERROR_MESSAGE);
		    if(paraProfError.exp!=null){
			System.out.println(paraProfError.exp.toString());
			paraProfError.exp.printStackTrace();
			System.out.println("\n");
		    }
		    if(paraProfError.location!=null)
			System.out.println("Location: " + paraProfError.location);
		    if(paraProfError.s0!=null)
			System.out.println(paraProfError.s0);
		    if(paraProfError.s1!=null)
			System.out.println(paraProfError.s1);
		    if(paraProfError.showContactString)
			System.out.println(ParaProfError.contactString);
		}
		else{
		    if((paraProfError.showPopup)&&(paraProfError.popupString!=null))
			JOptionPane.showMessageDialog(paraProfError.component,
						      paraProfError.popupString, "ParaProf Error", JOptionPane.ERROR_MESSAGE);
		    if(paraProfError.location!=null)
			System.out.println("Location: " + paraProfError.location);
		    if(paraProfError.s0!=null)
			System.out.println(paraProfError.s0);
		    if(paraProfError.s1!=null)
			System.out.println(paraProfError.s1);
		    if(paraProfError.showContactString)
			System.out.println(ParaProfError.contactString);
		}
		quit = paraProfError.quit;
	    }
	    else{
		System.out.println("An error has been detected: " + string);
	    }
	}
	else{
	    System.out.println("An error was detected at " + string);
	}
	System.out.println("####################################");
	if(quit)
	    System.exit(0);
    }
    //####################################
    //End - Error handling.
    //####################################

    //####################################
    //Test system state functions.
    //These functions are used to test
    //the current state of the system. 
    //####################################

    //Print the passed in data session data out to a file or to the console.
    //If the passed in File object is null, data is printed to the console. Component can be null.
    public static void outputData(DataSession dataSession, File file, Component component){
	try{
	    boolean toFile = false;
	    PrintWriter out = null;
	    Vector list = null;
	    GlobalMapping globalMapping = dataSession.getGlobalMapping();
	    int numberOfMetrics = dataSession.getNumberOfMetrics();
	    StringBuffer output = new StringBuffer(1000);
	    
	    if(file!=null){
		out = new PrintWriter(new FileWriter(file));
		toFile=true;
	    }

	    //######
	    //Metric data.
	    //######
	    if(toFile){
		out.println("<metrics>");
		out.println("<numofmetrics>"+numberOfMetrics+"</numofmetrics>");
	    }
	    else{
		System.out.println("<metrics>");
		System.out.println("<numofmetrics>"+numberOfMetrics+"</numofmetrics>");
	    }
	    for(int metric=0;metric<numberOfMetrics;metric++){
		    if(toFile)
			out.println(dataSession.getMetricName(metric));
		    else
			System.out.println(dataSession.getMetricName(metric));
	    }
	    if(toFile)
		out.println("</metrics>");
	    else
		System.out.println("</metrics>");
	    //######
	    //End - Metric data.
	    //######
	    
	    //####################################
	    //Global Data.
	    //####################################
	    
	    //######
	    //Function data.
	    //######
	    list = globalMapping.getMapping(0);
	    
	    //Name to ID map.
	    if(toFile){
		out.println("<funnameidmap>");
		out.println("<numoffunctions>"+list.size()+"</numoffunctions>");
	    }
	    else{
		System.out.println("<funnameidmap>");
		System.out.println("<numoffunctions>"+list.size()+"</numoffunctions>");
	    }
	    for(Enumeration e = list.elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		if(toFile)
		    out.println("\""+globalMappingElement.getMappingName()+"\""+globalMappingElement.getMappingID());
		else
		    System.out.println("\""+globalMappingElement.getMappingName()+"\""+globalMappingElement.getMappingID());
	    }
	    if(toFile)
		out.println("</funnameidmap>");
	    else
		System.out.println("</funnameidmap>");

	    if(toFile)
		out.println("id mincl(..) mexcl(..) minclp(..) mexclp(..) museccall(..) mnoc mnos");
	    else
		System.out.println("id mincl(..) mexcl(..) minclp(..) mexclp(..) museccall(..) mnoc mnos");
	    for(Enumeration e = list.elements(); e.hasMoreElements() ;){
		output.delete(0,output.length());
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		output.append(globalMappingElement.getMappingID()+" ");
		for(int metric=0;metric<numberOfMetrics;metric++){
		    output.append(globalMappingElement.getMeanInclusiveValue(metric)+" ");
		    output.append(globalMappingElement.getMeanExclusiveValue(metric)+" ");
		    output.append(globalMappingElement.getMeanInclusivePercentValue(metric)+" ");
		    output.append(globalMappingElement.getMeanExclusivePercentValue(metric)+" ");
		    output.append(globalMappingElement.getMeanUserSecPerCall(metric)+" ");
		}
		output.append(globalMappingElement.getMeanNumberOfCalls()+" ");
		output.append(globalMappingElement.getMeanNumberOfSubRoutines()+"");
		if(toFile)
		    out.println(output);
		else
		    System.out.println(output);
	    }
	    //######
	    //End - Function data.
	    //######

	    //######
	    //User event data.
	    //######
	    list = globalMapping.getMapping(2);
	    //Name to ID map.
	    if(toFile){
		out.println("<usereventnameidmap>");
		out.println("<numofuserevents>"+list.size()+"</numofuserevents>");
	    }
	    else{
		System.out.println("<usereventnameidmap>");
		System.out.println("<numofuserevents>"+list.size()+"</numofuserevents>");
	    }
	    for(Enumeration e = list.elements(); e.hasMoreElements() ;){
		GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
		if(toFile)
		    out.println("\""+globalMappingElement.getMappingName()+"\""+globalMappingElement.getMappingID());
		else
		    System.out.println("\""+globalMappingElement.getMappingName()+"\""+globalMappingElement.getMappingID());
	    }
	    if(toFile)
		out.println("</usereventnameidmap>");
	    else
		System.out.println("</usereventnameidmap>");
	    //######
	    //End - User event data.
	    //######

	    //####################################
	    //End - Global Data.
	    //####################################

	    //######
	    //Thread data.
	    //######
	    if(toFile){
		out.println("<threaddata>");
		out.println("funid incl(..) excl(..) inclp(..) exclp(..) useccall(..) mnoc mnos");
		out.println("usereventid num min max mean");
		out.println("<numofthreads>"+dataSession.getNCT().getTotalNumberOfThreads()+"</numofthreads>");
	    }
	    else{
		System.out.println("<threaddata>");
		System.out.println("id incl(..) excl(..) inclp(..) exclp(..) useccall(..) noc nos");
		System.out.println("usereventid num min max mean");
		System.out.println("<numofthreads>"+dataSession.getNCT().getTotalNumberOfThreads()+"</numofthreads>");
	    }
	    
	    String test = null;

	    for(Enumeration e1 = dataSession.getNCT().getNodes().elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    for(Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;){
			Thread thread = (Thread) e3.nextElement();
			ListIterator l = null;
			if(toFile)
			    out.println("<thread>"+thread.getNodeID()+","+thread.getContextID()+","+thread.getThreadID()+"</thread");
			else
			    System.out.println("<thread>"+thread.getNodeID()+","+thread.getContextID()+","+thread.getThreadID()+"</thread");
			if(toFile)
			    out.println("<functiondata>");
			else
			    System.out.println("<functiondata>");
			l = thread.getFunctionListIterator();
			while(l.hasNext()){
			    output.delete(0,output.length());
			    GlobalThreadDataElement globalThreadDataElement = (GlobalThreadDataElement) l.next();
			    if(globalThreadDataElement!=null){
				output.append(globalThreadDataElement.getMappingID()+" ");
				for(int metric=0;metric<numberOfMetrics;metric++){
				    output.append(globalThreadDataElement.getInclusiveValue(metric)+" ");
				    output.append(globalThreadDataElement.getExclusiveValue(metric)+" ");
				    output.append(globalThreadDataElement.getInclusivePercentValue(metric)+" ");
				    output.append(globalThreadDataElement.getExclusivePercentValue(metric)+" ");
				    output.append(globalThreadDataElement.getUserSecPerCall(metric)+" ");
				}
				output.append(globalThreadDataElement.getNumberOfCalls()+" ");
				output.append(globalThreadDataElement.getNumberOfSubRoutines()+"");
				if(toFile)
				    out.println(output);
				else
				    System.out.println(output);
			    }
			}
			if(toFile)
			    out.println("</functiondata>");
			else
			    System.out.println("</functiondata>");
			if(toFile)
			    out.println("<usereventdata>");
			else
			    System.out.println("<usereventdata>");
			l = thread.getUsereventListIterator();
			while(l.hasNext()){
			    output.delete(0,output.length());
			    GlobalThreadDataElement globalThreadDataElement = (GlobalThreadDataElement) l.next();
			    if(globalThreadDataElement!=null){
				output.append(globalThreadDataElement.getMappingID()+" ");
				for(int metric=0;metric<numberOfMetrics;metric++){
				    output.append(globalThreadDataElement.getUserEventNumberValue()+" ");
				    output.append(globalThreadDataElement.getUserEventMinValue()+" ");
				    output.append(globalThreadDataElement.getUserEventMaxValue()+" ");
				    output.append(globalThreadDataElement.getUserEventMeanValue()+"");
				}
				if(toFile)
				    out.println(output);
				else
				    System.out.println(output);
			    }
			}
			if(toFile)
			    out.println("</usereventdata>");
			else
			    System.out.println("</usereventdata>");
		    }
		}
	    }
	    
	    if(toFile)
		out.println("</threaddata>");
	    else
		System.out.println("</threaddata>");
	    //######
	    //End - Thread data.
	    //######
	    
	    //Flush output buffer and close file if required.
	    if(out!=null){
		out.flush();
		out.close();
	    }

	}
	catch(Exception exception){
	    UtilFncs.systemError(new ParaProfError("UF05", "File write error! Check console for details.",
						   "An error occurred whilst trying to save txt file.",null,
						   exception, component, null, null, true, false, false), null, null);
	}
    }
    //####################################
    //End - Test system state functions.
    //####################################
}
