/* 
   XMLSupport.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  This class handles support for XML.
*/

package paraprof;

import java.util.*;
import java.awt.*;
import javax.swing.*;
import java.io.*;
import java.net.InetAddress;

public class XMLSupport{

    public XMLSupport(){}

    public XMLSupport(ParaProfTrial paraProfTrial){
	this.paraProfTrial = paraProfTrial;}

    public void writeXmlFiles(int metricID, String fileName){
	GlobalMapping globalMapping = paraProfTrial.getGlobalMapping();
	
	//Build an array of group names.  This speeds lookup of group names.
	Vector groups = paraProfTrial.getGlobalMapping().getMapping(2);
	String[] groupNames = new String[groups.size()];
	int position = 0;
	for(Enumeration e = groups.elements(); e.hasMoreElements() ;){
	    GlobalMappingElement group = (GlobalMappingElement) e.nextElement();
	    groupNames[position++] = group.getMappingName();
	}

	//Get max node,context, and thread numbers.
	int[] maxNCT = paraProfTrial.getMaxNCTNumbers();
	
	try{
	    String sys = "";
	    String config = "";
	    String instru = "";
	    String compiler = "";
	    String appname = "";
	    String version = "";
	    String hostname = InetAddress.getLocalHost().getHostName();

	    File writeXml = new File(fileName);
	    BufferedWriter xwriter = new BufferedWriter(new FileWriter(writeXml));

	    xwriter.write("<?xml version=\"1.0\"?>", 0, ("<?xml version=\"1.0\"?>").length());
	    xwriter.newLine();
	    xwriter.write("<Trials>", 0, ("<Trials>").length());
	    xwriter.newLine();
	    
	    xwriter.write("    <Onetrial Metric='" + paraProfTrial.getMetricName(metricID) + "'>",
			  0, ("    <Onetrial Metric='" + paraProfTrial.getMetricName(metricID) + "'>").length());
	    xwriter.newLine();
	    
	    writeComputationModel(xwriter, maxNCT[0]+1, maxNCT[1]+1, maxNCT[2]+1);
	    
	    xwriter.write("\t<Env>", 0, ("\t<Env>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t   <AppID>" + paraProfTrial.getApplicationID() + "</AppID>",
			  0, ("\t   <AppID>" + paraProfTrial.getApplicationID() + "</AppID>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t   <ExpID>" + paraProfTrial.getExperimentID() + "</ExpID>",
			  0, ("\t   <ExpID>" + paraProfTrial.getExperimentID() + "</ExpID>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t   <TrialName>" + paraProfTrial.getName() + "</TrialName>",
			  0, ("\t   <TrialName>" + paraProfTrial.getName() + "</TrialName>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t</Env>", 0, ("\t</Env>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t<Trialtime>" + paraProfTrial.getTime() + "</Trialtime>", 0, ("\t<Trialtime>" + paraProfTrial.getTime() + "</Trialtime>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t<FunAmt>" + globalMapping.getNumberOfMappings(0) + "</FunAmt>", 0, ("\t<FunAmt>" + globalMapping.getNumberOfMappings(0) + "</FunAmt>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t<UserEventAmt>" + globalMapping.getNumberOfMappings(2) + "</UserEventAmt>", 0, ("\t<UserEventAmt>" 
													     + globalMapping.getNumberOfMappings(0) + "</UserEventAmt>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t<Pprof>", 0, ("\t<Pprof>").length());
	    xwriter.newLine();
	    
	    StringBuffer groupsStringBuffer = new StringBuffer(10);
	    Vector nodes = paraProfTrial.getNCT().getNodes();
	    for(Enumeration e1 = nodes.elements(); e1.hasMoreElements() ;){
		Node node = (Node) e1.nextElement();
		Vector contexts = node.getContexts();
		for(Enumeration e2 = contexts.elements(); e2.hasMoreElements() ;){
		    Context context = (Context) e2.nextElement();
		    Vector threads = context.getThreads();
		    for(Enumeration e3 = threads.elements(); e3.hasMoreElements() ;){
			Thread thread = (Thread) e3.nextElement();
			Vector functions = thread.getFunctionList();
			Vector userevents = thread.getUsereventList();
			//Write out the node,context and thread ids.
			writeIDs(xwriter,thread.getNodeID(),thread.getContextID(),thread.getThreadID());
			//Write out function data for this thread.
			for(Enumeration e4 = functions.elements(); e4.hasMoreElements() ;){
			    GlobalThreadDataElement function = (GlobalThreadDataElement) e4.nextElement();
			    if (function!=null){
				writeBeginObject(xwriter,16);
				writeFunName(xwriter,function.getMappingName());
				writeInt(xwriter,11,function.getMappingID());
				//Build group string.
				groupsStringBuffer.delete(0,groupsStringBuffer.length());
				int[] groupIDs = function.getGroups();
				for(int i=0;i<groupIDs.length;i++){
				    groupsStringBuffer=+groupNames[groupIDs[i]];}
				writeString(xwriter,14,groupsStringBuffer.toString());
				writeDouble(xwriter,0,function.getInclusivePercentValue());
				writeDouble(xwriter,1,function.getInclusiveValue());
				writeDouble(xwriter,2,function.getExclusivePercentValue());
				writeDouble(xwriter,3,function.getExclusiveValue());
				writeDouble(xwriter,5,function.getNumberOfCalls());
				writeDouble(xwriter,6,function.getNummberSubroutines());
				writeDouble(xwriter,4,function.getUserSecPerCall());
				writeEndObject(xwriter,16);
			    }
			}
			//Write out user event data for this thread.
			for(Enumeration e4 = UserThreadDataList.elements(); e4.hasMoreElements() ;){
			    GlobalThreadDataElement userevent = (GlobalThreadDataElement) e4.nextElement();
			    if (userevent!=null){
				writeBeginObject(xwriter,17);
				writeString(xwriter,15,userevent.getUserEventName());
				writeInt(xwriter,12,userevent.getUserEventID());
				writeInt(xwriter,13,userevent.getUserEventNumberValue());
				writeDouble(xwriter,7,userevent.getUserEventMaxValue());
				writeDouble(xwriter,8,userevent.getUserEventMinValue());
				writeDouble(xwriter,9,userevent.getUserEventMeanValue());
				writeDouble(xwriter,10,userevent.getUserEventStdDevValue());
				writeBeginObject(xwriter,17);
			    }
			}			
		    }
		}    
	    }
	    
	    xwriter.write("\t</Pprof>", 0, ("\t</Pprof>").length());
	    xwriter.newLine();
	    xwriter.newLine();
	    
	    xwriter.write("\t<totalfunsummary>", 0, ("\t<totalfunsummary>").length());
	    xwriter.newLine();
	    
	    //Write out total information.
	    globalMappingElementList = globalMapping.getMapping(0);
	    for(Enumeration e = globalMappingElementList.elements(); e.hasMoreElements() ;){
		globalMappingElement = (GlobalMappingElement) e.nextElement();
		if (globalMappingElement!=null){
		    writeBeginObject(xwriter,18);
		    writeFunName(xwriter,globalMappingElement.getFunctionName());
		    writeInt(xwriter,11,globalMappingElement.getGlobalID());
		    //Build group string.
		    groupsStringBuffer.delete(0,groupsStringBuffer.length());
		    int[] groupIDs = globalMappingElement.getGroups();
		    for(int i=0;i<groupIDs.length;i++){
			groupsStringBuffer=+groupNames[groupIDs[i]];}
		    writeString(xwriter,14,groupsStringBuffer.toString());
		    writeDouble(xwriter,0,globalMappingElement.getTotalInclusivePercentValue());
		    writeDouble(xwriter,1,globalMappingElement.getTotalInclusiveValue());
		    writeDouble(xwriter,2,globalMappingElement.getTotalExclusivePercentValue());
		    writeDouble(xwriter,3,globalMappingElement.getTotalExclusiveValue());
		    writeDouble(xwriter,5,globalMappingElement.getTotalCall());
		    writeDouble(xwriter,6,globalMappingElement.getTotalSubrs());
		    writeDouble(xwriter,4,globalMappingElement.getTotalInclPCall());
		    writeEndObject(xwriter,18);
		}
	    }
	    xwriter.write("\t</totalfunsummary>", 0, ("\t</totalfunsummary>").length());
	    xwriter.newLine();
	    xwriter.newLine();

	    xwriter.write("\t<meanfunsummary>", 0, ("\t<meanfunsummary>").length());
	    xwriter.newLine();
	    
	    //Write out mean information.
	    for(Enumeration e = globalMappingElementList.elements(); e.hasMoreElements() ;){
		globalMappingElement = (GlobalMappingElement) e.nextElement();
		if (globalMappingElement!=null){
		    writeBeginObject(xwriter,19);
		    writeFunName(xwriter,globalMappingElement.getFunctionName());
		    writeInt(xwriter,11,globalMappingElement.getGlobalID());
		    //Build group string.
		    groupsStringBuffer.delete(0,groupsStringBuffer.length());
		    int[] groupIDs = globalMappingElement.getGroups();
		    for(int i=0;i<groupIDs.length;i++){
			groupsStringBuffer=+groupNames[groupIDs[i]];}
		    writeString(xwriter,14,groupsStringBuffer.toString());
		    writeDouble(xwriter,0,globalMappingElement.getMeanInclusivePercentValue());
		    writeDouble(xwriter,1,globalMappingElement.getMeanInclusiveValue());
		    writeDouble(xwriter,2,globalMappingElement.getMeanExclusivePercentValue());
		    writeDouble(xwriter,3,globalMappingElement.getMeanExclusiveValue());
		    writeDouble(xwriter,5,globalMappingElement.getMeanCall());
		    writeDouble(xwriter,6,globalMappingElement.getMeanSubrs());
		    writeDouble(xwriter,4,globalMappingElement.getMeanInclPCall());
		}
	    }
	    
	    xwriter.write("\t</meanfunsummary>", 0, ("\t</meanfunsummary>").length());
	    xwriter.newLine();	   
	    
	    xwriter.write("    </Onetrial>", 0, ("    </Onetrial>").length());
	    xwriter.newLine();
	    
	    xwriter.write("</Trials>", 0, ("</Trials>").length());
	    xwriter.newLine();
	    xwriter.close();
	    
	    connector.dbclose();    	    		
	    }catch(Exception e){	    
		e.printStackTrace();
	    }
    }
    
    public void writeComputationModel(BufferedWriter writer, int node, int context, int thread){

	try{
	    writer.write("\t<ComputationModel>", 0, ("\t<ComputationModel>").length());
	    writer.newLine();

	    String tmpString;
	    tmpString = "\t   <node level=\"Top\" statis_info=\"sum\">"+node+"</node>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    tmpString = "\t   <context level=\"Secondary\" statis_info=\"contextPnode\">"+context+"</context>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    tmpString = "\t   <thread level=\"Lowest\" statis_info=\"threadPcontext\">"+thread+"</thread>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    writer.write("\t</ComputationModel>", 0, "\t</ComputationModel>".length());
	    writer.newLine();
	}
	catch(Exception e){
	    xmlWriteError.location = "writeComputationModel(BufferedWriter writer, int node, int context, int thread)";
	    UtilFncs.systemError(xmlWriteError, null, null);
       	}
    }

    public void writeIDs(BufferedWriter writer, int node, int context, int thread){
	
	try{
	    String tmpString;
	    tmpString = "\t   <nodeID>"+node+"</nodeID>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    tmpString = "\t   <contextID>"+context+"</contextID>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    tmpString = "\t   <threadID>"+thread+"</threadID>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	    
	}
	catch(Exception e){
	    xmlWriteError.location = "writeIDs(BufferedWriter writer, int node, int context, int thread)";
	    UtilFncs.systemError(xmlWriteError, null, null);
       	}

    }

    public void writeBeginObject(BufferedWriter writer, int type){
	try{
	    writer.write(this.lookupBegin[type], 0, this.lookupBegin[type].length());
	    writer.newLine();
	}
	catch(Exception e){
	    xmlWriteError.location = "writeBeginObject(BufferedWriter writer, int type)";
	    UtilFncs.systemError(xmlWriteError, null, null);
       	}
    }

    public void writeEndObject(BufferedWriter writer, int type){
	try{
	    writer.write(this.lookupEnd[type], 0, this.lookupEnd[type].length());
	    writer.newLine();
	}
	catch(Exception e){
	    xmlWriteError.location = "writeEndObject(BufferedWriter writer, int type)";
	    UtilFncs.systemError(xmlWriteError, null, null);
       	}
    }

    public void writeFunName(BufferedWriter writer, String funname){
	try{
	    String tmpString;
	    funname = replace(funname, "&", "&amp;");
	    funname = replace(funname, "<", "&lt;");
	    funname = replace(funname, ">", "&gt;");
	    tmpString = "\t\t<funname>"+funname+"</funname>";
	    writer.write(tmpString, 0, tmpString.length());
	    writer.newLine();
	}
	catch(Exception e){
	    xmlWriteError.location = "writeFunName(BufferedWriter writer, String funname)";
	    UtilFncs.systemError(xmlWriteError, null, null);
       	}

    }

    public void writeString(BufferedWriter writer, int type, String s1){
	try{
	    String s2 = this.lookupBegin[type]+s1+this.lookupEnd[type];
	    writer.write(s2, 0, s2.length());
	    writer.newLine();
	}
	catch(Exception e){
	    xmlWriteError.location = "writeString(BufferedWriter writer, int type, String s1)";
	    UtilFncs.systemError(xmlWriteError, null, null);
       	}
    }

    public void writeInt(BufferedWriter writer, int type, int i){
	try{
	    String s = this.lookupBegin[type]+i+this.lookupEnd[type];
	    writer.write(s, 0, s.length());
	    writer.newLine();
	}
	catch(Exception e){
	    xmlWriteError.location = "writeInt(BufferedWriter writer, int type, int i)";
	    UtilFncs.systemError(xmlWriteError, null, null);
       	}

    }

    public void writeDouble(BufferedWriter writer, int type, double d){
	try{
	    String s = this.lookupBegin[type]+d+this.lookupEnd[type];
	    writer.write(s, 0, s.length());
	    writer.newLine();
	}
	catch(Exception e){
	    xmlWriteError.location = "writeDouble(BufferedWriter writer, int type, double d)";
	    UtilFncs.systemError(xmlWriteError, null, null);
       	}

    }
    
    public String getMetric(String str){
	if (str.length() > 26)
		return(str.substring(26, str.length()));
	else return new String("");
    }

    public String replace(String str, String lstr, String rstr){
	String tempStr = "";
        int i;
        while ( (i=str.indexOf(lstr)) != -1) {
                if (i>0)
                        tempStr += str.substring(0,i);
                tempStr += rstr;
                str = str.substring(i+1);
        }
        tempStr += str;
        return tempStr;
    } 	
 
    public String getFunAmt(String inString){
	try{
		String tmpString;
		StringTokenizer funAmtTokenizer = new StringTokenizer(inString, " \t\n\r");
		tmpString = funAmtTokenizer.nextToken();
		return tmpString;
	}
	catch (Exception e){
		e.printStackTrace();
	}
	return null;
    }

    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial paraProfTrial = null;

    private String[] lookupBegin = {"\t\t<inclperc>","\t\t<inclutime>","\t\t<exclperc>","\t\t<exclutime>", //0,1,2,3
				  "\t\t<inclutimePcall>", //4
				  "\t\t<call>","\t\t<subrs>", //5,6
				  "\t\t<maxvalue>","\t\t<minvalue>","\t\t<meanvalue>","\t\t<stddevvalue>", //7,8,9,10
				  "\t\t<funID>","\t\t<ueID>","\t\t<numofsamples>", //11,12,13
				  "\t\t<fungroup>","\t\t<uename>", //14,15
				  "\t   <instrumentedobj>","\t   <userevent>","\t   <totalfunction>", //16,17,18
				  "\t   <meanfunction>"}; //19
    private String[] lookupEnd = {"</inclperc>","</inclutime>","</exclperc>","</exclutime>", //0,1,2,3
				"</inclutimePcall>", //4
				"</call>","</subrs>", //5,6
				"</maxvalue>","</minvalue>","</meanvalue>","</stddevvalue>", //7,8,9,10
				"</funID>","</ueID>","</numofsamples>", //11,12,13
				"\t\t<fungroup>","\t\t<uename>", //14,15
				"\t   </instrumentedobj>","\t   </userevent>","\t   </totalfunction>", //16,17,18
				"\t   </meanfunction>"}; //19

    //######
    //Error outupt.
    //######
    //For errors writing to the xml file.
    private ParaProfError xmlWriteError = new ParaProfError("", "XML Write Error: See console for details.",
						    "An error occured whilst writing XML file. Operation aborted!", 
						    "Note: Dependent operations also aborted.", null, false);

    //####################################
    //End - Instance data.
    //####################################
}
