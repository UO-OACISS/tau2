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

public class XMLSupport{

    public XMLSupport(){}

    public XMLSupport(ParaProfTrial paraProfTrial){
	this.paraProfTrial = paraProfTrial}

    public void writeXmlFiles(int metricID){
	GlobalMappingElement globalMapping = paraProfTrial.getGlobalMapping();
	try{
	    String sys = "";
	    String config = "";
	    String instru = "";
	    String compiler = "";
	    String appname = "";
	    String version = "";
	    String hostname = InetAddress.getLocalHost().getHostName();

	    BufferedWriter xwriter = new BufferedWriter(new FileWriter(writeXml));

	    xwriter.write("<?xml version=\"1.0\"?>", 0, ("<?xml version=\"1.0\"?>").length());
	    xwriter.newLine();
	    xwriter.write("<Trials>", 0, ("<Trials>").length());
	    xwriter.newLine();
	    
	    xwriter.write("    <Onetrial Metric='" + paraProfTrial.getMetricName(metricID) + "'>",
			  0, ("    <Onetrial Metric='" + paraProfTrial.getMetricName(metricID) + "'>").length());
	    xwriter.newLine();
	    
	    writeComputationModel(xwriter, maxNode+1, maxContext+1, maxThread+1);
	    
	    xwriter.write("\t<Env>", 0, ("\t<Env>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t   <AppID>" + paraProfTrial.getApplicationID() + "</AppID>",
			  0, ("\t   <AppID>" + paraProfTrial.getApplicationID() + "</AppID>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t   <ExpID>" + paraProfTrial.getExperimentID() + "</ExpID>",
			  0, ("\t   <ExpID>" + paraProfTrial.getExperimentID() + "</ExpID>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t   <TrialName>" + paraProfTrial.getTrialIdentifier() + "</TrialName>",
			  0, ("\t   <TrialName>" + paraProfTrial.getTrialIdentifier() + "</TrialName>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t</Env>", 0, ("\t</Env>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t<Trialtime>" + paraProfTrial.getTime() + "</Trialtime>", 0, ("\t<Trialtime>" + paraProfTrial.getTime + "</Trialtime>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t<FunAmt>" + globalMapping.getNumberOfMappings(0) + "</FunAmt>", 0, ("\t<FunAmt>" + globalMapping.getNumberOfMappings(0) + "</FunAmt>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t<UserEventAmt>" + globalMapping.getNumberOfMappings(2) + "</UserEventAmt>", 0, ("\t<UserEventAmt>" 
													     + globalMapping.getNumberOfMappings(0) + "</UserEventAmt>").length());
	    xwriter.newLine();
	    
	    xwriter.write("\t<Pprof>", 0, ("\t<Pprof>").length());
	    xwriter.newLine();
	    

	    ListIterator list = paraProfTrial.getNCT();
	    while(
	    for(Enumeration en = NodeList.elements(); en.hasMoreElements() ;){
		nodeObject = (GlobalNode) en.nextElement();
		currentnode = nodeObject.getNodeName();
		
		ContextList = nodeObject.getContextList();
		for(Enumeration ec = ContextList.elements(); ec.hasMoreElements() ;){
		    contextObject = (GlobalContext) ec.nextElement();
		    currentcontext = contextObject.getContextName();
		    
		    ThreadList = contextObject.getThreadList();
		    for(Enumeration et = ThreadList.elements(); et.hasMoreElements() ;){
			threadObject = (GlobalThread) et.nextElement();
			currentthread = threadObject.getThreadName();
			
			ThreadDataList = threadObject.getThreadDataList();
			UserThreadDataList = threadObject.getUserThreadDataList();
			
			writeIDs(xwriter,currentnode,currentcontext, currentthread);
			
			for(Enumeration ef = ThreadDataList.elements(); ef.hasMoreElements() ;){
			    funObject = (GlobalThreadDataElement) ef.nextElement();
			    if (funObject!=null){
				writeFunName(xwriter,funObject.getFunctionName());
				writeFunID(xwriter, funObject.getFunctionID());
				writeFunGroup(xwriter, funObject.getFunctionGroup());
				writeInclPerc(xwriter,funObject.getInclPercValue());
				writeIncl(xwriter,funObject.getInclValue());
				writeExclPerc(xwriter,funObject.getExclPercValue());
				writeExcl(xwriter, funObject.getExclValue());
				writeCall(xwriter,funObject.getCall());
				writeSubrs(xwriter,funObject.getSubrs());
				writeInclPCall(xwriter,funObject.getInclPCall());	
			    }
			}
			
			for(Enumeration uef = UserThreadDataList.elements(); uef.hasMoreElements() ;){
			    ueObject = (GlobalThreadDataElement) uef.nextElement();
			    if (ueObject!=null){
				writeUEName(xwriter,ueObject.getUserEventName());
				writeUEID(xwriter, ueObject.getUserEventID());
				writeNumofSamples(xwriter,ueObject.getUserEventNumberValue());
				writeMaxValue(xwriter,ueObject.getUserEventMaxValue());
				writeMinValue(xwriter,ueObject.getUserEventMinValue());
				writeMeanValue(xwriter, ueObject.getUserEventMeanValue());
				writeStdDevValue(xwriter, ueObject.getUserEventStdDevValue());
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
	    
	    // process total information
	    globalMappingElementList = globalMapping.getNameIDMapping();
	    
	    for(Enumeration et = globalMappingElementList.elements(); et.hasMoreElements() ;){
		
		globalmappingObject = (GlobalMappingElement) et.nextElement();
		if (globalmappingObject.getFunctionName() != "Name has not been set!"){
		    writeTotalFunName(xwriter, globalmappingObject.getFunctionName());
		    writeTotalFunID(xwriter, globalmappingObject.getGlobalID());
		    writeTotalFunGroup(xwriter, globalmappingObject.getFunctionGroup());
		    writeTotalInclPerc(xwriter, globalmappingObject.getTotalInclusivePercentValue());
		    writeTotalIncl(xwriter, globalmappingObject.getTotalInclusiveValue());
		    writeTotalExclPerc(xwriter, globalmappingObject.getTotalExclusivePercentValue());
		    writeTotalExcl(xwriter, globalmappingObject.getTotalExclusiveValue());
		    writeTotalCall(xwriter, globalmappingObject.getTotalCall());
		    writeTotalSubrs(xwriter, globalmappingObject.getTotalSubrs());
		    writeTotalInclPCall(xwriter, globalmappingObject.getTotalInclPCall());
		}
	    }
	    xwriter.write("\t</totalfunsummary>", 0, ("\t</totalfunsummary>").length());
	    xwriter.newLine();
	    xwriter.newLine();

	    xwriter.write("\t<meanfunsummary>", 0, ("\t<meanfunsummary>").length());
	    xwriter.newLine();
	    
	    for(Enumeration em = globalMappingElementList.elements(); em.hasMoreElements() ;){
		globalmappingObject = (GlobalMappingElement) em.nextElement();
		if (globalmappingObject.getFunctionName() != "Name has not been set!"){
		    // mean stuff 
		    writeMeanFunName(xwriter, globalmappingObject.getFunctionName());
		    writeMeanFunID(xwriter, globalmappingObject.getGlobalID());
		    writeMeanFunGroup(xwriter, globalmappingObject.getFunctionGroup());
		    writeMeanInclPerc(xwriter,globalmappingObject.getMeanInclusivePercentValue());
		    writeMeanIncl(xwriter,globalmappingObject.getMeanInclusiveValue());
		    writeMeanExclPerc(xwriter,globalmappingObject.getMeanExclusivePercentValue());
		    writeMeanExcl(xwriter,globalmappingObject.getMeanExclusiveValue());
		    writeMeanCall(xwriter,globalmappingObject.getMeanCall());
		    writeMeanSubrs(xwriter,globalmappingObject.getMeanSubrs());
		    writeMeanInclPCall(xwriter, globalmappingObject.getMeanInclPCall());			    		    
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

    private String lookupBegin = {"\t\t<inclperc>","\t\t<inclutime>","\t\t<exclperc>","\t\t<exclutime>", //0,1,2,3
				  "\t\t<inclutimePcall>", //4
				  "\t\t<call>","\t\t<subrs>", //5,6
				  "\t\t<maxvalue>","\t\t<minvalue>","\t\t<meanvalue>","\t\t<stddevvalue>", //7,8,9
				  "\t\t<funID>","\t\t<ueID>","\t\t<numofsamples>", //10,11,12
				  "\t\t<fungroup>","\t\t<uename>","\t\t<>","\t\t<>","\t\t<>",
				  "\t   <instrumentedobj>","\t   <userevent>","\t   <totalfunction>",
				  "\t   <meanfunction>"};
    private String lookupEnd = {"</inclperc>","</inclutime>","</exclperc>","</exclutime>", //0,1,2,3
				"</inclutimePcall>", //4
				"</call>","</subrs>", //5,6
				"</maxvalue>","</minvalue>","</meanvalue>","</stddevvalue>", //7,8,9
				"</funID>","</ueID>","</numofsamples>", //10,11,12
				"\t\t<fungroup>","\t\t<uename>","\t\t<>","\t\t<>","\t\t<>",
				"\t   </instrumentedobj>","\t   </userevent>","\t   </totalfunction>",
				"\t   </meanfunction>"};

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
