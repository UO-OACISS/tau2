/*
 * XMLSupport.java
 * 
 * Title: ParaProf Author: Robert Bell Description: This class handles support
 * for XML.
 */

package edu.uoregon.tau.perfdmf;


public class XMLSupport {

    public XMLSupport() {
    }

    public XMLSupport(Trial trial) {
//        this.trial = trial;
    }

    //public void writeXmlFiles(int metricID, File file) {
        
      
//        TrialData trialData = trial.getDataSource().getTrialData();
//
//        //Build an array of group names. This speeds lookup of group names.
//        Vector groups = trialData.getMapping(1);
//        String[] groupNames = new String[groups.size()];
//        int position = 0;
//        for (Enumeration e = groups.elements(); e.hasMoreElements();) {
//            GlobalMappingElement group = (GlobalMappingElement) e.nextElement();
//            groupNames[position++] = group.getMappingName();
//        }
//
//        //Get max node,context, and thread numbers.
//        int[] maxNCT = trial.getMaxNCTNumbers();
//
//        try {
//            String sys = "";
//            String config = "";
//            String instru = "";
//            String compiler = "";
//            String appname = "";
//            String version = "";
//            String hostname = InetAddress.getLocalHost().getHostName();
//
//            BufferedWriter xwriter = new BufferedWriter(new FileWriter(file));
//
//            xwriter.write("<?xml version=\"1.0\"?>", 0,
//                          ("<?xml version=\"1.0\"?>").length());
//            xwriter.newLine();
//            xwriter.write("<Trials>", 0, ("<Trials>").length());
//            xwriter.newLine();
//
//            xwriter.write(
//                          "    <Onetrial Metric='"
//                                  + trial.getDataSource().getMetricName(
//                                                                        metricID)
//                                  + "'>",
//                          0,
//                          ("    <Onetrial Metric='"
//                                  + trial.getDataSource().getMetricName(
//                                                                        metricID) + "'>").length());
//            xwriter.newLine();
//
//            writeComputationModel(xwriter, maxNCT[0] + 1, maxNCT[1] + 1,
//                                  maxNCT[2] + 1);
//
//            xwriter.write("\t<Env>", 0, ("\t<Env>").length());
//            xwriter.newLine();
//
//            xwriter.write(
//                          "\t   <AppID>" + trial.getApplicationID()
//                                  + "</AppID>",
//                          0,
//                          ("\t   <AppID>" + trial.getApplicationID() + "</AppID>").length());
//            xwriter.newLine();
//
//            xwriter.write(
//                          "\t   <ExpID>" + trial.getExperimentID() + "</ExpID>",
//                          0,
//                          ("\t   <ExpID>" + trial.getExperimentID() + "</ExpID>").length());
//            xwriter.newLine();
//
//            xwriter.write(
//                          "\t   <TrialName>" + trial.getName() + "</TrialName>",
//                          0,
//                          ("\t   <TrialName>" + trial.getName() + "</TrialName>").length());
//            xwriter.newLine();
//
//            xwriter.write("\t</Env>", 0, ("\t</Env>").length());
//            xwriter.newLine();
//
//            // 	    xwriter.write("\t<Trialtime>" + trial.getTime() + "</Trialtime>",
//            // 0, ("\t<Trialtime>" + trial.getTime() +
//            // "</Trialtime>").length());
//            // 	    xwriter.newLine();
//
//            xwriter.write(
//                          "\t<FunAmt>" + trialData.getNumberOfMappings(0)
//                                  + "</FunAmt>",
//                          0,
//                          ("\t<FunAmt>" + trialData.getNumberOfMappings(0) + "</FunAmt>").length());
//            xwriter.newLine();
//
//            xwriter.write(
//                          "\t<UserEventAmt>"
//                                  + trialData.getNumberOfMappings(2)
//                                  + "</UserEventAmt>",
//                          0,
//                          ("\t<UserEventAmt>"
//                                  + trialData.getNumberOfMappings(2) + "</UserEventAmt>").length());
//            xwriter.newLine();
//
//            xwriter.write("\t<Pprof>", 0, ("\t<Pprof>").length());
//            xwriter.newLine();
//
//            //Write out function name to id mapping.
//            writeBeginObject(xwriter, 20);
//            for (Enumeration e = trialData.getMapping(0).elements(); e.hasMoreElements();) {
//                GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
//                if (globalMappingElement != null)
//                    writeNameIDMap(xwriter,
//                                   globalMappingElement.getMappingName(),
//                                   globalMappingElement.getMappingID());
//            }
//            writeEndObject(xwriter, 20);
//
//            //Write out userevent name to id mapping.
//            writeBeginObject(xwriter, 21);
//            for (Enumeration e = trialData.getMapping(2).elements(); e.hasMoreElements();) {
//                GlobalMappingElement globalMappingElement = (GlobalMappingElement) e.nextElement();
//                if (globalMappingElement != null)
//                    writeNameIDMap(xwriter,
//                                   globalMappingElement.getMappingName(),
//                                   globalMappingElement.getMappingID());
//            }
//            writeEndObject(xwriter, 21);
//
//            StringBuffer groupsStringBuffer = new StringBuffer(10);
//            Vector nodes = trial.getDataSource().getNCT().getNodes();
//            for (Enumeration e1 = nodes.elements(); e1.hasMoreElements();) {
//                Node node = (Node) e1.nextElement();
//                Vector contexts = node.getContexts();
//                for (Enumeration e2 = contexts.elements(); e2.hasMoreElements();) {
//                    Context context = (Context) e2.nextElement();
//                    Vector threads = context.getThreads();
//                    for (Enumeration e3 = threads.elements(); e3.hasMoreElements();) {
//                        edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
//                        Vector functionProfiles = thread.getFunctionList();
//                        Vector userEvents = thread.getUsereventList();
//                        //Write out the node,context and thread ids.
//                        writeIDs(xwriter, thread.getNodeID(),
//                                 thread.getContextID(), thread.getThreadID());
//                        //Write out function data for this thread.
//                        for (Enumeration e4 = functionProfiles.elements(); e4.hasMoreElements();) {
//                            FunctionProfile function = (FunctionProfile) e4.nextElement();
//                            if (function != null) {
//                                writeBeginObject(xwriter, 16);
//                                /*
//                                 * @@@ Commented out as is questionable
//                                 * functionality. Add back in for legacy
//                                 * support.
//                                 * writeFunctionName(xwriter,function.getMappingName());
//                                 */
//                                writeInt(xwriter, 11, function.getMappingID());
//                                //Build group string.
//                                groupsStringBuffer.delete(
//                                                          0,
//                                                          groupsStringBuffer.length());
//                                int[] groupIDs = function.getGroups();
//                                for (int i = 0; i < groupIDs.length; i++) {
//                                    if (i == 0)
//                                        groupsStringBuffer.append(groupNames[groupIDs[i]]);
//                                    else
//                                        groupsStringBuffer.append(":"
//                                                + groupNames[groupIDs[i]]);
//                                }
//                                writeString(xwriter, 14,
//                                            groupsStringBuffer.toString());
//                                writeDouble(
//                                            xwriter,
//                                            0,
//                                            function.getInclusivePercentValue(metricID));
//                                writeDouble(
//                                            xwriter,
//                                            1,
//                                            function.getInclusiveValue(metricID));
//                                writeDouble(
//                                            xwriter,
//                                            2,
//                                            function.getExclusivePercentValue(metricID));
//                                writeDouble(
//                                            xwriter,
//                                            3,
//                                            function.getExclusiveValue(metricID));
//                                writeDouble(xwriter, 5,
//                                            function.getNumberOfCalls());
//                                writeDouble(xwriter, 6,
//                                            function.getNumberOfSubRoutines());
//                                writeDouble(
//                                            xwriter,
//                                            4,
//                                            function.getInclusivePerCall(metricID));
//                                writeEndObject(xwriter, 16);
//                            }
//                        }
//                        //Write out user event data for this thread.
//                        if (userEvents != null) {
//                            for (Enumeration e4 = userEvents.elements(); e4.hasMoreElements();) {
//                                UserEventProfile userevent = (UserEventProfile) e4.nextElement();
//                                if (userevent != null) {
//                                    writeBeginObject(xwriter, 17);
//                                    /*
//                                     * @@@ Commented out as is questionable
//                                     * functionality. Add back in for legacy
//                                     * support.
//                                     * writeString(xwriter,15,userevent.getUserEventName());
//                                     */
//                                    writeInt(xwriter, 12,
//                                             userevent.getMappingID());
//                                    writeInt(
//                                             xwriter,
//                                             13,
//                                             userevent.getUserEventNumberValue());
//                                    writeDouble(
//                                                xwriter,
//                                                7,
//                                                userevent.getUserEventMaxValue());
//                                    writeDouble(
//                                                xwriter,
//                                                8,
//                                                userevent.getUserEventMinValue());
//                                    writeDouble(
//                                                xwriter,
//                                                9,
//                                                userevent.getUserEventMeanValue());
//                                    //writeDouble(xwriter,10,userevent.getUserEventStdDevValue());@@@Commented
//                                    // out due to lack of ParaProf data support.
//                                    // Should probably add methods which
//                                    //generate this data. @@@
//                                    writeBeginObject(xwriter, 17);
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//
//            xwriter.write("\t</Pprof>", 0, ("\t</Pprof>").length());
//            xwriter.newLine();
//            xwriter.newLine();
//
//            /*
//             * @@@ Commented out as is questionable functionality. Add back in
//             * for legacy support. xwriter.write("\t <totalfunsummary>", 0, ("\t
//             * <totalfunsummary>").length()); xwriter.newLine();
//             */
//
//            /*
//             * @@@Commented out due to lack of ParaProf data support. Should
//             * probably add methods which generate this data. @@@ //Write out
//             * total information. for(Enumeration e =
//             * trialData.getMapping(0).elements(); e.hasMoreElements() ;){
//             * globalMappingElement = (GlobalMappingElement) e.nextElement(); if
//             * (globalMappingElement!=null){ writeBeginObject(xwriter,18);
//             * writeFunctionName(xwriter,globalMappingElement.getMappingName());
//             * writeInt(xwriter,11,globalMappingElement.getMappingID()); //Build
//             * group string.
//             * groupsStringBuffer.delete(0,groupsStringBuffer.length()); int[]
//             * groupIDs = globalMappingElement.getGroups(); for(int i=0;i
//             * <groupIDs.length;i++){
//             * groupsStringBuffer=+groupNames[groupIDs[i]];}
//             * writeString(xwriter,14,groupsStringBuffer.toString());
//             * writeDouble(xwriter,0,globalMappingElement.getTotalInclusivePercentValue(metricID));
//             * writeDouble(xwriter,1,globalMappingElement.getTotalInclusiveValue(metricID));
//             * writeDouble(xwriter,2,globalMappingElement.getTotalExclusivePercentValue(metricID));
//             * writeDouble(xwriter,3,globalMappingElement.getTotalExclusiveValue(metricID));
//             * writeDouble(xwriter,5,globalMappingElement.getTotalNumberOfCalls());
//             * writeDouble(xwriter,6,globalMappingElement.getTotalNummberOfSubroutines());
//             * writeDouble(xwriter,4,globalMappingElement.getTotalInclPCall());
//             * writeEndObject(xwriter,18); } }
//             */
//
//            /*
//             * @@@ Commented out as is questionable functionality. Add back in
//             * for legacy support. xwriter.write("\t </totalfunsummary>", 0,
//             * ("\t </totalfunsummary>").length()); xwriter.newLine();
//             * xwriter.newLine();
//             */
//
//            /*
//             * @@@ Commented out as is questionable functionality. Add back in
//             * for legacy support. xwriter.write("\t <meanfunsummary>", 0, ("\t
//             * <meanfunsummary>").length()); xwriter.newLine();
//             * 
//             * //Write out mean information. for(Enumeration e =
//             * trialData.getMapping(0).elements(); e.hasMoreElements() ;){
//             * GlobalMappingElement globalMappingElement =
//             * (GlobalMappingElement) e.nextElement(); if
//             * (globalMappingElement!=null){ writeBeginObject(xwriter,19);
//             * writeFunctionName(xwriter,globalMappingElement.getMappingName());
//             * writeInt(xwriter,11,globalMappingElement.getMappingID()); //Build
//             * group string.
//             * groupsStringBuffer.delete(0,groupsStringBuffer.length()); int[]
//             * groupIDs = globalMappingElement.getGroups(); for(int i=0;i
//             * <groupIDs.length;i++){ if(i==0)
//             * groupsStringBuffer.append(groupNames[groupIDs[i]]); else
//             * groupsStringBuffer.append(":"+groupNames[groupIDs[i]]); }
//             * writeString(xwriter,14,groupsStringBuffer.toString());
//             * writeDouble(xwriter,0,globalMappingElement.getMeanInclusivePercentValue(metricID));
//             * writeDouble(xwriter,1,globalMappingElement.getMeanInclusiveValue(metricID));
//             * writeDouble(xwriter,2,globalMappingElement.getMeanExclusivePercentValue(metricID));
//             * writeDouble(xwriter,3,globalMappingElement.getMeanExclusiveValue(metricID));
//             * writeDouble(xwriter,5,globalMappingElement.getMeanNumberOfCalls());
//             * writeDouble(xwriter,6,globalMappingElement.getMeanNumberOfSubRoutines());
//             * writeDouble(xwriter,4,globalMappingElement.getMeanUserSecPerCall(metricID)); } }
//             * 
//             * xwriter.write("\t </meanfunsummary>", 0, ("\t
//             * </meanfunsummary>").length()); xwriter.newLine();
//             */
//
//            xwriter.write("    </Onetrial>", 0, ("    </Onetrial>").length());
//            xwriter.newLine();
//
//            xwriter.write("</Trials>", 0, ("</Trials>").length());
//            xwriter.newLine();
//            xwriter.close();
//
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
    //}

//    public void writeComputationModel(BufferedWriter writer, int node,
//            int context, int thread) {
//
//        try {
//            writer.write("\t<ComputationModel>", 0,
//                         ("\t<ComputationModel>").length());
//            writer.newLine();
//
//            String tmpString;
//            tmpString = "\t   <node level=\"Top\" statis_info=\"sum\">" + node
//                    + "</node>";
//            writer.write(tmpString, 0, tmpString.length());
//            writer.newLine();
//            tmpString = "\t   <context level=\"Secondary\" statis_info=\"contextPnode\">"
//                    + context + "</context>";
//            writer.write(tmpString, 0, tmpString.length());
//            writer.newLine();
//            tmpString = "\t   <thread level=\"Lowest\" statis_info=\"threadPcontext\">"
//                    + thread + "</thread>";
//            writer.write(tmpString, 0, tmpString.length());
//            writer.newLine();
//            writer.write("\t</ComputationModel>", 0,
//                         "\t</ComputationModel>".length());
//            writer.newLine();
//        } catch (Exception e) {
//            xmlWriteError.location = "writeComputationModel(BufferedWriter writer, int node, int context, int thread)";
//            UtilFncs.systemError(xmlWriteError, null, null);
//        }
//    }
//
//    public void writeIDs(BufferedWriter writer, int node, int context,
//            int thread) {
//
//        try {
//            String tmpString;
//            tmpString = "\t   <nodeID>" + node + "</nodeID>";
//            writer.write(tmpString, 0, tmpString.length());
//            writer.newLine();
//            tmpString = "\t   <contextID>" + context + "</contextID>";
//            writer.write(tmpString, 0, tmpString.length());
//            writer.newLine();
//            tmpString = "\t   <threadID>" + thread + "</threadID>";
//            writer.write(tmpString, 0, tmpString.length());
//            writer.newLine();
//
//        } catch (Exception e) {
//            xmlWriteError.location = "writeIDs(BufferedWriter writer, int node, int context, int thread)";
//            UtilFncs.systemError(xmlWriteError, null, null);
//        }
//
//    }
//
//    public void writeBeginObject(BufferedWriter writer, int type) {
//        try {
//            writer.write(this.lookupBegin[type], 0,
//                         this.lookupBegin[type].length());
//            writer.newLine();
//        } catch (Exception e) {
//            xmlWriteError.location = "writeBeginObject(BufferedWriter writer, int type)";
//            UtilFncs.systemError(xmlWriteError, null, null);
//        }
//    }
//
//    public void writeEndObject(BufferedWriter writer, int type) {
//        try {
//            writer.write(this.lookupEnd[type], 0, this.lookupEnd[type].length());
//            writer.newLine();
//        } catch (Exception e) {
//            xmlWriteError.location = "writeEndObject(BufferedWriter writer, int type)";
//            UtilFncs.systemError(xmlWriteError, null, null);
//        }
//    }
//
//    public void writeFunctionName(BufferedWriter writer, String funname) {
//        try {
//            String tmpString;
//            funname = replace(funname, "&", "&amp;");
//            funname = replace(funname, "<", "&lt;");
//            funname = replace(funname, ">", "&gt;");
//            tmpString = "\t\t<funname>" + funname + "</funname>";
//            writer.write(tmpString, 0, tmpString.length());
//            writer.newLine();
//        } catch (Exception e) {
//            xmlWriteError.location = "writeFunctionName(BufferedWriter writer, String funname)";
//            UtilFncs.systemError(xmlWriteError, null, null);
//        }
//
//    }
//
//    public void writeNameIDMap(BufferedWriter writer, String funname, int id) {
//        try {
//            String tmpString;
//            funname = replace(funname, "&", "&amp;");
//            funname = replace(funname, "<", "&lt;");
//            funname = replace(funname, ">", "&gt;");
//            tmpString = "\t\t<nameid>" + "\"" + funname + "\"" + id
//                    + "</nameid>";
//            writer.write(tmpString, 0, tmpString.length());
//            writer.newLine();
//        } catch (Exception e) {
//            xmlWriteError.location = "writeFunctionName(BufferedWriter writer, String funname)";
//            UtilFncs.systemError(xmlWriteError, null, null);
//        }
//
//    }
//
//    public void writeString(BufferedWriter writer, int type, String s1) {
//        try {
//            String s2 = this.lookupBegin[type] + s1 + this.lookupEnd[type];
//            writer.write(s2, 0, s2.length());
//            writer.newLine();
//        } catch (Exception e) {
//            xmlWriteError.location = "writeString(BufferedWriter writer, int type, String s1)";
//            UtilFncs.systemError(xmlWriteError, null, null);
//        }
//    }
//
//    public void writeInt(BufferedWriter writer, int type, int i) {
//        try {
//            String s = this.lookupBegin[type] + i + this.lookupEnd[type];
//            writer.write(s, 0, s.length());
//            writer.newLine();
//        } catch (Exception e) {
//            xmlWriteError.location = "writeInt(BufferedWriter writer, int type, int i)";
//            UtilFncs.systemError(xmlWriteError, null, null);
//        }
//
//    }
//
//    public void writeDouble(BufferedWriter writer, int type, double d) {
//        try {
//            String s = this.lookupBegin[type] + d + this.lookupEnd[type];
//            writer.write(s, 0, s.length());
//            writer.newLine();
//        } catch (Exception e) {
//            xmlWriteError.location = "writeDouble(BufferedWriter writer, int type, double d)";
//            UtilFncs.systemError(xmlWriteError, null, null);
//        }
//
//    }
//
//    public String getMetric(String str) {
//        if (str.length() > 26)
//            return (str.substring(26, str.length()));
//        else
//            return new String("");
//    }
//
//    public String replace(String str, String lstr, String rstr) {
//        String tempStr = "";
//        int i;
//        while ((i = str.indexOf(lstr)) != -1) {
//            if (i > 0)
//                tempStr += str.substring(0, i);
//            tempStr += rstr;
//            str = str.substring(i + 1);
//        }
//        tempStr += str;
//        return tempStr;
//    }
//
//    public String getFunAmt(String inString) {
//        try {
//            String tmpString;
//            StringTokenizer funAmtTokenizer = new StringTokenizer(inString,
//                    " \t\n\r");
//            tmpString = funAmtTokenizer.nextToken();
//            return tmpString;
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//        return null;
//    }

    //####################################
    //Instance data.
    //####################################
//    private Trial trial = null;
//
//    private String[] lookupBegin = { "\t\t<inclperc>", "\t\t<inclutime>",
//            "\t\t<exclperc>", "\t\t<exclutime>", //0,1,2,3
//            "\t\t<inclutimePcall>", //4
//            "\t\t<call>", "\t\t<subrs>", //5,6
//            "\t\t<maxvalue>", "\t\t<minvalue>", "\t\t<meanvalue>",
//            "\t\t<stddevvalue>", //7,8,9,10
//            "\t\t<funID>", "\t\t<ueID>", "\t\t<numofsamples>", //11,12,13
//            "\t\t<fungroup>", "\t\t<uename>", //14,15
//            "\t   <instrumentedobj>", "\t   <userevent>",
//            "\t   <totalfunction>", //16,17,18
//            "\t   <meanfunction>", //19
//            "\t   <funnameidmap>", "\t   <uenameidmap>" }; //20,21
//    private String[] lookupEnd = { "</inclperc>", "</inclutime>",
//            "</exclperc>", "</exclutime>", //0,1,2,3
//            "</inclutimePcall>", //4
//            "</call>", "</subrs>", //5,6
//            "</maxvalue>", "</minvalue>", "</meanvalue>", "</stddevvalue>", //7,8,9,10
//            "</funID>", "</ueID>", "</numofsamples>", //11,12,13
//            "<fungroup>", "<uename>", //14,15
//            "\t   </instrumentedobj>", "\t   </userevent>",
//            "\t   </totalfunction>", //16,17,18
//            "\t   </meanfunction>", //19
//            "\t   </funnameidmap>", "\t   </uenameidmap>" }; //20,21

    //######
    //Error outupt.
    //######
    //For errors writing to the xml file.
//    private ParaProfError xmlWriteError = new ParaProfError("",
//            "XML Write Error: See console for details.",
//            "An error occurred while writing XML file. Operation aborted!",
//            "Note: Dependent operations also aborted.", null, false);

    //####################################
    //End - Instance data.
    //####################################
}