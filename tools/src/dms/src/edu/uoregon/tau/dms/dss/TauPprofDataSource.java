/*
 * Name: TauPprofDataSource.java Author: Robert Bell Description:
 */

/*
 * To do: 1) Add some sanity checks to make sure that multiple metrics really do
 * belong together. For example, wrap the creation of nodes, contexts, threads,
 * global mapping elements, and the like so that they do not occur after the
 * first metric has been loaded. This will not of course ensure 100% that the
 * data is consistent, but it will at least prevent the worst cases.
 */

package edu.uoregon.tau.dms.dss;

import java.io.*;
import java.util.*;

public class TauPprofDataSource extends DataSource {

    //####################################
    //Public Section.
    //####################################

    public TauPprofDataSource(Object initializeObject) {
        super();
        this.setMetrics(new Vector());
        this.initializeObject = initializeObject;
    }

    private Object initializeObject;

    public void cancelLoad() {
        return;
    }

    public int getProgress() {
        return 0;
    }

    public void load() {
        try {
            //######
            //Frequently used items.
            //######
            int metric = 0;

            Function function = null;
            UserEvent userEvent = null;
            FunctionProfile functionProfile = null;
            UserEventProfile userEventProfile = null;

            Node node = null;
            Context context = null;
            edu.uoregon.tau.dms.dss.Thread thread = null;
            
            meanData = new Thread(-1, -1, -1, 1);
            meanData.initializeFunctionList(0);
            FunctionProfile meanProfile = null;
            
            int nodeID = -1;
            int contextID = -1;
            int threadID = -1;

            String inputString = null;
            String s1 = null;
            String s2 = null;

            String tokenString;
            String groupNamesString = null;
            StringTokenizer genericTokenizer;

            //A loop counter.
            int bSDCounter = 0;

            Vector v = null;
            File[] files = null;
            //######
            //End - Frequently used items.
            //######
            v = (Vector) initializeObject;
            this.setFirstMetric(true);
            for (Enumeration e = v.elements(); e.hasMoreElements();) {
                files = (File[]) e.nextElement();
                System.out.println("Processing data file, please wait ......");
                long time = System.currentTimeMillis();

                FileInputStream fileIn = new FileInputStream(files[0]);
                InputStreamReader inReader = new InputStreamReader(fileIn);
                BufferedReader br = new BufferedReader(inReader);

                //####################################
                //First Line
                //####################################
                //This line is not required. Check to make sure that it is
                // there however.
                inputString = br.readLine();
                if (inputString == null)
                    return;
                bSDCounter++;
                //####################################
                //End - First Line
                //####################################

                //####################################
                //Second Line
                //####################################
                //This is an important line.
                inputString = br.readLine();
                //Set the metric name.
                String metricName = getMetricName(inputString);

                //Need to call increaseVectorStorage() on all objects that
                // require it.
                this.getTrialData().increaseVectorStorage();
                //Only need to call addDefaultToVectors() if not the first run.
                if (!(this.firstMetric())) {

                    for (Iterator i1 = this.getTrialData().getFunctions(); i1.hasNext();) {
                        Function tmpFunc = (Function) i1.next();
                        tmpFunc.incrementStorage();
                    }

                    this.meanData.incrementStorage();

                    for (Enumeration e3 = this.getNCT().getNodes().elements(); e3.hasMoreElements();) {
                        node = (Node) e3.nextElement();
                        for (Enumeration e4 = node.getContexts().elements(); e4.hasMoreElements();) {
                            context = (Context) e4.nextElement();
                            for (Enumeration e5 = context.getThreads().elements(); e5.hasMoreElements();) {
                                thread = (edu.uoregon.tau.dms.dss.Thread) e5.nextElement();
                                thread.incrementStorage();
                                for (Enumeration e6 = thread.getFunctionList().elements(); e6.hasMoreElements();) {
                                    FunctionProfile ref = (FunctionProfile) e6.nextElement();
                                    //Only want to add an element if this
                                    // mapping existed on this thread.
                                    //Check for this.
                                    if (ref != null)
                                        ref.incrementStorage();
                                }
                            }
                        }
                    }
                }

                //Now set the metric name.
                if (metricName == null)
                    metricName = new String("Time");

                //		System.out.println("Metric name is: " + metricName);

                metric = this.getNumberOfMetrics();
                this.addMetric(metricName);

                bSDCounter++;
                //####################################
                //End - Second Line
                //####################################

                //####################################
                //Third Line
                //####################################
                //Do not need the third line.
                inputString = br.readLine();
                if (inputString == null)
                    return;
                bSDCounter++;
                //####################################
                //End - Third Line
                //####################################

                while ((inputString = br.readLine()) != null) {
                    genericTokenizer = new StringTokenizer(inputString, " \t\n\r");

                    int lineType = -1;
                    /*
                     * (0) t-exclusive (1) t-inclusive (2) m-exclusive (3)
                     * m-inclusive (4) exclusive (5) inclusive (6) userevent
                     */

                    //Determine the lineType.
                    if ((inputString.charAt(0)) == 't') {
                        if (checkForExcInc(inputString, true, false))
                            lineType = 0;
                        else
                            lineType = 1;
                    } else if ((inputString.charAt(0)) == 'm') {
                        if (checkForExcInc(inputString, true, false))
                            lineType = 2;
                        else
                            lineType = 3;
                    } else if (checkForExcInc(inputString, true, true))
                        lineType = 4;
                    else if (checkForExcInc(inputString, false, true))
                        lineType = 5;
                    else if (noue(inputString))
                        lineType = 6;

                    //Common things to grab
                    if ((lineType != 6) && (lineType != -1)) {
                        this.getFunctionDataLine1(inputString);
                        function = this.getTrialData().addFunction(functionDataLine1.s0, 1);

                        // get/create the FunctionProfile for mean
                        meanProfile = meanData.getFunctionProfile(function);
                        if (meanProfile == null) {
                            meanProfile = new FunctionProfile(function, 1);
                            meanData.addFunctionProfile(meanProfile, function.getID());
                        }
                        function.setMeanProfile(meanProfile);

                    }

                    switch (lineType) {
                    case 0:
                        if (this.firstMetric()) {
                            //Grab the group names.
                            groupNamesString = getGroupNames(inputString);
                            if (groupNamesString != null) {
                                StringTokenizer st = new StringTokenizer(groupNamesString, " |");
                                while (st.hasMoreTokens()) {
                                    String tmpString = st.nextToken();
                                    if (tmpString != null) {
                                        Group group = this.getTrialData().addGroup(tmpString);
                                        function.addGroup(group);
                                    }
                                }
                            }
                        }

                        function.setTotalExclusive(metric, functionDataLine1.d0);
                        function.setTotalExclusivePercent(metric, functionDataLine1.d1);
                        break;
                    case 1:
                        function.setTotalInclusive(metric, functionDataLine1.d0);
                        function.setTotalInclusivePercent(metric, functionDataLine1.d1);
                        break;
                    case 2:
                        //Now set the values correctly.
                        if ((this.getTrialData().getMaxMeanExclusiveValue(metric)) < functionDataLine1.d0) {
                            this.getTrialData().setMaxMeanExclusiveValue(metric,
                                    functionDataLine1.d0);
                        }
                        if ((this.getTrialData().getMaxMeanExclusivePercentValue(metric)) < functionDataLine1.d1) {
                            this.getTrialData().setMaxMeanExclusivePercentValue(metric,
                                    functionDataLine1.d1);
                        }

                        meanProfile.setExclusive(metric, functionDataLine1.d0);
                        meanProfile.setExclusivePercent(metric, functionDataLine1.d1);
                        break;
                    case 3:
                        //Now set the values correctly.
                        if ((this.getTrialData().getMaxMeanInclusiveValue(metric)) < functionDataLine1.d0) {
                            this.getTrialData().setMaxMeanInclusiveValue(metric,
                                    functionDataLine1.d0);
                        }
                        if ((this.getTrialData().getMaxMeanInclusivePercentValue(metric)) < functionDataLine1.d1) {
                            this.getTrialData().setMaxMeanInclusivePercentValue(metric,
                                    functionDataLine1.d1);
                        }

                        meanProfile.setInclusive(metric, functionDataLine1.d0);
                        meanProfile.setInclusivePercent(metric, functionDataLine1.d1);

                        System.out.println ("value: " + functionDataLine1.d1);

                        
                        //Set number of calls/subroutines/usersec per call.
                        inputString = br.readLine();

                        this.getFunctionDataLine2(inputString);

                        //Set the values.
                        meanProfile.setNumCalls(functionDataLine2.d0);
                        meanProfile.setNumSubr(functionDataLine2.d1);
                        meanProfile.setInclusivePerCall(metric, functionDataLine2.d2);

                        //Set the max values.
                        if ((this.getTrialData().getMaxMeanNumberOfCalls()) < functionDataLine2.d0)
                            this.getTrialData().setMaxMeanNumberOfCalls(functionDataLine2.d0);

                        if ((this.getTrialData().getMaxMeanNumberOfSubRoutines()) < functionDataLine2.d1)
                            this.getTrialData().setMaxMeanNumberOfSubRoutines(functionDataLine2.d1);

                        if ((this.getTrialData().getMaxMeanInclusivePerCall(metric)) < functionDataLine2.d2)
                            this.getTrialData().setMaxMeanInclusivePerCall(metric,
                                    functionDataLine2.d2);

                        function.setMeanValuesSet(true);
                        break;
                    case 4:
                        if ((function.getMaxExclusive(metric)) < functionDataLine1.d0)
                            function.setMaxExclusive(metric, functionDataLine1.d0);
                        if ((function.getMaxExclusivePercent(metric)) < functionDataLine1.d1)
                            function.setMaxExclusivePercent(metric, functionDataLine1.d1);

                        //Get the node,context,thread.
                        int[] array = this.getNCT(inputString);
                        nodeID = array[0];
                        contextID = array[1];
                        threadID = array[2];

                        node = this.getNCT().getNode(nodeID);
                        if (node == null)
                            node = this.getNCT().addNode(nodeID);
                        context = node.getContext(contextID);
                        if (context == null)
                            context = node.addContext(contextID);
                        thread = context.getThread(threadID);
                        if (thread == null) {
                            thread = context.addThread(threadID);
                            thread.initializeFunctionList(this.getTrialData().getNumFunctions());
                        }

                        functionProfile = thread.getFunctionProfile(function);

                        if (functionProfile == null) {
                            functionProfile = new FunctionProfile(function);
                            thread.addFunctionProfile(functionProfile, function.getID());
                        }
                        functionProfile.setExclusive(metric, functionDataLine1.d0);
                        functionProfile.setExclusivePercent(metric, functionDataLine1.d1);
                        //Now check the max values on this thread.
                        if ((thread.getMaxExclusive(metric)) < functionDataLine1.d0)
                            thread.setMaxExclusive(metric, functionDataLine1.d0);
                        if ((thread.getMaxExclusivePercent(metric)) < functionDataLine1.d1)
                            thread.setMaxExclusivePercent(metric, functionDataLine1.d1);
                        break;
                    case 5:
                        if ((function.getMaxInclusive(metric)) < functionDataLine1.d0)
                            function.setMaxInclusive(metric, functionDataLine1.d0);

                        if ((function.getMaxInclusivePercent(metric)) < functionDataLine1.d1)
                            function.setMaxInclusivePercent(metric, functionDataLine1.d1);

                        thread = this.getNCT().getThread(nodeID, contextID, threadID);
                        functionProfile = thread.getFunctionProfile(function);

                        functionProfile.setInclusive(metric, functionDataLine1.d0);
                        functionProfile.setInclusivePercent(metric, functionDataLine1.d1);
                        if ((thread.getMaxInclusive(metric)) < functionDataLine1.d0)
                            thread.setMaxInclusive(metric, functionDataLine1.d0);
                        if ((thread.getMaxInclusivePercent(metric)) < functionDataLine1.d1)
                            thread.setMaxInclusivePercent(metric, functionDataLine1.d1);

                        //Get the number of calls and number of sub routines
                        inputString = br.readLine();
                        this.getFunctionDataLine2(inputString);

                        //Set the values.
                        functionProfile.setNumCalls(functionDataLine2.d0);
                        functionProfile.setNumSubr(functionDataLine2.d1);
                        functionProfile.setInclusivePerCall(metric, functionDataLine2.d2);

                        //Set the max values.
                        if (function.getMaxNumCalls() < functionDataLine2.d0)
                            function.setMaxNumCalls(functionDataLine2.d0);
                        if (thread.getMaxNumCalls() < functionDataLine2.d0)
                            thread.setMaxNumCalls(functionDataLine2.d0);

                        if (function.getMaxNumSubr() < functionDataLine2.d1)
                            function.setMaxNumSubr(functionDataLine2.d1);
                        if (thread.getMaxNumSubr() < functionDataLine2.d1)
                            thread.setMaxNumSubr(functionDataLine2.d1);

                        if (function.getMaxInclusivePerCall(metric) < functionDataLine2.d2)
                            function.setMaxInclusivePerCall(metric, functionDataLine2.d2);
                        if (thread.getMaxInclusivePerCall(metric) < functionDataLine2.d2)
                            thread.setMaxInclusivePerCall(metric, functionDataLine2.d2);
                        break;
                    case 6:
                        //Just ignore the string if this is not the first
                        // check.
                        //Assuming is that user events do not change for each
                        // counter value.
                        if (this.firstMetric()) {
                            //The first line will be the user event heading ...
                            // skip it.
                            br.readLine();
                            //Now that we know how many user events to expect,
                            // we can grab that number of lines.
                            //Note that inputString is still set the the line
                            // before the heading which is what we want.
                            int numberOfLines = getNumberOfUserEvents(inputString);
                            for (int j = 0; j < numberOfLines; j++) {
                                //Initialize the user list for this thread.
                                if (j == 0)
                                    (this.getNCT().getThread(nodeID, contextID, threadID)).initializeUsereventList(this.getTrialData().getNumUserEvents());

                                s1 = br.readLine();
                                s2 = br.readLine();
                                getUserEventData(s1);
                                // System.out.println("noc:"+usereventDataLine.i0+"min:"+usereventDataLine.d1+"max:"+usereventDataLine.d0+"mean:"+usereventDataLine.d2);

                                if (usereventDataLine.i0 != 0) {
                                    userEvent = this.getTrialData().addUserEvent(
                                            usereventDataLine.s0);
                                    userEventProfile = thread.getUserEvent(userEvent.getID());

                                    if (userEventProfile == null) {
                                        userEventProfile = new UserEventProfile(userEvent);
                                        thread.addUserEvent(userEventProfile, userEvent.getID());
                                    }

                                    userEventProfile.setUserEventNumberValue(usereventDataLine.i0);
                                    userEventProfile.setUserEventMinValue(usereventDataLine.d1);
                                    userEventProfile.setUserEventMaxValue(usereventDataLine.d0);
                                    userEventProfile.setUserEventMeanValue(usereventDataLine.d2);
                                    userEventProfile.setUserEventSumSquared(usereventDataLine.d3);

                                    if ((userEvent.getMaxUserEventNumberValue()) < usereventDataLine.i0)
                                        userEvent.setMaxUserEventNumberValue(usereventDataLine.i0);
                                    if ((userEvent.getMaxUserEventMaxValue()) < usereventDataLine.d0)
                                        userEvent.setMaxUserEventMaxValue(usereventDataLine.d0);
                                    if ((userEvent.getMaxUserEventMinValue()) < usereventDataLine.d1)
                                        userEvent.setMaxUserEventMinValue(usereventDataLine.d1);
                                    if ((userEvent.getMaxUserEventMeanValue()) < usereventDataLine.d2)
                                        userEvent.setMaxUserEventMeanValue(usereventDataLine.d2);

                                    if (userEvent.getMaxUserEventSumSquared() < usereventDataLine.d3)
                                        userEvent.setMaxUserEventSumSquared(usereventDataLine.d3);

                                }
                            }
                            //Now set the userevents flag.
                            setUserEventsPresent(true);
                        }
                        break;
                    default:
                        if (UtilFncs.debug) {
                            System.out.println("Skipping line: " + bSDCounter);
                        }
                        break;
                    }

                    //Increment the loop counter.
                    bSDCounter++;
                }

                //Close the file.
                br.close();

                if (UtilFncs.debug) {
                    System.out.println("The total number of threads is: "
                            + this.getNCT().getTotalNumberOfThreads());
                    System.out.println("The number of mappings is: "
                            + this.getTrialData().getNumFunctions());
                    System.out.println("The number of user events is: "
                            + this.getTrialData().getNumUserEvents());
                }

                //Set firstRead to false.
                this.setFirstMetric(false);
                
                meanData.setThreadDataAllMetrics();

                time = (System.currentTimeMillis()) - time;
                System.out.println("Done processing data file!");
                System.out.println("Time to process file (in milliseconds): " + time);
            }

            //System.out.println("Processing callpath data ...");
            if (CallPathUtilFuncs.isAvailable(getTrialData().getFunctions())) {
                setCallPathDataPresent(true);
                CallPathUtilFuncs.buildRelations(getTrialData());
            } else {
                //System.out.println("No callpath data found.");
            }
            //System.out.println("Done - Processing callpath data!");
        } catch (Exception e) {
            UtilFncs.systemError(
                    new ParaProfError(
                            this.toString() + ": run()",
                            null,
                            "An error occurred while trying to load!\nExpected format to be of type \"pprof\".",
                            "Please check for the correct file type or a corrupt file.", e, null,
                            null, null, false, false, false), null, null);
            if (this.debug())
                this.outputDebugMessage(this.toString()
                        + ": run()\nAn error occurred while trying to load!\nExpected format to be of type \"profiles\".");
        }
    }

    public void outputDebugMessage(String debugMessage) {
        UtilFncs.objectDebug.outputToFile(this.toString() + "\n" + debugMessage);
    }

    public String toString() {
        return this.getClass().getName();
    }

    public static void main(String[] args) {
    }

    //####################################
    //End - Public Section.
    //####################################

    //####################################
    //Private Section.
    //####################################

    //######
    //Pprof.dat string processing methods.
    //######
    private boolean noue(String s) {
        int stringPosition = 0;
        char tmpChar = s.charAt(stringPosition);
        while (tmpChar != '\u0020') {
            stringPosition++;
            tmpChar = s.charAt(stringPosition);
        }
        stringPosition++;
        tmpChar = s.charAt(stringPosition);
        if (tmpChar == 'u')
            return true;
        else
            return false;
    }

    private boolean checkForExcInc(String inString, boolean exclusive, boolean checkString) {
        boolean result = false;

        try {
            //In this function I need to be careful. If the mapping name
            // contains "excl", I
            //might interpret this line as being the exclusive line when in
            // fact it is not.

            if (checkString) {
                StringTokenizer checkTokenizer = new StringTokenizer(inString, " ");
                String tmpString2 = checkTokenizer.nextToken();
                if ((tmpString2.indexOf(",")) == -1)
                    return result;
            }

            // first, count the number of double-quotes to determine if the
            // function contains a double-quote
            int quoteCount = 0;
            for (int i = 0; i < inString.length(); i++) {
                if (inString.charAt(i) == '"')
                    quoteCount++;
            }

            StringTokenizer st2;

            String tmpString;
            if (quoteCount == 2 || quoteCount == 4) { // assume all is well
                StringTokenizer checkQuotesTokenizer = new StringTokenizer(inString, "\"");

                //Need to get the third token. Could do it in a loop, just as
                // quick this way.
                tmpString = checkQuotesTokenizer.nextToken();
                tmpString = checkQuotesTokenizer.nextToken();
                tmpString = checkQuotesTokenizer.nextToken();

            } else {

                // there is a quote in the name of the timer/function
                // we assume that TAU_GROUP="..." is there, so the end of the
                // name must be
                // at quoteCount - 2
                int count = 0;
                int i = 0;
                while (count < quoteCount - 2 && i < inString.length()) {
                    if (inString.charAt(i) == '"')
                        count++;
                    i++;
                }
                tmpString = inString.substring(i + 1);
            }

            //Ok, now, the string in tmpString should include at least "excl"
            // or "incl", and
            //also, the first token should be either "excl" or "incl".
            StringTokenizer checkForExclusiveTokenizer = new StringTokenizer(tmpString, " \t\n\r");
            tmpString = checkForExclusiveTokenizer.nextToken();

            //At last, do the check.
            if (exclusive) {
                if (tmpString.equals("excl"))
                    result = true;
            } else {
                if (tmpString.equals("incl"))
                    result = true;
            }
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SSD04");
        }
        return result;
    }

    private void getFunctionDataLine1(String string) {
        try {

            // first, count the number of double-quotes to determine if the
            // function contains a double-quote
            int quoteCount = 0;
            for (int i = 0; i < string.length(); i++) {
                if (string.charAt(i) == '"')
                    quoteCount++;
            }

            StringTokenizer st2;

            if (quoteCount == 2 || quoteCount == 4) { // assume all is well
                StringTokenizer st1 = new StringTokenizer(string, "\"");
                st1.nextToken();
                functionDataLine1.s0 = st1.nextToken(); //Name

                st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");

            } else {
                // there is a quote in the name of the timer/function
                // we assume that TAU_GROUP="..." is there, so the end of the
                // name must be
                // at quoteCount - 2
                int count = 0;
                int i = 0;

                int firstQuote = -1;
                while (count < quoteCount - 2 && i < string.length()) {
                    if (string.charAt(i) == '"') {
                        if (firstQuote == -1)
                            firstQuote = i;
                        count++;
                    }
                    i++;
                }

                functionDataLine1.s0 = string.substring(firstQuote + 1, i - 1); // get
                // the
                // name
                st2 = new StringTokenizer(string.substring(i + 1), " \t\n\r");
            }

            st2.nextToken();
            functionDataLine1.d0 = Double.parseDouble(st2.nextToken()); //Value
            functionDataLine1.d1 = Double.parseDouble(st2.nextToken()); //Percent
            // value
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SSD08");
        }
    }

    private void getFunctionDataLine2(String string) {
        try {
            StringTokenizer getMappingIDTokenizer = new StringTokenizer(string, " \t\n\r");
            getMappingIDTokenizer.nextToken();
            getMappingIDTokenizer.nextToken();
            getMappingIDTokenizer.nextToken();

            // number of calls
            functionDataLine2.d0 = Double.parseDouble(getMappingIDTokenizer.nextToken()); 

            // number of subroutines
            functionDataLine2.d1 = Double.parseDouble(getMappingIDTokenizer.nextToken()); 
            
            // inclusive per call
            functionDataLine2.d2 = Double.parseDouble(getMappingIDTokenizer.nextToken()); 
            
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SSD10");
        }
    }

    private void getUserEventData(String string) {
        try {

            // first, count the number of double-quotes to determine if the
            // user event contains a double-quote
            int quoteCount = 0;
            for (int i = 0; i < string.length(); i++) {
                if (string.charAt(i) == '"')
                    quoteCount++;
            }

            StringTokenizer st2;

            if (quoteCount == 2) { // proceed as usual
                StringTokenizer st1 = new StringTokenizer(string, "\"");
                String trash = st1.nextToken();
                usereventDataLine.s0 = st1.nextToken();
                st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
            } else {

                // there is a quote in the name of the user event
                int count = 0;
                int i = 0;

                int firstQuote = -1;
                while (count < quoteCount && i < string.length()) {
                    if (string.charAt(i) == '"') {
                        if (firstQuote == -1)
                            firstQuote = i;
                        count++;
                    }
                    i++;
                }

                usereventDataLine.s0 = string.substring(firstQuote + 1, i - 1);
                st2 = new StringTokenizer(string.substring(i + 1), " \t\n\r");
            }

            usereventDataLine.i0 = (int) Double.parseDouble(st2.nextToken()); //Number
            // of
            // calls.
            usereventDataLine.d0 = Double.parseDouble(st2.nextToken()); //Max
            usereventDataLine.d1 = Double.parseDouble(st2.nextToken()); //Min
            usereventDataLine.d2 = Double.parseDouble(st2.nextToken()); //Mean
            usereventDataLine.d3 = Double.parseDouble(st2.nextToken()); //Standard
            // Deviation.

            // Sum Squared = [ (stddev)^2 + (mean)^2] * N

            usereventDataLine.d3 = ((usereventDataLine.d3 * usereventDataLine.d3) + (usereventDataLine.d2 * usereventDataLine.d2))
                    * usereventDataLine.i0;

            /*
             * System.out.println ("numSamples = " + usereventDataLine.i0);
             * System.out.println ("max = " + usereventDataLine.d0);
             * System.out.println ("min = " + usereventDataLine.d1);
             * System.out.println ("mean = " + usereventDataLine.d2);
             * System.out.println ("sumsqr? = " + usereventDataLine.d3);
             */

        } catch (Exception e) {
            System.out.println("An error occurred!");
            e.printStackTrace();
        }
    }

    private String getGroupNames(String string) {
        try {
            StringTokenizer getMappingNameTokenizer = new StringTokenizer(string, "\"");
            getMappingNameTokenizer.nextToken();
            getMappingNameTokenizer.nextToken();
            String str = getMappingNameTokenizer.nextToken();

            //Just do the group check once.
            if (!(this.groupCheck())) {
                //If present, "GROUP=" will be in this token.
                int tmpInt = str.indexOf("GROUP=");
                if (tmpInt > 0) {
                    this.setGroupNamesPresent(true);
                }
                this.setGroupCheck(true);
            }

            if (groupNamesPresent()) {
                str = getMappingNameTokenizer.nextToken();
                return str;
            }
            //If here, this profile file does not track the group names.
            return null;
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SSD12");
        }
        return null;
    }

    private int getNumberOfUserEvents(String string) {
        try {
            StringTokenizer st = new StringTokenizer(string, " \t\n\r");
            return Integer.parseInt(st.nextToken());
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SSD16");
        }
        return -1;
    }

    private int[] getNCT(String string) {
        int[] nct = new int[3];
        StringTokenizer st = new StringTokenizer(string, " ,\t\n\r");
        nct[0] = Integer.parseInt(st.nextToken());
        nct[1] = Integer.parseInt(st.nextToken());
        nct[2] = Integer.parseInt(st.nextToken());
        return nct;
    }

    private String getMetricName(String inString) {
        try {
            String tmpString = null;
            int tmpInt = inString.indexOf("_MULTI_");

            if (tmpInt > 0) {
                //We are reading data from a multiple counter run.
                //Grab the counter name.
                tmpString = inString.substring(tmpInt + 7);
                return tmpString;
            }
            //We are not reading data from a multiple counter run.
            return tmpString;
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SSD26");
        }

        return null;
    }

    //######
    //End - Pprof.dat string processing methods.
    //######

    //####################################
    //End - Private Section.
    //####################################

    //####################################
    //Instance data.
    //####################################
    private LineData functionDataLine1 = new LineData();
    private LineData functionDataLine2 = new LineData();
    private LineData usereventDataLine = new LineData();
    //####################################
    //End - Instance data.
    //####################################
}