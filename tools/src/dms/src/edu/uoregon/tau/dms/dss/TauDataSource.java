/*
 * TauOutputSession.java
 * 
 * Title: ParaProf Author: Robert Bell Description:
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

public class TauDataSource extends DataSource {

    public TauDataSource(Object initializeObject) {
        super();
        this.initializeObject = initializeObject;
    }

    private Object initializeObject;

    public void cancelLoad() {
        abort = true;
        return;
    }

    private boolean abort = false;
    int totalFiles = 0;
    int filesRead = 0;

    public int getProgress() {
        if (totalFiles != 0)
            return (int) ((float) filesRead / (float) totalFiles * 100);
        return 0;
    }

    public void load() throws FileNotFoundException, IOException {
        //Record time.
        long time = System.currentTimeMillis();

        Vector v = (Vector) initializeObject;

        // first count the files (for progressbar)

        for (Enumeration e = v.elements(); e.hasMoreElements();) {
            File[] files = (File[]) e.nextElement();
            for (int i = 0; i < files.length; i++) {
                totalFiles++;
            }
        }

        //######
        //Frequently used items.
        //######
        int metric = 0;

        //A flag is needed to test whether we have processed the metric
        // name rather than just
        //checking whether this is the first file set. This is because we
        // might skip that first
        //file (for example if the name were profile.-1.0.0) and thus skip
        // setting the metric name.
        //Reference bug08.
        boolean metricNameProcessed = false;

        Function func = null;
        FunctionProfile functionProfile = null;

        UserEvent userEvent = null;
        UserEventProfile userEventProfile = null;

        Node node = null;
        Context context = null;
        edu.uoregon.tau.dms.dss.Thread thread = null;
        int nodeID = -1;
        int contextID = -1;
        int threadID = -1;

        String inputString = null;
        String s1 = null;
        String s2 = null;

        String tokenString;
        String groupNamesString = null;
        StringTokenizer genericTokenizer;

        int numberOfLines = 0;

        boolean validFilesFound = false;

        for (Enumeration e = v.elements(); e.hasMoreElements();) {
            //            System.out.println("Processing data, please wait ......");

            //Need to call increaseVectorStorage() on all objects that
            // require it.
            this.getTrialData().increaseVectorStorage();

            //Reset metricNameProcessed flag.
            metricNameProcessed = false;

            //Only need to call addDefaultToVectors() if not the first run.
            if (!(metric == 0)) {

                for (Iterator i1 = this.getTrialData().getFunctions(); i1.hasNext();) {
                    Function tmpFunc = (Function) i1.next();
                    tmpFunc.incrementStorage();
                }

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

            File[] files = (File[]) e.nextElement();
            for (int i = 0; i < files.length; i++) {
                filesRead++;

                if (abort)
                    return;

                if (this.debug()) {
                    System.out.println("######");
                    System.out.println("Processing file: " + files[i].getName());
                    System.out.println("######");
                    this.outputDebugMessage("Processing file: " + files[i].getName());
                }

                int[] nct = this.getNCT(files[i].getName());
                if (nct != null) {

                    FileInputStream fileIn = new FileInputStream(files[i]);
                    InputStreamReader inReader = new InputStreamReader(fileIn);
                    BufferedReader br = new BufferedReader(inReader);

                    nodeID = nct[0];
                    contextID = nct[1];
                    threadID = nct[2];

                    node = this.getNCT().getNode(nodeID);
                    if (node == null)
                        node = this.getNCT().addNode(nodeID);
                    context = node.getContext(contextID);
                    if (context == null)
                        context = node.addContext(contextID);
                    thread = context.getThread(threadID);
                    if (thread == null) {
                        thread = context.addThread(threadID);
                        thread.setDebug(this.debug());
                    }
                    if (this.debug())
                        this.outputDebugMessage("n,c,t: " + nct[0] + "," + nct[1] + "," + nct[2]);

                    //####################################
                    //First Line
                    //####################################
                    inputString = br.readLine();
                    if (inputString == null) {
                        System.out.println("Error processing file: " + files[i].getName());
                        System.out.println("Unexpected end of file!");
                        if (this.debug())
                            this.outputDebugMessage("Error processing file: " + files[i].getName()
                                    + "\nUnexpected end of file!");
                        return;
                    }
                    genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
                    //It's first token will be the number of function
                    // present.
                    tokenString = genericTokenizer.nextToken();

                    if (metricNameProcessed == false) {
                        //Set the metric name.
                        String metricName = getMetricName(inputString);
                        if (metricName == null)
                            metricName = new String("Time");
                        this.addMetric(metricName);
                        metricNameProcessed = true;
                        if (this.debug())
                            this.outputDebugMessage("metric name: " + metricName);
                    }
                    //####################################
                    //End - First Line
                    //####################################

                    //####################################
                    //Second Line
                    //####################################
                    inputString = br.readLine();
                    if (inputString == null) {
                        System.out.println("Error processing file: " + files[i].getName());
                        System.out.println("Unexpected end of file!");
                        if (this.debug())
                            this.outputDebugMessage("Error processing file: " + files[i].getName()
                                    + "\nUnexpected end of file!");
                        return;
                    }
                    if (i == 0) {
                        //Determine if profile stats or profile calls data
                        // is present.
                        if (inputString.indexOf("SumExclSqr") != -1)
                            this.setProfileStatsPresent(true);
                    }
                    //####################################
                    //End - Second Line
                    //####################################

                    //Process the appropriate number of function lines.
                    if (this.debug())
                        this.outputDebugMessage("processing functionProfiles");
                    numberOfLines = Integer.parseInt(tokenString);
                    for (int j = 0; j < numberOfLines; j++) {
                        //On the off chance that we start supporting
                        // profiles with no functionProfiles in them
                        //(for example only userevents), don't initialize
                        // the function list until now.
                        //Just as for userevents, this will cut down on
                        // storage.
                        if (j == 0 && (metric == 0))
                            thread.initializeFunctionList(this.getTrialData().getNumFunctions());

                        inputString = br.readLine();
                        if (inputString == null) {
                            System.out.println("Error processing file: " + files[i].getName());
                            System.out.println("Unexpected end of file!");
                            if (this.debug())
                                this.outputDebugMessage("Error processing file: "
                                        + files[i].getName() + "\nUnexpected end of file!");
                            return;
                        }

                        this.getFunctionDataLine(inputString);
                        String groupNames = this.getGroupNames(inputString);
                        //Calculate usec/call
                        double usecCall = functionDataLine.d1 / functionDataLine.i0;
                        if (this.debug())
                            this.outputDebugMessage("function line: " + inputString + "\nName:"
                                    + functionDataLine.s0 + "Calls:" + functionDataLine.i0
                                    + " Subrs:" + functionDataLine.i1 + " Excl:"
                                    + functionDataLine.d0 + " Incl:" + functionDataLine.d1
                                    + " SumExclSqr:" + functionDataLine.d2 + " ProfileCalls:"
                                    + functionDataLine.i2 + "\ngroupNames:" + groupNames);
                        if (functionDataLine.i0 != 0) {
                            func = this.getTrialData().addFunction(functionDataLine.s0, 1);

                            functionProfile = thread.getFunctionProfile(func);

                            if (functionProfile == null) {
                                functionProfile = new FunctionProfile(func);
                                thread.addFunctionProfile(functionProfile, func.getID());
                            }

                            //When we encounter duplicate names in the
                            // profile.x.x.x file, treat as additional
                            //data for the name (that is, don't just
                            // overwrite what was there before).
                            //See todo item 7 in the ParaProf docs.
                            // directory.
                            functionProfile.setExclusive(metric,
                                    functionProfile.getExclusive(metric) + functionDataLine.d0);
                            functionProfile.setInclusive(metric,
                                    functionProfile.getInclusive(metric) + functionDataLine.d1);
                            if (metric == 0) {
                                functionProfile.setNumCalls(functionProfile.getNumCalls()
                                        + functionDataLine.i0);
                                functionProfile.setNumSubr(functionProfile.getNumSubr()
                                        + functionDataLine.i1);
                            }
                            functionProfile.setInclusivePerCall(metric,
                                    functionProfile.getInclusivePerCall(metric) + usecCall);

                            //Set the max values (thread max values are
                            // calculated in the
                            // edu.uoregon.tau.dms.dss.Thread class).
                            if ((func.getMaxExclusive(metric)) < functionDataLine.d0)
                                func.setMaxExclusive(metric, functionDataLine.d0);
                            if ((func.getMaxInclusive(metric)) < functionDataLine.d1)
                                func.setMaxInclusive(metric, functionDataLine.d1);
                            if (func.getMaxNumCalls() < functionDataLine.i0)
                                func.setMaxNumCalls(functionDataLine.i0);
                            if (func.getMaxNumSubr() < functionDataLine.i1)
                                func.setMaxNumSubr(functionDataLine.i1);
                            if (func.getMaxInclusivePerCall(metric) < usecCall)
                                func.setMaxInclusivePerCall(metric, usecCall);

                            if (!(func.groupsSet())) {
                                if (metric == 0) {
                                    if (groupNames != null) {
                                        StringTokenizer st = new StringTokenizer(groupNames, " |");
                                        while (st.hasMoreTokens()) {
                                            String groupName = st.nextToken();
                                            if (groupName != null) {
                                                // The potential new group is
                                                // added here. If the
                                                // group is already present,
                                                // then the addGroup
                                                // function will just return the
                                                // already existing
                                                // group id. See the TrialData
                                                // class for more details.
                                                Group group = this.getTrialData().addGroup(
                                                        groupName);
                                                func.addGroup(group);
                                            }
                                        }
                                    }
                                }
                            }

                        }

                        if (this.debug())
                            this.outputDebugMessage("processing profile calls for function: "
                                    + functionDataLine.s0);
                        //Process the appropriate number of profile call
                        // lines.
                        for (int k = 0; k < functionDataLine.i2; k++) {
                            //this.setProfileCallsPresent(true);
                            inputString = br.readLine();
                            if (this.debug())
                                this.outputDebugMessage("Profile Calls line: " + inputString);
                            genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
                            //Arguments are evaluated left to right.
                            functionProfile.addCall(
                                    Double.parseDouble(genericTokenizer.nextToken()),
                                    Double.parseDouble(genericTokenizer.nextToken()));
                        }
                        if (this.debug())
                            this.outputDebugMessage("done processing profile calls for function: "
                                    + functionDataLine.s0);
                    }
                    if (this.debug())
                        this.outputDebugMessage("done processing functionProfiles");

                    if (this.debug())
                        this.outputDebugMessage("processing aggregates");
                    //Process the appropriate number of aggregate lines.
                    inputString = br.readLine();
                    //A valid profile.*.*.* will always contain this line.
                    if (inputString == null) {
                        System.out.println("Error processing file: " + files[i].getName());
                        System.out.println("Unexpected end of file!");
                        if (this.debug())
                            this.outputDebugMessage("Error processing file: " + files[i].getName()
                                    + "\nUnexpected end of file!");
                        return;
                    }
                    genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
                    //It's first token will be the number of aggregates.
                    tokenString = genericTokenizer.nextToken();

                    numberOfLines = Integer.parseInt(tokenString);
                    for (int j = 0; j < numberOfLines; j++) {
                        //this.setAggregatesPresent(true);
                        inputString = br.readLine();
                        if (this.debug())
                            this.outputDebugMessage("Aggregates line: " + inputString);
                    }
                    if (this.debug())
                        this.outputDebugMessage("done processing aggregates");

                    if (metric == 0) {
                        //Process the appropriate number of userevent
                        // lines.
                        if (this.debug())
                            this.outputDebugMessage("processing userevents");
                        inputString = br.readLine();
                        if (inputString == null) {
                            if (this.debug())
                                this.outputDebugMessage("No userevent data in this file.");
                        } else {
                            genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
                            //It's first token will be the number of
                            // userevents
                            tokenString = genericTokenizer.nextToken();
                            numberOfLines = Integer.parseInt(tokenString);
                            //Skip the heading.
                            br.readLine();
                            for (int j = 0; j < numberOfLines; j++) {
                                if (j == 0) {
                                    thread.initializeUsereventList(this.getTrialData().getNumUserEvents());
                                    setUserEventsPresent(true);
                                }

                                inputString = br.readLine();
                                if (inputString == null) {
                                    System.out.println("Error processing file: "
                                            + files[i].getName());
                                    System.out.println("Unexpected end of file!");
                                    if (this.debug())
                                        this.outputDebugMessage("Error processing file: "
                                                + files[i].getName() + "\nUnexpected end of file!");
                                    return;
                                }
                                this.getUserEventData(inputString);
                                if (this.debug())
                                    this.outputDebugMessage("userevent line: " + inputString
                                            + "\neventname:" + usereventDataLine.s0
                                            + "\nnumevents:" + usereventDataLine.i0 + " max:"
                                            + usereventDataLine.d0 + " min:" + usereventDataLine.d1
                                            + " mean:" + usereventDataLine.d2 + " sumsqr:"
                                            + usereventDataLine.d3);

                                // User events
                                if (usereventDataLine.i0 != 0) {

                                    userEvent = this.getTrialData().addUserEvent(
                                            usereventDataLine.s0);
                                    userEventProfile = thread.getUserEvent(userEvent.getID());

                                    if (userEventProfile == null) {
                                        userEventProfile = new UserEventProfile(userEvent);
                                        thread.addUserEvent(userEventProfile, userEvent.getID());
                                    }

                                    userEventProfile.setUserEventNumberValue(usereventDataLine.i0);
                                    userEventProfile.setUserEventMaxValue(usereventDataLine.d0);
                                    userEventProfile.setUserEventMinValue(usereventDataLine.d1);
                                    userEventProfile.setUserEventMeanValue(usereventDataLine.d2);
                                    userEventProfile.setUserEventSumSquared(usereventDataLine.d3);

                                    userEventProfile.updateMax();

                                }
                            }
                        }
                        if (this.debug())
                            this.outputDebugMessage("done processing userevents");
                    }
                    //Remove after testing is complete.
                    //thread.setThreadData(metric);

                    validFilesFound = true;

                    br.close();
                    inReader.close();
                    fileIn.close();
                }

            }
            metric++;
        }

        //        if (!validFilesFound)
        //            throw new Exception("No valid profiles found");

        //Generate derived data.
        this.generateDerivedData();

        if (CallPathUtilFuncs.isAvailable(getTrialData().getFunctions())) {
            setCallPathDataPresent(true);
            CallPathUtilFuncs.buildRelations(getTrialData());
        }

        //        time = (System.currentTimeMillis()) - time;
        //        System.out.println("Done processing data!");
        //        System.out.println("Time to process (in milliseconds): " + time);
    }

    public void outputDebugMessage(String debugMessage) {
        UtilFncs.objectDebug.outputToFile(this.toString() + "\n" + debugMessage);
    }

    public String toString() {
        return this.getClass().getName();
    }

    //####################################
    //End - Public Section.
    //####################################

    //####################################
    //Private Section.
    //####################################

    //######
    //profile.*.*.* string processing methods.
    //######
    private int[] getNCT(String string) {
        try {
            int[] nct = new int[3];
            StringTokenizer st = new StringTokenizer(string, ".\t\n\r");
            st.nextToken();
            nct[0] = Integer.parseInt(st.nextToken());
            nct[1] = Integer.parseInt(st.nextToken());
            nct[2] = Integer.parseInt(st.nextToken());

            if (nct[0] < 0 || nct[1] < 0 || nct[2] < 0) {
                UtilFncs.systemError(new ParaProfError(this.toString() + ": getNCT(...)",
                        "An error occurred while processing file: " + string,
                        "This file will be ignored!"), null, null);
                if (this.debug())
                    this.outputDebugMessage("getNCT(...)\nAn error occurred while processing file: "
                            + string + "\nThis file will be ignored!");
                return null;
            }
            return nct;
        } catch (Exception e) {
            UtilFncs.systemError(new ParaProfError(this.toString() + ": getNCT(...)",
                    "An error occurred while processing file: " + string,
                    "This file will be ignored!"), null, null);
            if (this.debug())
                this.outputDebugMessage("getNCT(...)\nAn error occurred while processing file: "
                        + string + "\nThis file will be ignored!");
            return null;
        }
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

            tmpInt = inString.indexOf("hw_counters");
            if (tmpInt > 0) {
                //We are reading data from a hardware counter run.
                return "Hardware Counter";
            }

            //We are not reading data from a multiple counter run or hardware
            // counter run.
            return tmpString;
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SSD26");
        }

        return null;
    }

    private void getFunctionDataLine(String string) {

        // first, count the number of double-quotes to determine if the
        // function contains a double-quote
        int quoteCount = 0;
        for (int i = 0; i < string.length(); i++) {
            if (string.charAt(i) == '"')
                quoteCount++;
        }

        if (quoteCount == 0) {
            //throw new 
        }

        StringTokenizer st2;

        if (quoteCount == 2 || quoteCount == 4) { // assume all is well
            StringTokenizer st1 = new StringTokenizer(string, "\"");
            functionDataLine.s0 = st1.nextToken(); //Name

            st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
        } else {

            // there is a quote in the name of the timer/function
            // we assume that TAU_GROUP="..." is there, so the end of the name
            // must be
            // at quoteCount - 2
            int count = 0;
            int i = 0;
            while (count < quoteCount - 2 && i < string.length()) {
                if (string.charAt(i) == '"')
                    count++;
                i++;
            }

            functionDataLine.s0 = string.substring(1, i - 1);
            st2 = new StringTokenizer(string.substring(i + 1), " \t\n\r");
        }

        functionDataLine.i0 = Integer.parseInt(st2.nextToken()); //Calls
        functionDataLine.i1 = Integer.parseInt(st2.nextToken()); //Subroutines
        functionDataLine.d0 = Double.parseDouble(st2.nextToken()); //Exclusive
        functionDataLine.d1 = Double.parseDouble(st2.nextToken()); //Inclusive
        if (this.profileStatsPresent())
            functionDataLine.d2 = Double.parseDouble(st2.nextToken()); //SumExclSqr
        functionDataLine.i2 = Integer.parseInt(st2.nextToken()); //ProfileCalls

    }

    private String getGroupNames(String string) {
        try {

            // first, count the number of double-quotes to determine if the
            // function contains a double-quote
            int quoteCount = 0;
            for (int i = 0; i < string.length(); i++) {
                if (string.charAt(i) == '"')
                    quoteCount++;
            }

            // there is a quote in the name of the timer/function
            // we assume that TAU_GROUP="..." is there, so the end of the name
            // must be
            // at quoteCount - 2
            int count = 0;
            int i = 0;
            while (count < quoteCount - 2 && i < string.length()) {
                if (string.charAt(i) == '"')
                    count++;
                i++;
            }

            StringTokenizer getMappingNameTokenizer = new StringTokenizer(string.substring(i + 1),
                    "\"");
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
                usereventDataLine.s0 = st1.nextToken();
                st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
            } else {

                // there is a quote in the name of the user event
                int count = 0;
                int i = 0;
                while (count < quoteCount && i < string.length()) {
                    if (string.charAt(i) == '"')
                        count++;
                    i++;
                }

                usereventDataLine.s0 = string.substring(1, i - 1);
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
        } catch (Exception e) {
            System.out.println("An error occurred!");
            e.printStackTrace();
        }
    }

    //######
    //End - profile.*.*.* string processing methods.
    //######

    //####################################
    //End - Private Section.
    //####################################

    //####################################
    //Instance data.
    //####################################
    private LineData functionDataLine = new LineData();
    private LineData usereventDataLine = new LineData();
    //####################################
    //End - Instance data.
    //####################################
}