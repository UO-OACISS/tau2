/*
 * Name: MpiPDataSource.java 
 * Author: Kevin Huck 
 * Description: Parse an mpiP data file.
 */

/*
 * To do: The mpiP data has min, mean, max values. What should be done with
 * these values? Should they be stored in a user event?
 * 
 * Support older versions of the output file
 */

package edu.uoregon.tau.dms.dss;

import java.io.*;
import java.util.*;

import java.text.SimpleDateFormat;
import java.util.Date;

public class MpiPDataSource extends DataSource {

    public MpiPDataSource(Object initializeObject) {
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
            FunctionProfile functionProfile = null;

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

            Vector v = null;
            File[] files = null;
            //######
            //End - Frequently used items.
            //######

            v = (Vector) initializeObject;
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
                //####################################
                //End - First Line
                //####################################

                //Set the metric name.
                String metricName = "Time";

                //Need to call increaseVectorStorage() on all objects that
                // require it.
                this.getTrialData().increaseVectorStorage();

                System.out.println("Metric name is: " + metricName);

                metric = this.getNumberOfMetrics();
                this.addMetric(metricName);

                Date startTime = null;
                Date stopTime = null;

                SimpleDateFormat sdfInput = new SimpleDateFormat("yyyy MM dd hh:mm:ss");

                // find the start and stop time
                while ((inputString = br.readLine()) != null) {
                    if (inputString.startsWith("@ Start time")) {

                        String remainder = inputString.substring(inputString.indexOf(':') + 1);

                        try {
                            startTime = sdfInput.parse(remainder);
                        } catch (java.text.ParseException except) {
                            System.out.println("Warning, couldn't parse date \"" + remainder + "\"");
                        }

                        // exit this while loop
                        break;
                    }
                }

                // find the start and stop time
                while ((inputString = br.readLine()) != null) {
                    if (inputString.startsWith("@ Stop time")) {

                        String remainder = inputString.substring(inputString.indexOf(':') + 1);

                        try {
                            stopTime = sdfInput.parse(remainder);
                        } catch (java.text.ParseException except) {
                            System.out.println("Warning, couldn't parse date \"" + remainder + "\"");
                        }

                        // exit this while loop
                        break;
                    }
                }

                int eventCount = 0;
                // find the callsite names
                while ((inputString = br.readLine()) != null) {
                    if (inputString.startsWith("@--- Callsites:")) {
                        // System.out.print("Found callsites: ");
                        genericTokenizer = new StringTokenizer(inputString, ":");
                        // left side
                        inputString = genericTokenizer.nextToken();
                        // right side
                        inputString = genericTokenizer.nextToken();
                        genericTokenizer = new StringTokenizer(inputString, " ");
                        // get the callsite count
                        eventCount = Integer.parseInt(genericTokenizer.nextToken());
                        // the callsite names are indexed at 1, not 0.
                        eventNames = new String[eventCount + 1];
                        // System.out.println(eventCount + " callsites found.");
                        // ignore the next two lines
                        br.readLine();
                        br.readLine();
                        // exit this while loop
                        break;
                    }
                }

                if (inputString != null) {
                    // parse each of the event names
                    for (int i = 1; i <= eventCount; i++) {
                        inputString = br.readLine();
                        getCallsiteHeaders(inputString);
                        eventNames[i] = new String(callsiteHeader.s1 + " => " + "MPI_"
                                + callsiteHeader.s0 + " file: " + callsiteHeader.s2 + " line:"
                                + callsiteHeader.i1);

                        //eventNames[i] = new String(callsiteHeader.s1 + " => "
                        // + "MPI_" + callsiteHeader.s0);
                    }
                }

                // find the callsite data
                int eventDataCount = 0;
                while ((inputString = br.readLine()) != null) {
                    // 0.9
                    //if (inputString.startsWith("@--- Callsite statistics")) {
                    if (inputString.startsWith("@--- Callsite Time statistics")) {
                        // exit this while loop
                        // System.out.print("Found callsite data: ");
                        genericTokenizer = new StringTokenizer(inputString, ":");
                        // left side
                        inputString = genericTokenizer.nextToken();
                        // right side
                        inputString = genericTokenizer.nextToken();
                        genericTokenizer = new StringTokenizer(inputString, " ");
                        // get the callsite count
                        eventDataCount = Integer.parseInt(genericTokenizer.nextToken());
                        // System.out.println(eventDataCount + " callsite data
                        // lines found.");
                        // ignore the next two lines
                        br.readLine();
                        br.readLine();
                        break;
                    }
                }

                if (inputString != null) {
                    // parse each of the event names
                    for (int i = 0; i < eventDataCount; i++) {
                        inputString = br.readLine();
                        while (inputString != null && (inputString.length() == 0))
                            inputString = br.readLine();
                        if (inputString != null) {
                            getCallsiteData(inputString);
                            
                            function = this.getTrialData().addFunction(eventNames[callsiteData.i0], 1);

                            
                            if (callsiteData.i1 >= 0) {
                                if ((function.getMaxExclusive(metric)) < callsiteData.d5) {
                                    function.setMaxExclusive(metric,
                                            callsiteData.d5);
                                    function.setMaxInclusive(metric,
                                            callsiteData.d5);
                                }
                                if ((function.getMaxExclusivePercent(metric)) < callsiteData.d3) {
                                    function.setMaxExclusivePercent(metric,
                                            callsiteData.d3);
                                    function.setMaxInclusivePercent(metric,
                                            callsiteData.d3);
                                }
                                if (function.getMaxNumCalls() < callsiteData.i2)
                                    function.setMaxNumCalls(callsiteData.i2);
                                function.setMaxNumSubr(0);
                                if (function.getMaxInclusivePerCall(metric) < (callsiteData.d1))
                                    function.setMaxInclusivePerCall(metric,
                                            (callsiteData.d1));
                                // get the node data
                                nodeID = callsiteData.i1;
                                contextID = 0;
                                threadID = 0;
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
                                    thread.initializeFunctionList(this.getTrialData().getNumFunctions());
                                }
                                functionProfile = thread.getFunctionProfile(function);
                                if (functionProfile == null) {
                                    functionProfile = new FunctionProfile(function);
                                    thread.addFunctionProfile(functionProfile, function.getID());
                                }
                                functionProfile.setExclusive(metric, callsiteData.d5);
                                functionProfile.setExclusivePercent(metric,
                                        callsiteData.d3);
                                functionProfile.setInclusive(metric, callsiteData.d5);
                                functionProfile.setInclusivePercent(metric,
                                        callsiteData.d3);
                                functionProfile.setNumCalls(callsiteData.i2);
                                functionProfile.setNumSubr(0);
                                functionProfile.setInclusivePerCall(metric, callsiteData.d1);

                                //Now check the max values on this thread.
                                if (thread.getMaxNumCalls() < callsiteData.i2)
                                    thread.setMaxNumCalls(callsiteData.i2);
                                thread.setMaxNumSubr(0);
                                if (thread.getMaxInclusivePerCall(metric) < callsiteData.d1)
                                    thread.setMaxInclusivePerCall(metric, callsiteData.d1);
                                if ((thread.getMaxExclusive(metric)) < callsiteData.d5) {
                                    thread.setMaxExclusive(metric, callsiteData.d5);
                                    thread.setMaxInclusive(metric, callsiteData.d5);
                                }
                                if ((thread.getMaxExclusivePercent(metric)) < callsiteData.d4) {
                                    thread.setMaxExclusivePercent(metric, callsiteData.d4);
                                    thread.setMaxInclusivePercent(metric, callsiteData.d4);
                                }
                            } else {
                                // save the total data
                                i = i - 1;
                            }

                        }
                    }
                }

                //Close the file.
                br.close();

                // Add the .MpiP.Application

                if (startTime != null && stopTime != null) {

                    long inclusive = (stopTime.getTime() - startTime.getTime()) * 1000;

                    function = this.getTrialData().addFunction(".MpiP.Application", 1);

                    for (Enumeration e1 = this.getNCT().getNodes().elements(); e1.hasMoreElements();) {
                        node = (Node) e1.nextElement();
                        for (Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements();) {
                            context = (Context) e2.nextElement();
                            for (Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements();) {
                                thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();

                                Vector functions = thread.getFunctionList();

                                double exclusive = inclusive;
                                double numSubroutines = 0;
                                for (Enumeration e4 = functions.elements(); e4.hasMoreElements();) {
                                    FunctionProfile fp = (FunctionProfile) e4.nextElement();
                                    if (fp != null) {
                                        numSubroutines = numSubroutines
                                                + fp.getNumCalls();
                                        exclusive = exclusive - fp.getExclusive(metric);
                                    }
                                }

                                // set max values for the function
                                if (exclusive > function.getMaxExclusive(metric)) {
                                    function.setMaxExclusive(metric, exclusive);
                                }

                                double exclusivePercent = exclusive / inclusive;

                                if (exclusivePercent > function.getMaxExclusivePercent(metric)) {
                                    function.setMaxExclusivePercent(metric,
                                            exclusivePercent);
                                }

                                function.setMaxInclusive(metric, inclusive);
                                function.setMaxInclusivePercent(metric, inclusive);

                                functionProfile = new FunctionProfile(function);

                                thread.addFunctionProfile(functionProfile, function.getID());
                                //				functionProfile =
                                // thread.getFunction(mappingID);

                                functionProfile.setInclusive(metric, inclusive);
                                functionProfile.setInclusivePercent(metric, 100);

                                functionProfile.setExclusive(metric, exclusive);
                                functionProfile.setExclusivePercent(metric,
                                        exclusivePercent);
                                functionProfile.setNumCalls(1);
                                functionProfile.setNumSubr(numSubroutines);
                                functionProfile.setInclusivePerCall(metric, inclusive);

                                thread.setMaxInclusive(metric, inclusive);
                                thread.setMaxInclusivePercent(metric, 100);

                                //Now check the max values on this thread.
                                if (thread.getMaxNumCalls() < 1)
                                    thread.setMaxNumCalls(1);
                                thread.setMaxNumSubr(numSubroutines);

                                thread.setMaxInclusivePerCall(metric, inclusive);

                                if ((thread.getMaxExclusive(metric)) < exclusive) {
                                    thread.setMaxExclusive(metric, exclusive);
                                }
                                if ((thread.getMaxExclusivePercent(metric)) < exclusivePercent) {
                                    thread.setMaxExclusivePercent(metric, exclusivePercent);
                                    thread.setMaxInclusivePercent(metric, exclusivePercent);
                                }

                            }
                        }
                    }

                }

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

                time = (System.currentTimeMillis()) - time;
                System.out.println("Done processing data file!");
                System.out.println("Time to process file (in milliseconds): " + time);
            }
            //Generate derived data.
            this.generateDerivedData();
            //Remove after testing is complete.
            //this.setMeanDataAllMetrics(0);

        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SSD01");
        }
    }

    //####################################
    //Private Section.
    //####################################

    //######
    //Pprof.dat string processing methods.
    //######

    private void getCallsiteHeaders(String string) {
        try {
            StringTokenizer st1 = new StringTokenizer(string, " ");
            callsiteHeader.i0 = Integer.parseInt(st1.nextToken()); // callsite
                                                                   // index

            // 0.9
            //callsiteHeader.s0 = st1.nextToken(); // MPI function
            //callsiteHeader.s1 = st1.nextToken(); // Parent Function
            //callsiteHeader.s2 = st1.nextToken(); // Filename
            //callsiteHeader.i1 = Integer.parseInt(st1.nextToken()); // Line
            //callsiteHeader.s3 = st1.nextToken(); // PC

            int lev = Integer.parseInt(st1.nextToken()); // Lev, unknown
            callsiteHeader.s2 = st1.nextToken(); // Filename
            callsiteHeader.i1 = Integer.parseInt(st1.nextToken()); // Line
            callsiteHeader.s1 = st1.nextToken(); // Parent Function
            callsiteHeader.s0 = st1.nextToken(); // MPI function

            //	    System.out.println ("function: " + callsiteHeader.s1);

        } catch (Exception e) {
            System.out.println("An error occurred while parsing the callsite header!");
            e.printStackTrace();
        }
    }

    private void getCallsiteData(String string) {
        try {
            StringTokenizer st1 = new StringTokenizer(string, " ");
            callsiteData.s0 = st1.nextToken(); // MPI function
            callsiteData.i0 = Integer.parseInt(st1.nextToken()); // callsite
                                                                 // index

            String tmpString = st1.nextToken(); // rank
            if (tmpString.equals("*"))
                callsiteData.i1 = -1;
            else
                callsiteData.i1 = Integer.parseInt(tmpString); // rank
            callsiteData.i2 = Integer.parseInt(st1.nextToken()); // count
            callsiteData.d0 = Double.parseDouble(st1.nextToken()); // Max
            callsiteData.d1 = Double.parseDouble(st1.nextToken()); // Mean
            callsiteData.d2 = Double.parseDouble(st1.nextToken()); // Min
            callsiteData.d3 = Double.parseDouble(st1.nextToken()); // App%
            callsiteData.d4 = Double.parseDouble(st1.nextToken()); // MPI%
            callsiteData.d5 = callsiteData.d1 * callsiteData.i2 * 1000; // Total
                                                                        // time
                                                                        // for
                                                                        // this
                                                                        // node
        } catch (Exception e) {
            System.out.println("An error occurred while parsing the callsite data!");
            e.printStackTrace();
        }
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
    private LineData callsiteHeader = new LineData();
    private LineData callsiteData = new LineData();
    private String[] eventNames = null;
    //####################################
    //End - Instance data.
    //####################################
}