/*
 * Name: DynaprofDataSource.java Author: Robert Bell Description:
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

public class DynaprofDataSource extends DataSource {

    public DynaprofDataSource(Object initializeObject) {
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

    public void load() throws FileNotFoundException, IOException {
        
            //######
            //Frequently used items.
            //######
            int metric = -1;

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

            int numberOfLines = 0;

            Vector v = null;
            File[] files = null;
            //######
            //End - Frequently used items.
            //######
            v = (Vector) initializeObject;
            for (Enumeration e = v.elements(); e.hasMoreElements();) {
                System.out.println("Processing data, please wait ......");
                long time = System.currentTimeMillis();

                files = (File[]) e.nextElement();
                for (int i = 0; i < files.length; i++) {
                    if (this.debug()) {
                        System.out.println("######");
                        System.out.println("Processing file: " + files[i].getName());
                        System.out.println("######");
                    }

                    FileInputStream fileIn = new FileInputStream(files[i]);
                    InputStreamReader inReader = new InputStreamReader(fileIn);
                    BufferedReader br = new BufferedReader(inReader);

                    //For Dynaprof, we assume that the number of metrics remain
                    // fixed over
                    //all threads. Since this information is stored in the
                    // header of the papiprobe
                    //output of each file, we only need to get this information
                    // from the first file.

                    if (!(this.headerProcessed())) {
                        //If header is present, its lines will begin with '#'
                        inputString = br.readLine();
                        if (inputString == null) {
                            System.out.println("Error processing file: " + files[i].getName());
                            System.out.println("Unexpected end of file!");
                            return;
                        } else if ((inputString.charAt(0)) == '#') {
                            if (this.debug())
                                System.out.println("Header present");
                            //Do not need second header line at the moment..
                            br.readLine();
                            //Third header line contains the number of metrics.
                            genericTokenizer = new StringTokenizer(br.readLine(), " \t\n\r");
                            genericTokenizer.nextToken();
                            tokenString = genericTokenizer.nextToken();
                            for (int j = (Integer.parseInt(tokenString)); j > 0; j--) {
                                inputString = br.readLine();
                                //Metric name is second token on line.
                                genericTokenizer = new StringTokenizer(inputString, " :\t\n\r");
                                genericTokenizer.nextToken();
                                String metricName = genericTokenizer.nextToken();
                                this.addMetric(metricName);
                                if (this.debug())
                                    System.out.println("metric name found: " + metricName);
                            }
                            if (this.debug())
                                System.out.println("Number of metrics: "
                                        + this.getNumberOfMetrics());
                        } else {
                            if (this.debug())
                                System.out.println("No header present");
                            this.addMetric("default");
                            metric = 0;
                        }

                        this.setHeaderProcessed(true);
                    }

                    //Metrics should be set by now.
                    int[] nct = this.getNCT(files[i].getName());
                    nodeID = nct[0];
                    contextID = nct[1];
                    threadID = nct[2];

                    node = this.getNode(nodeID);
                    if (node == null)
                        node = this.addNode(nodeID);
                    context = node.getContext(contextID);
                    if (context == null)
                        context = node.addContext(contextID);
                    thread = context.getThread(threadID);
                    if (thread == null) {
                        thread = context.addThread(threadID);
                        for (int j = this.getNumberOfMetrics(); j > 0; j--)
                            thread.incrementStorage();
                        //For Dynaprof, we can be reasonable sure that
                        // functionProfiles are the only
                        //things likely to be tracked. See comment before
                        // Thread.initializeFunctionList(...)
                        //in TauOutputSession for more details on the
                        // positioning in that class.
                    }
                    if (this.debug())
                        System.out.println("n,c,t: " + nct[0] + "," + nct[1] + "," + nct[2]);

                    //####################################
                    //First Line
                    //####################################
                    while ((inputString = br.readLine()) != null) {
                        this.getFunctionDataLine(inputString);
                        boolean totalLine = (inputString.indexOf("TOTAL") == 0);

                        //The start of a new metric is indicated by the
                        // presence of TOTAL.
                        //However, is this is the first metric, we do not need
                        // to compute
                        //derived data, as nothing has been read yet.
                        if (totalLine) {
                            if (metric >= 0) {
                                //Compute derived data for this metric.
                                //Remove after testing is complete ... check to
                                // see if this if/else structure is needed at
                                // all after the removal.
                                //thread.setThreadData(metric);
                                metric++;
                            } else
                                metric++;
                        }

                        //Calculate usec/call
                        double usecCall = functionDataLine.d0 / functionDataLine.i1;
                        if (this.debug()) {
                            System.out.println("function line: " + inputString);
                            System.out.println("name:" + functionDataLine.s0);
                            System.out.println("number_of_children:" + functionDataLine.i0);
                            System.out.println("excl.total:" + functionDataLine.d0);
                            System.out.println("excl.calls:" + functionDataLine.i1);
                            System.out.println("excl.min:" + functionDataLine.d1);
                            System.out.println("excl.max:" + functionDataLine.d2);
                            System.out.println("incl.total:" + functionDataLine.d3);
                            System.out.println("incl.calls:" + functionDataLine.i2);
                            System.out.println("incl.min:" + functionDataLine.d4);
                            System.out.println("incl.max:" + functionDataLine.d5);
                        }
                        if (!totalLine && functionDataLine.i1 != 0) {
                            function = this.addFunction(functionDataLine.s0,
                                    this.getNumberOfMetrics());

                            functionProfile = thread.getFunctionProfile(function);

                            if (functionProfile == null) {
                                functionProfile = new FunctionProfile(function, this.getNumberOfMetrics());
                                thread.addFunctionProfile(functionProfile);
                            }

                            functionProfile.setExclusive(metric, functionDataLine.d0);
                            functionProfile.setInclusive(metric, functionDataLine.d3);
                            functionProfile.setNumCalls(functionDataLine.i1);
                            functionProfile.setNumSubr(functionDataLine.i0);
                            functionProfile.setInclusivePerCall(metric, usecCall);

                            //Set the max values.
                            if ((function.getMaxExclusive(metric)) < functionDataLine.d0)
                                function.setMaxExclusive(metric,
                                        functionDataLine.d0);
                            if ((thread.getMaxExclusive(metric)) < functionDataLine.d0)
                                thread.setMaxExclusive(metric, functionDataLine.d0);

                            if ((function.getMaxInclusive(metric)) < functionDataLine.d3)
                                function.setMaxInclusive(metric,
                                        functionDataLine.d3);
                            if ((thread.getMaxInclusive(metric)) < functionDataLine.d3)
                                thread.setMaxInclusive(metric, functionDataLine.d3);

                            if (function.getMaxNumCalls() < functionDataLine.i1)
                                function.setMaxNumCalls(functionDataLine.i1);
                            if (thread.getMaxNumCalls() < functionDataLine.i1)
                                thread.setMaxNumCalls(functionDataLine.i1);

                            if (function.getMaxNumSubr() < functionDataLine.i0)
                                function.setMaxNumSubr(functionDataLine.i0);
                            if (thread.getMaxNumSubr() < functionDataLine.i0)
                                thread.setMaxNumSubr(functionDataLine.i0);

                            if (function.getMaxInclusivePerCall(metric) < usecCall)
                                function.setMaxInclusivePerCall(metric, usecCall);
                            if (thread.getMaxInclusivePerCall(metric) < usecCall)
                                thread.setMaxInclusivePerCall(metric, usecCall);
                        }
                        //Enter the child for loop.
                        for (int j = 0; j < functionDataLine.i0; j++) {
                            inputString = br.readLine();
                            System.out.println("function child line: " + inputString);
                            this.getFunctionChildDataLine(inputString);
                            if (this.debug()) {
                                System.out.println("function child line: " + inputString);
                                System.out.println("name:" + functionChildDataLine.s0);
                                System.out.println("incl.total:" + functionChildDataLine.d3);
                                System.out.println("incl.calls:" + functionChildDataLine.i2);
                                System.out.println("incl.min:" + functionChildDataLine.d4);
                                System.out.println("incl.max:" + functionChildDataLine.d5);
                            }
                            if (functionDataLine.i1 != 0) {
                                function = this.addFunction(
                                        functionChildDataLine.s0 + " > child", this.getNumberOfMetrics());

                                functionProfile = thread.getFunctionProfile(function);

                                if (functionProfile == null) {
                                    functionProfile = new FunctionProfile(function, this.getNumberOfMetrics());
                                    thread.addFunctionProfile(functionProfile);
                                }

                                //Since this is the child thread, increment the
                                // values.
                                double d1 = functionProfile.getInclusive(metric);
                                double d2 = d1 + functionChildDataLine.d3;
                                functionProfile.setExclusive(metric, d2);
                                functionProfile.setInclusive(metric, d2);

                                double i1 = functionProfile.getNumCalls();
                                if (metric == 0) {
                                    i1 = functionProfile.getNumCalls()
                                            + functionChildDataLine.i2;
                                    functionProfile.setNumCalls(i1);
                                }
                                functionProfile.setInclusivePerCall(metric, d2 / i1);

                                //Set the max values.
                                if ((function.getMaxExclusive(metric)) < d2)
                                    function.setMaxExclusive(metric, d2);
                                if ((thread.getMaxExclusive(metric)) < d2)
                                    thread.setMaxExclusive(metric, d2);

                                if ((function.getMaxInclusive(metric)) < d2)
                                    function.setMaxInclusive(metric, d2);
                                if ((thread.getMaxInclusive(metric)) < d2)
                                    thread.setMaxInclusive(metric, d2);

                                if (function.getMaxNumCalls() < i1)
                                    function.setMaxNumCalls(i1);
                                if (thread.getMaxNumCalls() < i1)
                                    thread.setMaxNumCalls(i1);

                                if (function.getMaxInclusivePerCall(metric) < usecCall)
                                    function.setMaxInclusivePerCall(metric, usecCall);
                                if (thread.getMaxInclusivePerCall(metric) < usecCall)
                                    thread.setMaxInclusivePerCall(metric, usecCall);

                                //Add as a call path from the parent above.
                                function = this.addFunction(
                                        functionDataLine.s0 + " => " + functionChildDataLine.s0
                                                + " > child  ", this.getNumberOfMetrics());
                                functionProfile = thread.getFunctionProfile(function);

                                if (functionProfile == null) {
                                    functionProfile = new FunctionProfile(function, this.getNumberOfMetrics());
                                    thread.addFunctionProfile(functionProfile);
                                }

                                functionProfile.setExclusive(metric, functionChildDataLine.d3);
                                functionProfile.setInclusive(metric, functionChildDataLine.d3);
                                functionProfile.setNumCalls(functionChildDataLine.i2);
                                functionProfile.setInclusivePerCall(metric,
                                        functionChildDataLine.d3 / functionChildDataLine.i2);

                                //Set the max values.
                                if ((function.getMaxExclusive(metric)) < functionChildDataLine.d3)
                                    function.setMaxExclusive(metric,
                                            functionChildDataLine.d3);
                                if ((thread.getMaxExclusive(metric)) < functionChildDataLine.d3)
                                    thread.setMaxExclusive(metric, functionChildDataLine.d3);

                                if ((function.getMaxInclusive(metric)) < functionChildDataLine.d3)
                                    function.setMaxInclusive(metric,
                                            functionChildDataLine.d3);
                                if ((thread.getMaxInclusive(metric)) < functionChildDataLine.d3)
                                    thread.setMaxInclusive(metric, functionChildDataLine.d3);

                                if (function.getMaxNumCalls() < functionChildDataLine.i2)
                                    function.setMaxNumCalls(functionChildDataLine.i2);
                                if (thread.getMaxNumCalls() < functionChildDataLine.i2)
                                    thread.setMaxNumCalls(functionChildDataLine.i2);

                                if (function.getMaxInclusivePerCall(metric) < usecCall)
                                    function.setMaxInclusivePerCall(metric, usecCall);
                                if (thread.getMaxInclusivePerCall(metric) < usecCall)
                                    thread.setMaxInclusivePerCall(metric, usecCall);

                            }
                        }
                    }
                    //Remove after testing is complete.
                    //thread.setThreadData(metric);
                }

                //Generate derived data.
                this.generateDerivedData();
                //Remove after testing is complete.
                //this.setMeanDataAllMetrics(0);

                System.out.println("Processing callpath data ...");
                if (CallPathUtilFuncs.checkCallPathsPresent(getFunctions())) {
                    setCallPathDataPresent(true);
                    CallPathUtilFuncs.buildRelations(this);
                } else {
                    System.out.println("No callpath data found.");
                }
                System.out.println("Done - Processing callpath data!");

                time = (System.currentTimeMillis()) - time;
                System.out.println("Done processing data!");
                System.out.println("Time to process (in milliseconds): " + time);

            }
       
    }

    //####################################
    //Private Section.
    //####################################

    //######
    //Dynaprof string processing methods.
    //######
    private int[] getNCT(String string) {
        int[] nct = new int[3];
        Vector tokens = new Vector();
        StringTokenizer st = new StringTokenizer(string, ".\t\n\r");
        int numberOfTokens = st.countTokens();
        nct[0] = 0;
        for (int i = 0; i < numberOfTokens - 2; i++) {
            st.nextToken();
        }
        String penultimate = st.nextToken();
        String last = st.nextToken();

        try {
            nct[1] = Integer.parseInt(penultimate);
            nct[2] = Integer.parseInt(last);
        } catch (NumberFormatException e1) {
            System.out.println("edu.uoregon.tau.dms.dss.Thread identifier not present ... grabbing context ...");
            try {
                nct[1] = Integer.parseInt(last);
            } catch (NumberFormatException e2) {
                System.out.println("Error, unable to find context identifier");
            }
        }
        return nct;
    }

    private void getFunctionDataLine(String string) {
        
            StringTokenizer st = new StringTokenizer(string, ",\t\n\r");
            functionDataLine.s0 = st.nextToken(); //name
            functionDataLine.i0 = Integer.parseInt(st.nextToken()); //number_of_children
            functionDataLine.d0 = Double.parseDouble(st.nextToken()); //excl.total
            functionDataLine.i1 = Integer.parseInt(st.nextToken()); //excl.calls
            functionDataLine.d1 = Double.parseDouble(st.nextToken()); //excl.min
            functionDataLine.d2 = Double.parseDouble(st.nextToken()); //excl.max
            functionDataLine.d3 = Double.parseDouble(st.nextToken()); //incl.total
            functionDataLine.i2 = Integer.parseInt(st.nextToken()); //incl.calls
            functionDataLine.d4 = Double.parseDouble(st.nextToken()); //incl.min
            functionDataLine.d5 = Double.parseDouble(st.nextToken()); //incl.max
       
    }

    private void getFunctionChildDataLine(String string) {
       
            StringTokenizer st = new StringTokenizer(string, ",\t\n\r");
            functionChildDataLine.s0 = st.nextToken(); //name
            functionChildDataLine.d3 = Double.parseDouble(st.nextToken()); //incl.total
            functionChildDataLine.i2 = Integer.parseInt(st.nextToken()); //incl.calls
            functionChildDataLine.d4 = Double.parseDouble(st.nextToken()); //incl.min
            functionChildDataLine.d5 = Double.parseDouble(st.nextToken()); //incl.max
        
    }

    private void setHeaderProcessed(boolean headerProcessed) {
        this.headerProcessed = headerProcessed;
    }

    private boolean headerProcessed() {
        return headerProcessed;
    }

    //######
    //End - Dynaprof string processing methods.
    //######

    //####################################
    //End - Private Section.
    //####################################

    //####################################
    //Instance data.
    //####################################
    private LineData functionDataLine = new LineData();
    private LineData functionChildDataLine = new LineData();
    private boolean headerProcessed = false;
    //####################################
    //End - Instance data.
    //####################################
}