/*
 * Name: DynaprofDataSource.java 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.perfdmf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class DynaprofDataSource extends DataSource {

    public DynaprofDataSource(File[] files) {
        super();
        this.files = files;
    }

    private File files[];

    public void cancelLoad() {
        return;
    }

    public int getProgress() {
        return 0;
    }

    public void load() throws FileNotFoundException, IOException {

        int metric = -1;

        Function function = null;
        FunctionProfile functionProfile = null;

        Node node = null;
        Context context = null;
        edu.uoregon.tau.perfdmf.Thread thread = null;
        int nodeID = 0;
        int threadID = -1;

        String inputString = null;
        //String s1 = null;
        //String s2 = null;

        String tokenString;
        //String groupNamesString = null;
        StringTokenizer genericTokenizer;

        //int numberOfLines = 0;

        //System.out.println("Processing data, please wait ......");
        long time = System.currentTimeMillis();

        for (int i = 0; i < files.length; i++) {

            FileInputStream fileIn = new FileInputStream(files[i]);
            InputStreamReader inReader = new InputStreamReader(fileIn);
            BufferedReader br = new BufferedReader(inReader);

            // For Dynaprof, we assume that the number of metrics remain
            // fixed over all threads. Since this information is stored in the
            // header of the papiprobe output of each file, we only need to get this information
            // from the first file.

            if (!(this.headerProcessed())) {
                //If header is present, its lines will begin with '#'
                inputString = br.readLine();
                if (inputString == null) {
                    System.out.println("Error processing file: " + files[i].getName());
                    System.out.println("Unexpected end of file!");
                    return;
                } else if ((inputString.charAt(0)) == '#') {
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
                    }
                } else {
                    this.addMetric("Time");
                    metric = 0;
                }

                this.setHeaderProcessed(true);
            }

            int[] nct = this.getNCT(files[i].getName());
            nodeID++;
            threadID = nct[2];

            node = this.addNode(nodeID);
            context = node.addContext(0);
            thread = context.getThread(threadID);

            if (thread == null) {
                thread = context.addThread(threadID, this.getNumberOfMetrics());
            }

            while ((inputString = br.readLine()) != null) {
                this.getFunctionDataLine(inputString);
                boolean totalLine = (inputString.indexOf("TOTAL") == 0);

                // The start of a new metric is indicated by the presence of TOTAL.
                // However, if this is the first metric, we do not need to compute
                // derived data, as nothing has been read yet.
                if (totalLine) {
                    metric++;
                }

                //double inclusivePerCall = functionDataLine.d3 / functionDataLine.i1;
                if (!totalLine && functionDataLine.i1 != 0) {
                    function = this.addFunction(functionDataLine.s0, this.getNumberOfMetrics());

                    functionProfile = thread.getFunctionProfile(function);

                    if (functionProfile == null) {
                        functionProfile = new FunctionProfile(function, this.getNumberOfMetrics());
                        thread.addFunctionProfile(functionProfile);
                    }

                    functionProfile.setExclusive(metric, functionDataLine.d0);
                    functionProfile.setInclusive(metric, functionDataLine.d3);
                    functionProfile.setNumCalls(functionDataLine.i1);
                    functionProfile.setNumSubr(functionDataLine.i2);
                    //functionProfile.setInclusivePerCall(metric, inclusivePerCall);

                    //Set the max values.
//                    if ((function.getMaxExclusive(metric)) < functionDataLine.d0)
//                        function.setMaxExclusive(metric, functionDataLine.d0);
//
//                    if ((function.getMaxInclusive(metric)) < functionDataLine.d3)
//                        function.setMaxInclusive(metric, functionDataLine.d3);
//
//                    if (function.getMaxNumCalls() < functionDataLine.i1)
//                        function.setMaxNumCalls(functionDataLine.i1);
//
//                    if (function.getMaxNumSubr() < functionDataLine.i0)
//                        function.setMaxNumSubr(functionDataLine.i0);
//
//                    if (function.getMaxInclusivePerCall(metric) < inclusivePerCall)
//                        function.setMaxInclusivePerCall(metric, inclusivePerCall);
                }
                //Enter the child for loop.
                for (int j = 0; j < functionDataLine.i0; j++) {
                    inputString = br.readLine();
                    if (inputString.charAt(1) != ',') { // I don't know why these lines are in the files sometimes, but wallclockrpt seems to skip over them too
                        this.getFunctionChildDataLine(inputString);
                        if (functionDataLine.i1 != 0) {
                            function = this.addFunction(
                                    functionDataLine.s0 + " => " + functionChildDataLine.s0,
                                    this.getNumberOfMetrics());

                            functionProfile = thread.getFunctionProfile(function);

                            if (functionProfile == null) {
                                functionProfile = new FunctionProfile(function, this.getNumberOfMetrics());
                                thread.addFunctionProfile(functionProfile);
                            }

                            double inclusive = functionProfile.getInclusive(metric) + functionChildDataLine.d3;
                            functionProfile.setInclusive(metric, inclusive);
                            functionProfile.setExclusive(metric, inclusive);

                            double numCalls = functionProfile.getNumCalls();
                            if (metric == 0) {
                                numCalls = functionProfile.getNumCalls() + functionChildDataLine.i2;
                                functionProfile.setNumCalls(numCalls);
                            }
                            //functionProfile.setInclusivePerCall(metric, functionProfile.getInclusive(metric)
                            //        / numCalls);

                            //Set the max values.
//                            if ((function.getMaxExclusive(metric)) < inclusive)
//                                function.setMaxExclusive(metric, inclusive);
//                            if ((function.getMaxInclusive(metric)) < inclusive)
//                                function.setMaxInclusive(metric, inclusive);
//                            if (function.getMaxNumCalls() < numCalls)
//                                function.setMaxNumCalls(numCalls);
//                            if (function.getMaxInclusivePerCall(metric) < inclusivePerCall)
//                                function.setMaxInclusivePerCall(metric, inclusivePerCall);

                        }
                    }
                }
            }
        }

        this.generateDerivedData();

        time = (System.currentTimeMillis()) - time;
        //System.out.println("Done processing data!");
        //System.out.println("Time to process (in milliseconds): " + time);

    }

    private int[] getNCT(String string) {
        int[] nct = new int[3];
        //Vector tokens = new Vector();
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
            //System.out.println("Thread identifier not present ... grabbing context ...");
            try {
                nct[1] = Integer.parseInt(last);
            } catch (NumberFormatException e2) {
              //  System.out.println("Error, unable to find context identifier");
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

    private LineData functionDataLine = new LineData();
    private LineData functionChildDataLine = new LineData();
    private boolean headerProcessed = false;
}