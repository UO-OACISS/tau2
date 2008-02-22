/*
 * TauOutputSession.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
 */

/*
 * To do: 1) Add some sanity checks to make sure that multiple metrics really do
 * belong together. For example, wrap the creation of nodes, contexts, threads,
 * functions, and the like so that they do not occur after the
 * first metric has been loaded. This will not of course ensure 100% that the
 * data is consistent, but it will at least prevent the worst cases.
 */

package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.lang.reflect.Method;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.*;

public class TauDataSource extends DataSource {

    private volatile boolean abort = false;
    private volatile int totalFiles = 0;
    private volatile int filesRead = 0;
    private boolean profileStatsPresent = false;
    private boolean groupCheck = false;
    private List dirs; // list of directories (e.g. MULTI__PAPI_FP_INS, MULTI__PAPI_L1_DCM)
	private int currFunction = 0;

    private File fileToMonitor;

    public TauDataSource(List dirs) {
        super();
        this.dirs = dirs;

        if (dirs.size() > 0) {
            File[] files = (File[]) dirs.get(0);
            if (files.length > 0) {
                fileToMonitor = files[0];
            }
        }
    }

    public void cancelLoad() {
        abort = true;
        return;
    }

    public int getProgress() {
        if (totalFiles != 0)
            return (int) ((float) filesRead / (float) totalFiles * 100);
        return 0;
    }

    public List getFiles() {
        List list = new ArrayList();
        list.add(fileToMonitor);
        return list;
    }
    
    public void load() throws FileNotFoundException, IOException, DataSourceException {
        long time = System.currentTimeMillis();

        boolean modernJava = false;
        try {
            Method m = FileInputStream.class.getMethod("getChannel", null);
            modernJava = true;
        } catch (NoSuchMethodException nsme) {
            // way to go java 1.3
        }

        // first count the files (for progressbar)
        for (Iterator e = dirs.iterator(); e.hasNext();) {
            File[] files = (File[]) e.next();
            for (int i = 0; i < files.length; i++) {
                totalFiles++;
            }
        }

        boolean foundValidFile = false;
        int metric = 0;

        // A flag is needed to test whether we have processed the metric
        // name rather than just checking whether this is the first file set. This is because we
        // might skip that first file (for example if the name were profile.-1.0.0) and thus skip
        // setting the metric name.
        //Reference bug08.

        boolean metricNameProcessed = false;

        int nodeID = -1;
        int contextID = -1;
        int threadID = -1;

        String inputString = null;

        String tokenString;
        StringTokenizer genericTokenizer;

        // iterate through the vector of File arrays (each directory)
        for (Iterator e = dirs.iterator(); e.hasNext();) {
            File[] files = (File[]) e.next();

            //Reset metricNameProcessed flag.
            metricNameProcessed = false;

            for (int i = 0; i < files.length; i++) {

                boolean finished = false;
                int ioExceptionsEncountered = 0;

                while (!finished) {
                    Thread thread = null;
                    try {

                        filesRead++;

                        if (abort) {
                            return;
                        }

                        int[] nct = getNCT(files[i].getName());

                        if (nct == null && dirs.size() == 1 && files.length == 1) {
                            throw new DataSourceException(
                                    files[i].getName()
                                            + ": This doesn't look like a TAU profile\nDid you mean do use the -f option to specify a file format?");
                        }

                        if (nct == null) {
                            finished = true;
                            continue;
                        }

                        nodeID = nct[0];
                        contextID = nct[1];
                        threadID = nct[2];

                        thread = this.addThread(nodeID, contextID, threadID);

                        foundValidFile = true;

                        FileInputStream fileIn = new FileInputStream(files[i]);
                        FileChannel channel;
                        FileLock lock = null;

                        if (modernJava && monitored) {
                            channel = fileIn.getChannel();
                            try {
                                lock = channel.lock(0, Long.MAX_VALUE, true);
                            } catch (IOException ioe) {
                                modernJava = false;
                                lock = null;
                            }
                        }
                        InputStreamReader inReader = new InputStreamReader(fileIn);
                        BufferedReader br = new BufferedReader(inReader);

                        // First Line (e.g. "601 templated_functions")
                        inputString = br.readLine();
                        if (inputString == null) {
                            throw new DataSourceException("Unexpected end of file: " + files[i].getName()
                                    + "\nLooking for 'templated_functions' line");
                        }
                        genericTokenizer = new StringTokenizer(inputString, " \t\n\r");

                        // the first token is the number of functions
                        tokenString = genericTokenizer.nextToken();

                        int numFunctions;
                        try {
                            numFunctions = Integer.parseInt(tokenString);
                        } catch (NumberFormatException nfe) {
                            throw new DataSourceException(files[i].getName()
                                    + ": Couldn't read number of functions, bad TAU Profile?");
                        }

                        // grab the (possible) metric name
                        String metricName = getMetricName(inputString);

                        // Second Line (e.g. "# Name Calls Subrs Excl Incl ProfileCalls")
                        inputString = br.readLine();
                        if (inputString == null) {
                            throw new DataSourceException("Unexpected end of file: " + files[i].getName()
                                    + "\nLooking for '# Name Calls ...' line");
                        }

                        if (inputString.indexOf("<metadata>") != -1) {
                            int start = inputString.indexOf("<metadata>");
                            int end = inputString.indexOf("</metadata>") + 11;
                            String metadata = inputString.substring(start, end);
                            MetaDataParser.parse(thread.getMetaData(), metadata);
                        }

                        // there may or may not be a metric name in the metadata
                        String metaDataMetricName = (String) thread.getMetaData().get("Metric Name");
                        
                        // remove it if it was there
                        thread.getMetaData().remove("Metric Name");
                        
                        if (metricNameProcessed == false) {
                            if (metaDataMetricName != null) {
                                metricName = metaDataMetricName;
                            }
                            //Set the metric name.
                            if (metricName == null) {
                                metricName = "Time";
                            }
                            this.addMetric(metricName);
                            metricNameProcessed = true;
                        }

                        if (i == 0) {
                            //Determine if profile stats or profile calls data is present.
                            if (inputString.indexOf("SumExclSqr") != -1)
                                this.setProfileStatsPresent(true);
                        }

                        for (int j = 0; j < numFunctions; j++) {
							this.currFunction = j;

                            inputString = br.readLine();
                            if (inputString == null) {
                                throw new DataSourceException("Unexpected end of file: " + files[i].getName() + "\nOnly found "
                                        + (j - 2) + " of " + numFunctions + " Function Lines");
                            }

                            this.processFunctionLine(inputString, thread, metric);

                            // unused profile calls

                            //                        //Process the appropriate number of profile call lines.
                            //                        for (int k = 0; k < functionDataLine.i2; k++) {
                            //                            //this.setProfileCallsPresent(true);
                            //                            inputString = br.readLine();
                            //                            genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
                            //                            //Arguments are evaluated left to right.
                            //                            functionProfile.addCall(Double.parseDouble(genericTokenizer.nextToken()),
                            //                                    Double.parseDouble(genericTokenizer.nextToken()));
                            //                        }
                        }

                        //Process the appropriate number of aggregate lines.
                        inputString = br.readLine();

                        //A valid profile.*.*.* will always contain this line.
                        if (inputString == null) {
                            throw new DataSourceException("Unexpected end of file: " + files[i].getName()
                                    + "\nLooking for 'aggregates' line");
                        }
                        genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
                        //It's first token will be the number of aggregates.
                        tokenString = genericTokenizer.nextToken();

                        numFunctions = Integer.parseInt(tokenString);
                        for (int j = 0; j < numFunctions; j++) {
                            //this.setAggregatesPresent(true);
                            inputString = br.readLine();
                        }

                        if (metric == 0) {
                            //Process the appropriate number of userevent lines.
                            inputString = br.readLine();
                            if (inputString != null) {
                                genericTokenizer = new StringTokenizer(inputString, " \t\n\r");
                                //It's first token will be the number of userEvents
                                tokenString = genericTokenizer.nextToken();
                                int numUserEvents = Integer.parseInt(tokenString);

                                //Skip the heading (e.g. "# eventname numevents max min mean sumsqr")
                                br.readLine();
                                for (int j = 0; j < numUserEvents; j++) {
                                    if (j == 0) {
                                        setUserEventsPresent(true);
                                    }

                                    inputString = br.readLine();
                                    if (inputString == null) {
                                        throw new DataSourceException("Unexpected end of file: " + files[i].getName()
                                                + "\nOnly found " + (j - 2) + " of " + numUserEvents + " User Event Lines");
                                    }

                                    this.processUserEventLine(inputString, thread);

                                }
                            }
                        }

                        if (lock != null) {
                            lock.release();
                        }
                        br.close();
                        inReader.close();
                        fileIn.close();

                        finished = true;
                    } catch (Exception ex) {
                        ex.printStackTrace();
						System.out.println("Current Function: " + currFunction);
                        //System.out.println("ex:);
                        if (!(ex instanceof IOException || ex instanceof FileNotFoundException)) {
                            throw new RuntimeException(ex == null ? null : ex.toString());
                        }

                        //if (!reloading) {
                        //    throw new RuntimeException(ex);
                        //} else {
                        try {
                            java.lang.Thread.sleep(250);
                        } catch (Exception e2) {
                            // eat it
                        }

                        // retry the load once, maybe the profiles were being written
                        if (ioExceptionsEncountered > 5) {
                            System.err.println("too many exceptions caught");
                            ex.printStackTrace();
                            finished = true;
                        } else {
                            if (thread != null) {
                                for (Iterator it2 = thread.getFunctionProfileIterator(); it2.hasNext();) {
                                    FunctionProfile fp = (FunctionProfile) it2.next();
                                    if (fp != null) {
                                        for (int m = 0; m < this.getNumberOfMetrics(); m++) {
                                            fp.setExclusive(m, 0);
                                            fp.setInclusive(m, 0);
                                        }
                                        fp.setNumSubr(0);
                                        fp.setNumCalls(0);
                                    }
                                }
                            }
                        }
                        ioExceptionsEncountered++;
                        //}
                    }

                }
            }
            metric++;
        }

        if (foundValidFile == false) {
            throw new DataSourceException(
                    "Didn't find any valid files.\nAre you sure these are TAU profiles? (e.g. profile.*.*.*)");
        }

        //time = (System.currentTimeMillis()) - time;
        //System.out.println("Time to process (in milliseconds): " + time);
        //        time = System.currentTimeMillis();

        //Generate derived data.
        this.generateDerivedData();
        this.aggregateMetaData();
        this.buildXMLMetaData();

        //time = (System.currentTimeMillis()) - time;
        //System.out.println("Time to process (in milliseconds): " + time);
    }

    public String toString() {
        return this.getClass().getName();
    }

    //profile.*.*.* string processing methods.
    public static int[] getNCT(String string) {
        try {
            int[] nct = new int[3];
            StringTokenizer st = new StringTokenizer(string, ".\t\n\r");
            st.nextToken();
            nct[0] = Integer.parseInt(st.nextToken());
            nct[1] = Integer.parseInt(st.nextToken());
            nct[2] = Integer.parseInt(st.nextToken());

            if (nct[0] < 0 || nct[1] < 0 || nct[2] < 0) {
                // I'm 99% sure that this doesn't happen anymore
                return null;
            }
            return nct;
        } catch (Exception e) {
            // I'm 99% sure that this doesn't happen anymore
            return null;
        }
    }

    private String getMetricName(String inString) {
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

    }

    private void processFunctionLine(String string, Thread thread, int metric) throws DataSourceException {

        String name;
        double numcalls;
        double numsubr;
        double exclusive;
        double inclusive;
        double profileCalls;
        double sumExclSqr;

        String groupNames = this.getGroupNames(string);

        // first, count the number of double-quotes to determine if the
        // function contains a double-quote
        int quoteCount = 0;
        for (int i = 0; i < string.length(); i++) {
            if (string.charAt(i) == '"')
                quoteCount++;
        }

        if (quoteCount == 0) {
            throw new DataSourceException("Looking for function line, found '" + string + "' instead");
        }

        StringTokenizer st2;

        if (quoteCount == 2 || quoteCount == 4) { // assume all is well
            StringTokenizer st1 = new StringTokenizer(string, "\"");
            name = st1.nextToken(); //Name

            st2 = new StringTokenizer(st1.nextToken(), " \t\n\r");
        } else {

            // there is a quote in the name of the timer/function
            // we assume that TAU_GROUP="..." is there, so the end of the name
            // must be at (quoteCount - 2)
            int count = 0;
            int i = 0;
            while (count < quoteCount - 2 && i < string.length()) {
                if (string.charAt(i) == '"')
                    count++;
                i++;
            }

            name = string.substring(1, i - 1);
            st2 = new StringTokenizer(string.substring(i + 1), " \t\n\r");
        }

        numcalls = Double.parseDouble(st2.nextToken()); //Calls
        numsubr = Double.parseDouble(st2.nextToken()); //Subroutines
        exclusive = Double.parseDouble(st2.nextToken()); //Exclusive
        inclusive = Double.parseDouble(st2.nextToken()); //Inclusive
        if (this.getProfileStatsPresent()) {
            sumExclSqr = Double.parseDouble(st2.nextToken()); //SumExclSqr
        }

        profileCalls = Integer.parseInt(st2.nextToken()); //ProfileCalls

        if (inclusive < 0) {
            System.err.println("Warning, negative values found in profile, ignoring! (routine: " + name + ")");
            inclusive = 0;
        }
        if (exclusive < 0) {
            System.err.println("Warning, negative values found in profile, ignoring! (routine: " + name + ")");
            exclusive = 0;
        }

        if (numcalls != 0) {
            Function func = this.addFunction(name, 1);

            FunctionProfile functionProfile = thread.getFunctionProfile(func);

            if (functionProfile == null) {
                functionProfile = new FunctionProfile(func);
                thread.addFunctionProfile(functionProfile);
            }

            //When we encounter duplicate names in the profile.x.x.x file, treat as additional
            //data for the name (that is, don't just overwrite what was there before).
            functionProfile.setExclusive(metric, functionProfile.getExclusive(metric) + exclusive);
            functionProfile.setInclusive(metric, functionProfile.getInclusive(metric) + inclusive);
            if (metric == 0) {
                functionProfile.setNumCalls(functionProfile.getNumCalls() + numcalls);
                functionProfile.setNumSubr(functionProfile.getNumSubr() + numsubr);
            }

            if (metric == 0) {
                addGroups(groupNames, func);
            }
        }
    }

    private String getGroupNames(String string) {

        // first, count the number of double-quotes to determine if the
        // function contains a double-quote
        int quoteCount = 0;
        for (int i = 0; i < string.length(); i++) {
            if (string.charAt(i) == '"')
                quoteCount++;
        }

        // there is a quote in the name of the timer/function
        // we assume that TAU_GROUP="..." is there, so the end of the name
        // must be (at quoteCount - 2)
        int count = 0;
        int i = 0;
        while (count < quoteCount - 2 && i < string.length()) {
            if (string.charAt(i) == '"')
                count++;
            i++;
        }

        StringTokenizer getNameTokenizer = new StringTokenizer(string.substring(i + 1), "\"");
        String str = getNameTokenizer.nextToken();

        //Just do the group check once.
        if (!(this.getGroupCheck())) {
            //If present, "GROUP=" will be in this token.
            int tmpInt = str.indexOf("GROUP=");
            if (tmpInt > 0) {
                this.setGroupNamesPresent(true);
            }
            this.setGroupCheck(true);
        }

        if (getGroupNamesPresent()) {
            try {
                str = getNameTokenizer.nextToken();
                return str;
            } catch (NoSuchElementException e) {
                // possibly GROUP=""
                return null;
            }
        }
        //If here, this profile file does not track the group names.
        return null;
    }

    private void processUserEventLine(String string, Thread thread) {

        String name;
        double numSamples;
        double sampleMax;
        double sampleMin;
        double sampleMean;
        double sampleSumSquared;

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
            name = st1.nextToken();
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

            name = string.substring(1, i - 1);
            st2 = new StringTokenizer(string.substring(i + 1), " \t\n\r");
        }

        numSamples = Double.parseDouble(st2.nextToken()); //Number of calls
        sampleMax = Double.parseDouble(st2.nextToken()); //Max
        sampleMin = Double.parseDouble(st2.nextToken()); //Min
        sampleMean = Double.parseDouble(st2.nextToken()); //Mean
        sampleSumSquared = Double.parseDouble(st2.nextToken()); //Standard Deviation

        if (numSamples != 0) {
            UserEvent userEvent = this.addUserEvent(name);
            UserEventProfile userEventProfile = thread.getUserEventProfile(userEvent);

            if (userEventProfile == null) {
                userEventProfile = new UserEventProfile(userEvent);
                thread.addUserEventProfile(userEventProfile);
            }

            userEventProfile.setNumSamples(numSamples);
            userEventProfile.setMaxValue(sampleMax);
            userEventProfile.setMinValue(sampleMin);
            userEventProfile.setMeanValue(sampleMean);
            userEventProfile.setSumSquared(sampleSumSquared);
            userEventProfile.updateMax();
        }
    }

    protected void setProfileStatsPresent(boolean profileStatsPresent) {
        this.profileStatsPresent = profileStatsPresent;
    }

    public boolean getProfileStatsPresent() {
        return profileStatsPresent;
    }

    protected void setGroupCheck(boolean groupCheck) {
        this.groupCheck = groupCheck;
    }

    protected boolean getGroupCheck() {
        return groupCheck;
    }

}
