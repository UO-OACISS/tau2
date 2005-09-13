/*
 * TauOutputSession.java
 * 
 * Title: ParaProf Author: Robert Bell Description:
 */

/*
 * To do: 1) Add some sanity checks to make sure that multiple metrics really do
 * belong together. For example, wrap the creation of nodes, contexts, threads,
 * functions, and the like so that they do not occur after the
 * first metric has been loaded. This will not of course ensure 100% that the
 * data is consistent, but it will at least prevent the worst cases.
 */

package edu.uoregon.tau.dms.dss;

import java.io.*;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.util.*;

public class TauDataSource extends DataSource {

    private volatile boolean abort = false;
    private volatile int totalFiles = 0;
    private volatile int filesRead = 0;
    private LineData functionDataLine = new LineData();
    private LineData usereventDataLine = new LineData();
    private boolean profileStatsPresent = false;
    private boolean groupCheck = false;
    private List dirs; // list of directories (e.g. MULTI__PAPI_FP_INS, MULTI__PAPI_L1_DCM)

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

        // first count the files (for progressbar)
        for (Iterator e = dirs.iterator(); e.hasNext();) {
            File[] files = (File[]) e.next();
            for (int i = 0; i < files.length; i++) {
                totalFiles++;
            }
        }

        boolean finished = false;
        boolean foundValidFile = false;
        boolean ioExceptionEncountered = false;

        while (!finished) {
            try {
                int metric = 0;

                // A flag is needed to test whether we have processed the metric
                // name rather than just checking whether this is the first file set. This is because we
                // might skip that first file (for example if the name were profile.-1.0.0) and thus skip
                // setting the metric name.
                //Reference bug08.

                boolean metricNameProcessed = false;

                Function func = null;
                FunctionProfile functionProfile = null;

                UserEventProfile userEventProfile = null;

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

                    //Only need to call addDefaultToVectors() if not the first run.
                    if (metric != 0) { // If this isn't the first metric, call incrementStorage
                        for (Iterator it = this.getNodes(); it.hasNext();) {
                            Node node = (Node) it.next();
                            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                                Context context = (Context) it2.next();
                                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                                    Thread thread = (Thread) it3.next();
                                    thread.incrementStorage();
                                    for (Iterator e6 = thread.getFunctionProfiles().iterator(); e6.hasNext();) {
                                        FunctionProfile fp = (FunctionProfile) e6.next();
                                        if (fp != null) // fp == null would mean this thread didn't call this function
                                            fp.incrementStorage();
                                    }
                                }
                            }
                        }

                    }

                    for (int i = 0; i < files.length; i++) {
                        filesRead++;

                        if (abort)
                            return;

                        int[] nct = this.getNCT(files[i].getName());

                        if (nct == null && dirs.size() == 1 && files.length == 1) {
                            throw new DataSourceException(
                                    files[i].getName()
                                            + ": This doesn't look like a TAU profile\nDid you mean do use the -f option to specify a file format?");
                        }

                        if (nct != null) {
                            foundValidFile = true;

                            
                            
                            FileInputStream fileIn = new FileInputStream(files[i]);
                            FileChannel channel = fileIn.getChannel();
                            FileLock lock = channel.lock(0,Long.MAX_VALUE,true);
                            InputStreamReader inReader = new InputStreamReader(fileIn);
                            BufferedReader br = new BufferedReader(inReader);

                            
                            
                            nodeID = nct[0];
                            contextID = nct[1];
                            threadID = nct[2];

                            Node node = this.addNode(nodeID);
                            Context context = node.addContext(contextID);
                            Thread thread = context.getThread(threadID);
                            if (thread == null) {
                                thread = context.addThread(threadID);
                            }

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

                            if (metricNameProcessed == false) {
                                //Set the metric name.
                                String metricName = getMetricName(inputString);
                                if (metricName == null)
                                    metricName = new String("Time");
                                this.addMetric(metricName);
                                metricNameProcessed = true;
                            }

                            // Second Line (e.g. "# Name Calls Subrs Excl Incl ProfileCalls")
                            inputString = br.readLine();
                            if (inputString == null) {
                                throw new DataSourceException("Unexpected end of file: " + files[i].getName()
                                        + "\nLooking for '# Name Calls ...' line");
                            }
                            if (i == 0) {
                                //Determine if profile stats or profile calls data is present.
                                if (inputString.indexOf("SumExclSqr") != -1)
                                    this.setProfileStatsPresent(true);
                            }

                            for (int j = 0; j < numFunctions; j++) {

                                inputString = br.readLine();
                                if (inputString == null) {
                                    throw new DataSourceException("Unexpected end of file: " + files[i].getName()
                                            + "\nOnly found " + (j - 2) + " of " + numFunctions + " Function Lines");
                                }

                                this.getFunctionDataLine(inputString);
                                String groupNames = this.getGroupNames(inputString);

                                if (functionDataLine.i0 != 0) {
                                    func = this.addFunction(functionDataLine.s0, 1);

                                    functionProfile = thread.getFunctionProfile(func);

                                    if (functionProfile == null) {
                                        functionProfile = new FunctionProfile(func);
                                        thread.addFunctionProfile(functionProfile);
                                    }

                                    //When we encounter duplicate names in the profile.x.x.x file, treat as additional
                                    //data for the name (that is, don't just overwrite what was there before).
                                    functionProfile.setExclusive(metric, functionProfile.getExclusive(metric)
                                            + functionDataLine.d0);
                                    functionProfile.setInclusive(metric, functionProfile.getInclusive(metric)
                                            + functionDataLine.d1);
                                    if (metric == 0) {
                                        functionProfile.setNumCalls(functionProfile.getNumCalls() + functionDataLine.i0);
                                        functionProfile.setNumSubr(functionProfile.getNumSubr() + functionDataLine.i1);
                                    }

                                    if (metric == 0 && groupNames != null) {
                                        StringTokenizer st = new StringTokenizer(groupNames, "|");
                                        while (st.hasMoreTokens()) {
                                            String groupName = st.nextToken();
                                            if (groupName != null) {
                                                // The potential new group is added here. If the group is already present,
                                                // then the addGroup function will just return the
                                                // already existing group id. See the TrialData
                                                // class for more details.
                                                Group group = this.addGroup(groupName.trim());
                                                func.addGroup(group);
                                            }
                                        }
                                    }

                                }

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

                                        this.getUserEventData(inputString);

                                        // User events
                                        if (usereventDataLine.i0 != 0) {

                                            UserEvent userEvent = this.addUserEvent(usereventDataLine.s0);
                                            userEventProfile = thread.getUserEventProfile(userEvent);

                                            if (userEventProfile == null) {
                                                userEventProfile = new UserEventProfile(userEvent);
                                                thread.addUserEvent(userEventProfile);
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
                            }

                            lock.release();
                            br.close();
                            inReader.close();
                            fileIn.close();
                        }
                    }
                    metric++;
                }

                finished = true;
            } catch (IOException ioe) {
                // retry the load once, maybe the profiles were being written
                if (ioExceptionEncountered) {
                    finished = true;
                    //ioe.printStackTrace();
                } else {
                    cleanData();
                }
                ioExceptionEncountered = true;
            }

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

        if (CallPathUtilFuncs.checkCallPathsPresent(this.getFunctions())) {
            setCallPathDataPresent(true);
        }

        //time = (System.currentTimeMillis()) - time;
        //System.out.println("Time to process (in milliseconds): " + time);
    }

    public String toString() {
        return this.getClass().getName();
    }

    //profile.*.*.* string processing methods.
    private int[] getNCT(String string) {
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

    private void getFunctionDataLine(String string) throws DataSourceException {

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
            functionDataLine.s0 = st1.nextToken(); //Name

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

            functionDataLine.s0 = string.substring(1, i - 1);
            st2 = new StringTokenizer(string.substring(i + 1), " \t\n\r");
        }

        functionDataLine.i0 = Integer.parseInt(st2.nextToken()); //Calls
        functionDataLine.i1 = Integer.parseInt(st2.nextToken()); //Subroutines
        functionDataLine.d0 = Double.parseDouble(st2.nextToken()); //Exclusive
        functionDataLine.d1 = Double.parseDouble(st2.nextToken()); //Inclusive
        if (this.getProfileStatsPresent())
            functionDataLine.d2 = Double.parseDouble(st2.nextToken()); //SumExclSqr
        functionDataLine.i2 = Integer.parseInt(st2.nextToken()); //ProfileCalls

        if (functionDataLine.d0 < 0) {
            System.err.println("Warning, negative values found in profile, ignoring!");
            System.err.println("string = " + string);

            functionDataLine.d0 = 0;
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

    private void getUserEventData(String string) {

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

        usereventDataLine.i0 = (int) Double.parseDouble(st2.nextToken()); //Number of calls
        usereventDataLine.d0 = Double.parseDouble(st2.nextToken()); //Max
        usereventDataLine.d1 = Double.parseDouble(st2.nextToken()); //Min
        usereventDataLine.d2 = Double.parseDouble(st2.nextToken()); //Mean
        usereventDataLine.d3 = Double.parseDouble(st2.nextToken()); //Standard Deviation
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