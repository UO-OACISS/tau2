/*
 * Name: TauPprofDataSource.java 
 * Author: Robert Bell, Alan Morris
 * Description:
 */


package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;

public class TauPprofDataSource extends DataSource {

    private volatile boolean abort = false;
    private volatile long totalBytes = 0;
    private volatile long bytesRead = 0;
    private LineData functionDataLine1 = new LineData();
    private LineData functionDataLine2 = new LineData();
    private LineData usereventDataLine = new LineData();
    protected boolean firstMetric;
    protected boolean groupCheck = false;

    public TauPprofDataSource(Object initializeObject) {
        super();
        this.initializeObject = initializeObject;
    }

    private Object initializeObject;

    public void cancelLoad() {
        abort = true;
        return;
    }


    public int getProgress() {
        if (totalBytes != 0)
            return (int) ((float) bytesRead / (float) totalBytes * 100);
        return 0;
    }

    public void load() throws FileNotFoundException, IOException {
        int metric = 0;

        Function function = null;
        UserEvent userEvent = null;
        FunctionProfile functionProfile = null;
        UserEventProfile userEventProfile = null;

        edu.uoregon.tau.perfdmf.Thread thread = null;

        meanData = new Thread(-1, -1, -1, 1, this);
        FunctionProfile meanProfile = null;
        
        totalData = new Thread(-2, -2, -2, 1, this);
        FunctionProfile totalProfile = null;

        int nodeID = -1;
        int contextID = -1;
        int threadID = -1;

        String inputString = null;
        String s1 = null;
        String s2 = null;

        String groupNamesString = null;

        //A loop counter.
        int bSDCounter = 0;

        //######
        //End - Frequently used items.
        //######
        List v = (List) initializeObject;
        this.setFirstMetric(true);
        
        for (Iterator e = v.iterator(); e.hasNext();) {
            File files[] = (File[]) e.next();
            long time = System.currentTimeMillis();

            totalBytes = files[0].length();
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

                if (abort)
                    return;

                bytesRead += inputString.length();

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
                    function = this.addFunction(functionDataLine1.s0, 1);

                    // get/create the FunctionProfile for mean
                    meanProfile = meanData.getFunctionProfile(function);
                    if (meanProfile == null) {
                        meanProfile = new FunctionProfile(function, 1);
                        meanData.addFunctionProfile(meanProfile);
                    }
                    function.setMeanProfile(meanProfile);

                    // get/create the FunctionProfile for total
                    totalProfile = totalData.getFunctionProfile(function);
                    if (totalProfile == null) {
                        totalProfile = new FunctionProfile(function, 1);
                        totalData.addFunctionProfile(totalProfile);
                    }
                    function.setTotalProfile(totalProfile);

                }

                switch (lineType) {
                case 0:
                    if (this.getFirstMetric()) {
                        //Grab the group names.
                        groupNamesString = getGroupNames(inputString);
                        if (groupNamesString != null) {
                            StringTokenizer st = new StringTokenizer(groupNamesString, " |");
                            while (st.hasMoreTokens()) {
                                String tmpString = st.nextToken();
                                if (tmpString != null) {
                                    Group group = this.addGroup(tmpString);
                                    function.addGroup(group);
                                }
                            }
                        }
                    }

                    
                    totalProfile.setExclusive(metric, functionDataLine1.d0);
                    //totalProfile.setExclusivePercent(metric, functionDataLine1.d1);
                    
                    break;
                case 1:

                    totalProfile.setInclusive(metric, functionDataLine1.d0);
                    //totalProfile.setInclusivePercent(metric, functionDataLine1.d1);

                    

                    //Set number of calls/subroutines/inclusivePerCall.
                    inputString = br.readLine();
                    bytesRead += inputString.length();

                    this.getFunctionDataLine2(inputString);

                    //Set the values.

                    totalProfile.setNumCalls(functionDataLine2.d0);
                    totalProfile.setNumSubr(functionDataLine2.d1);
                    //totalProfile.setInclusivePerCall(metric, functionDataLine2.d2);

                    
                    break;
                case 2:
                    //Now set the values correctly.
                

                    meanProfile.setExclusive(metric, functionDataLine1.d0);
                    //meanProfile.setExclusivePercent(metric, functionDataLine1.d1);
                    break;
                case 3:
                    //Now set the values correctly.

                    meanProfile.setInclusive(metric, functionDataLine1.d0);
                    //meanProfile.setInclusivePercent(metric, functionDataLine1.d1);

                    //Set number of calls/subroutines/inclusivePerCall.
                    inputString = br.readLine();
                    bytesRead += inputString.length();

                    this.getFunctionDataLine2(inputString);

                    //Set the values.
                    meanProfile.setNumCalls(functionDataLine2.d0);
                    meanProfile.setNumSubr(functionDataLine2.d1);
                    //meanProfile.setInclusivePerCall(metric, functionDataLine2.d2);


                    break;
                case 4:
//                    if ((function.getMaxExclusive(metric)) < functionDataLine1.d0)
//                        function.setMaxExclusive(metric, functionDataLine1.d0);
//                    if ((function.getMaxExclusivePercent(metric)) < functionDataLine1.d1)
//                        function.setMaxExclusivePercent(metric, functionDataLine1.d1);

                    //Get the node,context,thread.
                    int[] array = this.getNCT(inputString);
                    nodeID = array[0];
                    contextID = array[1];
                    threadID = array[2];

                    Node node = this.addNode(nodeID);
                    Context context = node.addContext(contextID);
                    thread = context.getThread(threadID);
                    if (thread == null) {
                        thread = context.addThread(threadID);
                    }

                    functionProfile = thread.getFunctionProfile(function);

                    if (functionProfile == null) {
                        functionProfile = new FunctionProfile(function);
                        thread.addFunctionProfile(functionProfile);
                    }
                    functionProfile.setExclusive(metric, functionDataLine1.d0);
                    //functionProfile.setExclusivePercent(metric, functionDataLine1.d1);
                    //Now check the max values on this thread.
//                    if ((thread.getMaxExclusive(metric)) < functionDataLine1.d0)
//                        thread.setMaxExclusive(metric, functionDataLine1.d0);
//                    if ((thread.getMaxExclusivePercent(metric)) < functionDataLine1.d1)
//                        thread.setMaxExclusivePercent(metric, functionDataLine1.d1);
                    break;
                case 5:
//                    if ((function.getMaxInclusive(metric)) < functionDataLine1.d0)
//                        function.setMaxInclusive(metric, functionDataLine1.d0);
//
//                    if ((function.getMaxInclusivePercent(metric)) < functionDataLine1.d1)
//                        function.setMaxInclusivePercent(metric, functionDataLine1.d1);

                    thread = this.getThread(nodeID, contextID, threadID);
                    functionProfile = thread.getFunctionProfile(function);

                    functionProfile.setInclusive(metric, functionDataLine1.d0);
                    //functionProfile.setInclusivePercent(metric, functionDataLine1.d1);
//                    if ((thread.getMaxInclusive(metric)) < functionDataLine1.d0)
//                        thread.setMaxInclusive(metric, functionDataLine1.d0);
//                    if ((thread.getMaxInclusivePercent(metric)) < functionDataLine1.d1)
//                        thread.setMaxInclusivePercent(metric, functionDataLine1.d1);

                    //Get the number of calls and number of sub routines
                    inputString = br.readLine();
                    bytesRead += inputString.length();
                    this.getFunctionDataLine2(inputString);

                    //Set the values.
                    functionProfile.setNumCalls(functionDataLine2.d0);
                    functionProfile.setNumSubr(functionDataLine2.d1);
                    //functionProfile.setInclusivePerCall(metric, functionDataLine2.d2);

//                    //Set the max values.
//                    if (thread.getMaxNumCalls() < functionDataLine2.d0)
//                        thread.setMaxNumCalls(functionDataLine2.d0);
//
//                    if (thread.getMaxNumSubr() < functionDataLine2.d1)
//                        thread.setMaxNumSubr(functionDataLine2.d1);
//
//                    if (thread.getMaxInclusivePerCall(metric) < functionDataLine2.d2)
//                        thread.setMaxInclusivePerCall(metric, functionDataLine2.d2);
                    break;
                case 6:
                    //Just ignore the string if this is not the first
                    // check.
                    //Assuming is that user events do not change for each
                    // counter value.
                    if (this.getFirstMetric()) {
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

                            s1 = br.readLine();
                            bytesRead += s1.length();
                            s2 = br.readLine();
                            bytesRead += s2.length();

                            getUserEventData(s1);
                            // System.out.println("noc:"+usereventDataLine.i0+"min:"+usereventDataLine.d1+"max:"+usereventDataLine.d0+"mean:"+usereventDataLine.d2);

                            if (usereventDataLine.i0 != 0) {
                                userEvent = this.addUserEvent(usereventDataLine.s0);
                                userEventProfile = thread.getUserEventProfile(userEvent);

                                if (userEventProfile == null) {
                                    userEventProfile = new UserEventProfile(userEvent);
                                    thread.addUserEventProfile(userEventProfile);
                                }

                                userEventProfile.setNumSamples(usereventDataLine.i0);
                                userEventProfile.setMinValue(usereventDataLine.d1);
                                userEventProfile.setMaxValue(usereventDataLine.d0);
                                userEventProfile.setMeanValue(usereventDataLine.d2);
                                userEventProfile.setSumSquared(usereventDataLine.d3);

                                userEventProfile.updateMax();

                            }
                        }
                        //Now set the userEvents flag.
                        setUserEventsPresent(true);
                    }
                    break;
                default:
                    break;
                }

                //Increment the loop counter.
                bSDCounter++;
            }

            //Close the file.
            br.close();


            //Set firstRead to false.
            this.setFirstMetric(false);
//
//            meanData.setThreadDataAllMetrics();

            
            time = (System.currentTimeMillis()) - time;
            //System.out.println("Time to process file (in milliseconds): " + time);
        }

        this.generateDerivedData();
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

        //In this function I need to be careful. If the function name
        // contains "excl", I might interpret this line as being the exclusive line when in
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
        return result;
    }

    private void getFunctionDataLine1(String string) {

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
    }

    private void getFunctionDataLine2(String string) {
        StringTokenizer tokenizer = new StringTokenizer(string, " \t\n\r");
        tokenizer.nextToken();
        tokenizer.nextToken();
        tokenizer.nextToken();

        // number of calls
        functionDataLine2.d0 = Double.parseDouble(tokenizer.nextToken());

        // number of subroutines
        functionDataLine2.d1 = Double.parseDouble(tokenizer.nextToken());

        // inclusive per call
        functionDataLine2.d2 = Double.parseDouble(tokenizer.nextToken());
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
    }

    private String getGroupNames(String string) {
        StringTokenizer tokenizer = new StringTokenizer(string, "\"");
        tokenizer.nextToken();
        tokenizer.nextToken();
        String str = tokenizer.nextToken();

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
            str = tokenizer.nextToken();
            return str;
        }
        //If here, this profile file does not track the group names.
        return null;
    }

    private int getNumberOfUserEvents(String string) {
        StringTokenizer st = new StringTokenizer(string, " \t\n\r");
        return Integer.parseInt(st.nextToken());
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

    }

    

    protected void setFirstMetric(boolean firstMetric) {
        this.firstMetric = firstMetric;
    }

    protected boolean getFirstMetric() {
        return firstMetric;
    }


    protected void setGroupCheck(boolean groupCheck) {
        this.groupCheck = groupCheck;
    }
    protected boolean getGroupCheck() {
        return groupCheck;
    }
    
}