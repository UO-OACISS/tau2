package edu.uoregon.tau.perfdmf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.sql.SQLException;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

public class TimeSeriesDataSource extends DataSource {

    
    private List<File[]> dirlist;
    private boolean profileStatsPresent = false;
    private boolean groupCheck = false;

    int numSnapShots;
    
    public TimeSeriesDataSource(File dir) {
        FileList fl = new FileList();
        dirlist = fl.helperFindTimeSeriesProfiles(System.getProperty("user.dir"));
        numSnapShots = dirlist.size();
    
        // do some error checking on the dirs
    }

    
    public void load() throws FileNotFoundException, IOException, DataSourceException, SQLException {

        int snapshot = -1;
        for (Iterator<File[]> it = dirlist.iterator(); it.hasNext();) {
            File[] files = (File[]) it.next();

            snapshot++;
            
            System.out.println("loading snapshot " + snapshot);
            
            int metric = snapshot;
            
            
            this.addMetric("Snapshot " + snapshot);
            
            for (int i = 0; i < files.length; i++) {

                int[] nct = TauDataSource.getNCT(files[i].getName());

                int nodeID = nct[0];
                int contextID = nct[1];
                int threadID = nct[2];

                
                Node node = this.addNode(nodeID);
                Context context = node.addContext(contextID);
                Thread thread = context.addThread(threadID, numSnapShots);

                FileInputStream fileIn = new FileInputStream(files[i]);
                InputStreamReader inReader = new InputStreamReader(fileIn);
                BufferedReader br = new BufferedReader(inReader);

                // First Line (e.g. "601 templated_functions")
                String inputString = br.readLine();
                if (inputString == null) {
                    throw new DataSourceException("Unexpected end of file: " + files[i].getName()
                            + "\nLooking for 'templated_functions' line");
                }

                StringTokenizer genericTokenizer = new StringTokenizer(inputString, " \t\n\r");

                // the first token is the number of functions
                String tokenString = genericTokenizer.nextToken();

                int numFunctions;
                try {
                    numFunctions = Integer.parseInt(tokenString);
                } catch (NumberFormatException nfe) {
                    throw new DataSourceException(files[i].getName()
                            + ": Couldn't read number of functions, bad TAU Profile?");
                }

                // Second Line (e.g. "# Name Calls Subrs Excl Incl ProfileCalls")
                inputString = br.readLine();
                if (inputString == null) {
                    throw new DataSourceException("Unexpected end of file: " + files[i].getName()
                            + "\nLooking for '# Name Calls ...' line");
                }

                if (snapshot == 0) {
                    //Determine if profile stats or profile calls data is present.
                    if (inputString.indexOf("SumExclSqr") != -1) {
                        profileStatsPresent = true;
                    }
                }

                
                
                
                for (int j = 0; j < numFunctions; j++) {

                    inputString = br.readLine();
                    if (inputString == null) {
                        throw new DataSourceException("Unexpected end of file: " + files[i].getName() + "\nOnly found "
                                + (j - 2) + " of " + numFunctions + " Function Lines");
                    }

                    this.processFunctionLine(inputString, thread, metric);
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

//              Process the appropriate number of userevent lines.
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
                
                br.close();
                inReader.close();
                fileIn.close();
            }
        }

        this.generateDerivedData();
    }

    
    public int getProgress() {
        return 0;
    }

    public void cancelLoad() {
    }

    
    
    
    
    
    
    private void processFunctionLine(String string, Thread thread, int metric) throws DataSourceException {

        String name;
        double numcalls;
        double numsubr;
        double exclusive;
        double inclusive;
        //double profileCalls;
        //double sumExclSqr;

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
        if (profileStatsPresent) {
            //sumExclSqr = Double.parseDouble(st2.nextToken()); //SumExclSqr
        }

        //profileCalls = Integer.parseInt(st2.nextToken()); //ProfileCalls

        if (inclusive < 0) {
            System.err.println("Warning, negative values found in profile, ignoring!");
            inclusive = 0;
        }
        if (exclusive < 0) {
            System.err.println("Warning, negative values found in profile, ignoring!");
            exclusive = 0;
        }

        if (numcalls != 0) {
            Function func = this.addFunction(name, numSnapShots);

            FunctionProfile functionProfile = thread.getFunctionProfile(func);

            if (functionProfile == null) {
                functionProfile = new FunctionProfile(func, numSnapShots);
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
        if (!groupCheck) {
            //If present, "GROUP=" will be in this token.
            int tmpInt = str.indexOf("GROUP=");
            if (tmpInt > 0) {
                this.setGroupNamesPresent(true);
            }
            groupCheck = true;
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

    
    
    
    
    
}
