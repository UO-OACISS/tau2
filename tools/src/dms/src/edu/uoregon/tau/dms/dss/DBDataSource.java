/*
 * To do: 1) Add some sanity checks to make sure that multiple metrics really do
 * belong together. For example, wrap the creation of nodes, contexts, threads,
 * global mapping elements, and the like so that they do not occur after the
 * first metric has been loaded. This will not of course ensure 100% that the
 * data is consistent, but it will at least prevent the worst cases.
 */

package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.sql.*;

public class DBDataSource extends DataSource {

    public DBDataSource(Object initializeObject) {
        super();
        this.setMetrics(new Vector());
        this.initializeObject = initializeObject;
    }

    private Object initializeObject;

    
    
    

    private boolean abort = false;
    int totalItems = 0;
    int itemsDone = 0;

    public int getProgress() {
        return DatabaseAPI.getProgress();
        
        
        //        if (totalItems != 0)
//            return (int) ((float) itemsDone / (float) totalItems * 100);
//        return 0;
    }
    public void cancelLoad() {
        abort = true;
        return;
    }

    
    
    public void load() throws SQLException {
        
            //######
            //Frequently used items.
            //######
            DatabaseAPI databaseAPI = (DatabaseAPI) initializeObject;
            
            Function function = null;
            UserEvent userEvent = null;
            FunctionProfile functionProfile = null;
            UserEventProfile userEventProfile = null;

            Node node = null;
            Context context = null;
            edu.uoregon.tau.dms.dss.Thread thread = null;
            int nodeID = -1;
            int contextID = -1;
            int threadID = -1;
            int mappingID = -1;

            //Vector localMap = new Vector();
            //######
            //End - Frequently used items.
            //######
           // System.out.println("Processing data, please wait ......");
            long time = System.currentTimeMillis();

            int numberOfMetrics = databaseAPI.getNumberOfMetrics();
            //System.out.println("Found " + numberOfMetrics + " metrics.");
            for (int i = 0; i < numberOfMetrics; i++) {
                this.addMetric(databaseAPI.getMetricName(i));
                this.getTrialData().increaseVectorStorage();
            }

            //Add the functionProfiles.
            ListIterator l = databaseAPI.getIntervalEvents();
            
            meanData = new Thread(-1,-1,-1, numberOfMetrics);
            meanData.initializeFunctionList(this.getTrialData().getNumFunctions());
    
            totalItems += this.getTrialData().getNumFunctions();
            
            while (l.hasNext()) {
                IntervalEvent ie = (IntervalEvent) l.next();
                
                function = this.getTrialData().addFunction(ie.getName(), numberOfMetrics);

                FunctionProfile meanProfile = new FunctionProfile(function, numberOfMetrics);
                function.setMeanProfile(meanProfile);
                meanData.addFunctionProfile(meanProfile,function.getID());
                
                
                //Add element to the localMap for more efficient lookup later
                // in the function.
                //localMap.add(new FunIndexFunIDPair(f.getIndexID(), id));

                IntervalLocationProfile ilp = ie.getMeanSummary();

                if (ie.getGroup() != null) {
                    Group group = this.getTrialData().addGroup(ie.getGroup());
                    function.addGroup(group);
                    function.setGroupsSet(true);
                    this.setGroupNamesPresent(true);
                }

                for (int i = 0; i < numberOfMetrics; i++) {
                    meanProfile.setExclusive(i, ilp.getExclusive(i));
                    meanProfile.setExclusivePercent(i,
                            ilp.getExclusivePercentage(i));
                    meanProfile.setInclusive(i, ilp.getInclusive(i));
                    meanProfile.setInclusivePercent(i,
                            ilp.getInclusivePercentage(i));
                    meanProfile.setInclusivePerCall(i, ilp.getInclusivePerCall(i));
                    meanProfile.setNumCalls(ilp.getNumCalls());
                    meanProfile.setNumSubr(ilp.getNumSubroutines());

                    if ((this.getTrialData().getMaxMeanExclusiveValue(i)) < ilp.getExclusive(i)) {
                        this.getTrialData().setMaxMeanExclusiveValue(i, ilp.getExclusive(i));
                    }
                    if ((this.getTrialData().getMaxMeanExclusivePercentValue(i)) < ilp.getExclusivePercentage(i)) {
                        this.getTrialData().setMaxMeanExclusivePercentValue(i,
                                ilp.getExclusivePercentage(i));
                    }
                    if ((this.getTrialData().getMaxMeanInclusiveValue(i)) < ilp.getInclusive(i)) {
                        this.getTrialData().setMaxMeanInclusiveValue(i, ilp.getInclusive(i));
                    }
                    if ((this.getTrialData().getMaxMeanInclusivePercentValue(i)) < ilp.getInclusivePercentage(i)) {
                        this.getTrialData().setMaxMeanInclusivePercentValue(i,
                                ilp.getInclusivePercentage(i));
                    }

                    if ((this.getTrialData().getMaxMeanInclusivePerCall(i)) < ilp.getInclusivePerCall(i)) {
                        this.getTrialData().setMaxMeanInclusivePerCall(i,
                                ilp.getInclusivePerCall(i));
                    }

                    if ((this.getTrialData().getMaxMeanNumberOfCalls()) < ilp.getNumCalls()) {
                        this.getTrialData().setMaxMeanNumberOfCalls(ilp.getNumCalls());
                    }

                    if ((this.getTrialData().getMaxMeanNumberOfSubRoutines()) < ilp.getNumSubroutines()) {
                        this.getTrialData().setMaxMeanNumberOfSubRoutines(
                                ilp.getNumSubroutines());
                    }
                }

                
                meanData.setThreadDataAllMetrics();

                function.setMeanValuesSet(true);

                ilp = ie.getTotalSummary();
                for (int i = 0; i < numberOfMetrics; i++) {
                    function.setTotalExclusive(i, ilp.getExclusive(i));
                    function.setTotalExclusivePercent(i,
                            ilp.getExclusivePercentage(i));
                    function.setTotalInclusive(i, ilp.getInclusive(i));
                    function.setTotalInclusivePercent(i,
                            ilp.getInclusivePercentage(i));
                    function.setTotalInclusivePerCall(i, ilp.getInclusivePerCall(i));
                    function.setTotalNumCalls(ilp.getNumCalls());
                    function.setTotalNumSubr(ilp.getNumSubroutines());

                }
            }

            //Collections.sort(localMap);

           

            l = databaseAPI.getIntervalEventData();
            
            while (l.hasNext()) {
                IntervalLocationProfile fdo = (IntervalLocationProfile) l.next();
                node = this.getNCT().getNode(fdo.getNode());
                if (node == null)
                    node = this.getNCT().addNode(fdo.getNode());
                context = node.getContext(fdo.getContext());
                if (context == null)
                    context = node.addContext(fdo.getContext());
                thread = context.getThread(fdo.getThread());
                if (thread == null) {
                    thread = context.addThread(fdo.getThread(), numberOfMetrics);
                    thread.setDebug(this.debug());
                    thread.initializeFunctionList(this.getTrialData().getNumFunctions());

                }

                //Get Function and FunctionProfile.

                //Obtain the mapping id from the local map.
                //int pos = Collections.binarySearch(localMap, new
                // FunIndexFunIDPair(fdo.getIntervalEventID(),0));
                //mappingID =
                // ((FunIndexFunIDPair)localMap.elementAt(pos)).paraProfId;

                function = this.getTrialData().getFunction(databaseAPI.getIntervalEvent(fdo.getIntervalEventID()).getName());
                functionProfile = thread.getFunctionProfile(function);
                
                if (functionProfile == null) {
                    functionProfile = new FunctionProfile(function, numberOfMetrics);
                    thread.addFunctionProfile(functionProfile, function.getID());
                }

                for (int i = 0; i < numberOfMetrics; i++) {
                    functionProfile.setExclusive(i, fdo.getExclusive(i));
                    functionProfile.setInclusive(i, fdo.getInclusive(i));
                    functionProfile.setExclusivePercent(i, fdo.getExclusivePercentage(i));
                    functionProfile.setInclusivePercent(i, fdo.getInclusivePercentage(i));
                    functionProfile.setInclusivePerCall(i, fdo.getInclusivePerCall(i));
                    functionProfile.setNumCalls(fdo.getNumCalls());
                    functionProfile.setNumSubr(fdo.getNumSubroutines());

                    //Set the max values.
                    if ((function.getMaxExclusive(i)) < fdo.getExclusive(i))
                        function.setMaxExclusive(i, fdo.getExclusive(i));
                    if ((function.getMaxExclusivePercent(i)) < fdo.getExclusivePercentage(i))
                        function.setMaxExclusivePercent(i,
                                fdo.getExclusivePercentage(i));
                    if ((function.getMaxInclusive(i)) < fdo.getInclusive(i))
                        function.setMaxInclusive(i, fdo.getInclusive(i));
                    if ((function.getMaxInclusivePercent(i)) < fdo.getInclusivePercentage(i))
                        function.setMaxInclusivePercent(i,
                                fdo.getInclusivePercentage(i));
                    if (function.getMaxNumCalls() < fdo.getNumCalls())
                        function.setMaxNumCalls(fdo.getNumCalls());
                    if (function.getMaxNumSubr() < fdo.getNumSubroutines())
                        function.setMaxNumSubr(fdo.getNumSubroutines());
                    if (function.getMaxInclusivePerCall(i) < fdo.getInclusivePerCall(i))
                        function.setMaxInclusivePerCall(i, fdo.getInclusivePerCall(i));

                    if ((thread.getMaxExclusive(i)) < fdo.getExclusive(i))
                        thread.setMaxExclusive(i, fdo.getExclusive(i));
                    if ((thread.getMaxExclusivePercent(i)) < fdo.getExclusivePercentage(i))
                        thread.setMaxExclusivePercent(i, fdo.getExclusivePercentage(i));
                    if ((thread.getMaxInclusive(i)) < fdo.getInclusive(i))
                        thread.setMaxInclusive(i, fdo.getInclusive(i));
                    if ((thread.getMaxInclusivePercent(i)) < fdo.getInclusivePercentage(i))
                        thread.setMaxInclusivePercent(i, fdo.getInclusivePercentage(i));
                    if (thread.getMaxNumCalls() < fdo.getNumCalls())
                        thread.setMaxNumCalls(fdo.getNumCalls());
                    if (thread.getMaxNumSubr() < fdo.getNumSubroutines())
                        thread.setMaxNumSubr(fdo.getNumSubroutines());
                    if (thread.getMaxInclusivePerCall(i) < fdo.getInclusivePerCall(i))
                        thread.setMaxInclusivePerCall(i, fdo.getInclusivePerCall(i));
                }
            }

            l = databaseAPI.getAtomicEvents();
            while (l.hasNext()) {
                AtomicEvent ue = (AtomicEvent) l.next();
                this.getTrialData().addUserEvent(ue.getName());
            }

            l = databaseAPI.getAtomicEventData();
            while (l.hasNext()) {
                AtomicLocationProfile alp = (AtomicLocationProfile) l.next();

                // do we need to do this?
                node = this.getNCT().getNode(alp.getNode());
                if (node == null)
                    node = this.getNCT().addNode(alp.getNode());
                context = node.getContext(alp.getContext());
                if (context == null)
                    context = node.addContext(alp.getContext());
                thread = context.getThread(alp.getThread());
                if (thread == null) {
                    thread = context.addThread(alp.getThread(), numberOfMetrics);
                }


                if (thread.getUsereventList() == null) {
                    thread.initializeUsereventList(this.getTrialData().getNumUserEvents());
                    setUserEventsPresent(true);
                }

                userEvent = this.getTrialData().getUserEvent(databaseAPI.getAtomicEvent(alp.getAtomicEventID()).getName());

                userEventProfile = thread.getUserEvent(userEvent.getID());

                if (userEventProfile == null) {
                    userEventProfile = new UserEventProfile(userEvent);
                    thread.addUserEvent(userEventProfile, userEvent.getID());
                }

                userEventProfile.setUserEventNumberValue(alp.getSampleCount());
                userEventProfile.setUserEventMaxValue(alp.getMaximumValue());
                userEventProfile.setUserEventMinValue(alp.getMinimumValue());
                userEventProfile.setUserEventMeanValue(alp.getMeanValue());
                userEventProfile.setUserEventSumSquared(alp.getSumSquared());
                userEventProfile.updateMax();
                
            }

            System.out.println("Processing callpath data ...");
            if (CallPathUtilFuncs.isAvailable(getTrialData().getFunctions())) {
                setCallPathDataPresent(true);
                if (CallPathUtilFuncs.buildRelations(getTrialData()) != 0) {
                    setCallPathDataPresent(false);
                }
            }

            time = (System.currentTimeMillis()) - time;
            System.out.println("Done processing data file!");
            System.out.println("Time to process file (in milliseconds): " + time);
        
    }

    //####################################
    //Instance data.
    //####################################
    private LineData functionDataLine = new LineData();
    private LineData usereventDataLine = new LineData();
    //####################################
    //End - Instance data.
    //####################################
}

/*
 * class FunIndexFunIDPair implements Comparable{ public FunIndexFunIDPair(int
 * functionIndex, int paraProfId){ this.functionIndex = functionIndex;
 * this.paraProfId = paraProfId; }
 * 
 * public int compareTo(Object obj){ return functionIndex -
 * ((FunIndexFunIDPair)obj).functionIndex;}
 * 
 * public int functionIndex; public int paraProfId; }
 */
