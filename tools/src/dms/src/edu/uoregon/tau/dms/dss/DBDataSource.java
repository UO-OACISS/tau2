
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
            }

            //Add the functionProfiles.
            ListIterator l = databaseAPI.getIntervalEvents();
            
            meanData = new Thread(-1,-1,-1, numberOfMetrics);
            totalData = new Thread(-2,-2,-2, numberOfMetrics);
    
            totalItems += this.getNumFunctions();
            
            while (l.hasNext()) {
                IntervalEvent ie = (IntervalEvent) l.next();
                
                function = this.addFunction(ie.getName(), numberOfMetrics);

                FunctionProfile meanProfile = new FunctionProfile(function, numberOfMetrics);
                function.setMeanProfile(meanProfile);
                meanData.addFunctionProfile(meanProfile);

                FunctionProfile totalProfile = new FunctionProfile(function, numberOfMetrics);
                function.setTotalProfile(totalProfile);
                totalData.addFunctionProfile(totalProfile);

                
                //Add element to the localMap for more efficient lookup later
                // in the function.
                //localMap.add(new FunIndexFunIDPair(f.getIndexID(), id));

                IntervalLocationProfile ilp = ie.getMeanSummary();

                if (ie.getGroup() != null) {
                    Group group = this.addGroup(ie.getGroup());
                    function.addGroup(group);
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

                   
                }

                
//                meanData.setThreadDataAllMetrics();


                ilp = ie.getTotalSummary();
                for (int i = 0; i < numberOfMetrics; i++) {
//                    function.setTotalExclusive(i, ilp.getExclusive(i));
//                    function.setTotalExclusivePercent(i,
//                            ilp.getExclusivePercentage(i));
//                    function.setTotalInclusive(i, ilp.getInclusive(i));
//                    function.setTotalInclusivePercent(i,
//                            ilp.getInclusivePercentage(i));
//                    function.setTotalInclusivePerCall(i, ilp.getInclusivePerCall(i));
//                    function.setTotalNumCalls(ilp.getNumCalls());
//                    function.setTotalNumSubr(ilp.getNumSubroutines());


                
                    totalProfile.setExclusive(i, ilp.getExclusive(i));
                    totalProfile.setExclusivePercent(i,
                            ilp.getExclusivePercentage(i));
                    totalProfile.setInclusive(i, ilp.getInclusive(i));
                    totalProfile.setInclusivePercent(i,
                            ilp.getInclusivePercentage(i));
                    totalProfile.setInclusivePerCall(i, ilp.getInclusivePerCall(i));
                    totalProfile.setNumCalls(ilp.getNumCalls());
                    totalProfile.setNumSubr(ilp.getNumSubroutines());

                
                }
            }

            //Collections.sort(localMap);

           

            l = databaseAPI.getIntervalEventData();
            
            while (l.hasNext()) {
                IntervalLocationProfile fdo = (IntervalLocationProfile) l.next();
                node = this.getNode(fdo.getNode());
                if (node == null)
                    node = this.addNode(fdo.getNode());
                context = node.getContext(fdo.getContext());
                if (context == null)
                    context = node.addContext(fdo.getContext());
                thread = context.getThread(fdo.getThread());
                if (thread == null) {
                    thread = context.addThread(fdo.getThread(), numberOfMetrics);
                }

                //Get Function and FunctionProfile.


                function = this.getFunction(databaseAPI.getIntervalEvent(fdo.getIntervalEventID()).getName());
                functionProfile = thread.getFunctionProfile(function);
                
                if (functionProfile == null) {
                    functionProfile = new FunctionProfile(function, numberOfMetrics);
                    thread.addFunctionProfile(functionProfile);
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
//                    if ((function.getMaxExclusive(i)) < fdo.getExclusive(i))
//                        function.setMaxExclusive(i, fdo.getExclusive(i));
//                    if ((function.getMaxExclusivePercent(i)) < fdo.getExclusivePercentage(i))
//                        function.setMaxExclusivePercent(i,
//                                fdo.getExclusivePercentage(i));
//                    if ((function.getMaxInclusive(i)) < fdo.getInclusive(i))
//                        function.setMaxInclusive(i, fdo.getInclusive(i));
//                    if ((function.getMaxInclusivePercent(i)) < fdo.getInclusivePercentage(i))
//                        function.setMaxInclusivePercent(i,
//                                fdo.getInclusivePercentage(i));
//                    if (function.getMaxNumCalls() < fdo.getNumCalls())
//                        function.setMaxNumCalls(fdo.getNumCalls());
//                    if (function.getMaxNumSubr() < fdo.getNumSubroutines())
//                        function.setMaxNumSubr(fdo.getNumSubroutines());
//                    if (function.getMaxInclusivePerCall(i) < fdo.getInclusivePerCall(i))
//                        function.setMaxInclusivePerCall(i, fdo.getInclusivePerCall(i));

//                    if ((thread.getMaxExclusive(i)) < fdo.getExclusive(i))
//                        thread.setMaxExclusive(i, fdo.getExclusive(i));
//                    if ((thread.getMaxExclusivePercent(i)) < fdo.getExclusivePercentage(i))
//                        thread.setMaxExclusivePercent(i, fdo.getExclusivePercentage(i));
//                    if ((thread.getMaxInclusive(i)) < fdo.getInclusive(i))
//                        thread.setMaxInclusive(i, fdo.getInclusive(i));
//                    if ((thread.getMaxInclusivePercent(i)) < fdo.getInclusivePercentage(i))
//                        thread.setMaxInclusivePercent(i, fdo.getInclusivePercentage(i));
//                    if (thread.getMaxNumCalls() < fdo.getNumCalls())
//                        thread.setMaxNumCalls(fdo.getNumCalls());
//                    if (thread.getMaxNumSubr() < fdo.getNumSubroutines())
//                        thread.setMaxNumSubr(fdo.getNumSubroutines());
//                    if (thread.getMaxInclusivePerCall(i) < fdo.getInclusivePerCall(i))
//                        thread.setMaxInclusivePerCall(i, fdo.getInclusivePerCall(i));
                }
            }

            l = databaseAPI.getAtomicEvents();
            while (l.hasNext()) {
                AtomicEvent ue = (AtomicEvent) l.next();
                this.addUserEvent(ue.getName());
            }

            l = databaseAPI.getAtomicEventData();
            while (l.hasNext()) {
                AtomicLocationProfile alp = (AtomicLocationProfile) l.next();

                // do we need to do this?
                node = this.getNode(alp.getNode());
                if (node == null)
                    node = this.addNode(alp.getNode());
                context = node.getContext(alp.getContext());
                if (context == null)
                    context = node.addContext(alp.getContext());
                thread = context.getThread(alp.getThread());
                if (thread == null) {
                    thread = context.addThread(alp.getThread(), numberOfMetrics);
                }


                if (thread.getUserEventProfiles() == null) {
                    setUserEventsPresent(true);
                }

                userEvent = this.getUserEvent(databaseAPI.getAtomicEvent(alp.getAtomicEventID()).getName());

                userEventProfile = thread.getUserEventProfile(userEvent);

                if (userEventProfile == null) {
                    userEventProfile = new UserEventProfile(userEvent);
                    thread.addUserEvent(userEventProfile);
                }

                userEventProfile.setUserEventNumberValue(alp.getSampleCount());
                userEventProfile.setUserEventMaxValue(alp.getMaximumValue());
                userEventProfile.setUserEventMinValue(alp.getMinimumValue());
                userEventProfile.setUserEventMeanValue(alp.getMeanValue());
                userEventProfile.setUserEventSumSquared(alp.getSumSquared());
                userEventProfile.updateMax();
                
            }

            for (Enumeration e = this.getThreads().elements(); e.hasMoreElements();) {
                ((Thread) e.nextElement()).setThreadDataAllMetrics();
            }
            this.meanData.setThreadDataAllMetrics();


            //System.out.println("Processing callpath data ...");
            if (CallPathUtilFuncs.checkCallPathsPresent(this.getFunctions())) {
                setCallPathDataPresent(true);
            }

            time = (System.currentTimeMillis()) - time;
            //System.out.println("Done processing data file!");
            //System.out.println("Time to process file (in milliseconds): " + time);
        
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
