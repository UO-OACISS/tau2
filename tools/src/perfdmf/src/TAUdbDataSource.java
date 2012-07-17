package edu.uoregon.tau.perfdmf;


import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;

import edu.uoregon.tau.perfdmf.database.DB;

/**
 * Reads a single trial from the database
 */
public class TAUdbDataSource extends DataSource {

    private TAUdbDatabaseAPI databaseAPI;  

    public TAUdbDataSource(DatabaseAPI dbAPI) {
        super();
        this.setMetrics(new Vector<Metric>());
        this.databaseAPI = new TAUdbDatabaseAPI(dbAPI);
    }

    public int getProgress() {
        return 0;
        //return DatabaseAPI.getProgress();
    }

    public void cancelLoad() {
        //abort = true;
        return;
    }


    private void fastGetIntervalEventData(int trialID, Map<Integer, Function> ieMap, Map<Integer, Metric> metricMap) throws SQLException {
        int numMetrics = getNumberOfMetrics();
        DB db = databaseAPI.getDb();

 
		String buf = "select v.timer, v.metric, h.node_rank as node, h.context_rank as context, h.thread_rank as thread, " +
				"v.inclusive_value as inclusive, v.exclusive_value as inclusive, cp.calls, cp.subroutines " +
				"from timer_value v left outer join " +
				db.getSchemaPrefix() +"timer t on v.timer = t.id " +
				"left outer join " +
				db.getSchemaPrefix() +"thread h on v.thread = h.id left outer join" +
				db.getSchemaPrefix() +" timer_callpath cp on v.timer = cp.timer and v.thread = cp.thread where h.node_rank > -1 and t.trial = "+trialID;

        /*
         1 - timer
         2 - metric
         3 - node
         4 - context
         5 - thread
         6 - inclusive
         7 - exclusive
         8 - num_calls
         9 - num_subrs
         */

        // get the results
        long time = System.currentTimeMillis();
        //System.out.println(buf.toString());
        ResultSet resultSet = db.executeQuery(buf.toString());
        time = (System.currentTimeMillis()) - time;
        //System.out.println("Query : " + time);
        //System.out.print(time + ", ");

        time = System.currentTimeMillis();
        while (resultSet.next() != false) {
//SELECT node, context, thread, metric, incl, exc, calls, sub FROM timer_values v, JOIN timer t ON v.timer=t.id
            int intervalEventID = resultSet.getInt(1);
            Function function = ieMap.get(new Integer(intervalEventID));

            int nodeID = resultSet.getInt(3);
            int contextID = resultSet.getInt(4);
            int threadID = resultSet.getInt(5);

            Thread thread = addThread(nodeID, contextID, threadID);
            FunctionProfile functionProfile = thread.getFunctionProfile(function);

            if (functionProfile == null) {
                functionProfile = new FunctionProfile(function, numMetrics);
                thread.addFunctionProfile(functionProfile);
            }

            int metricIndex = metricMap.get(new Integer(resultSet.getInt(2))).getID();
            double inclusive, exclusive;

            inclusive = resultSet.getDouble(6);
            exclusive = resultSet.getDouble(7);
            double numcalls = resultSet.getDouble(8);
            double numsubr = resultSet.getDouble(9);

            functionProfile.setNumCalls(numcalls);
            functionProfile.setNumSubr(numsubr);
            functionProfile.setExclusive(metricIndex, exclusive);
            functionProfile.setInclusive(metricIndex, inclusive);
        }
        time = (System.currentTimeMillis()) - time;
        //System.out.println("Processing : " + time);
        //System.out.print(time + ", ");

        resultSet.close();
    }

  
    public void load() throws SQLException {

        //System.out.println("Processing data, please wait ......");
        long time = System.currentTimeMillis();
        int trialID = databaseAPI.getTrial().getID();
        databaseAPI.getTrial().setDataSource(this);
        DB db = databaseAPI.getDb();
        StringBuffer joe = new StringBuffer();
        joe.append("SELECT id, name ");
        joe.append("FROM " + db.getSchemaPrefix() + "metric ");
        joe.append("WHERE trial = ");
        joe.append(databaseAPI.getTrial().getID());
        joe.append(" ORDER BY id ");

        Map<Integer, Metric> metricMap = new HashMap<Integer, Metric>();

        ResultSet resultSet = db.executeQuery(joe.toString());
        int numberOfMetrics = 0;
        while (resultSet.next() != false) {
            int id = resultSet.getInt(1);
            String name = resultSet.getString(2);
            Metric metric = this.addMetricNoCheck(name);
            metric.setDbMetricID(id);
            metricMap.put(new Integer(id), metric);
            numberOfMetrics++;
        }
        resultSet.close();

        // map Interval Event ID's to Function objects
        Map<Integer, Function> ieMap = new HashMap<Integer, Function>();

        // iterate over interval events (functions), create the function objects and add them to the map
        ieMap = databaseAPI.getIntervalEvents(this, numberOfMetrics);

        //getIntervalEventData(ieMap);
        fastGetIntervalEventData(trialID,ieMap, metricMap);

        // get the user event counters and counter data
        databaseAPI.getAtomicEvents();
        databaseAPI.getAtomicEventData(this);

       //downloadMetaData();
        Trial t = databaseAPI.getTrial();
        databaseAPI.getTrial().loadXMLMetadata(db);
        //ParaProf uses the metadata in the datas ource to load the side bar rather than 
        //what's in the trial so you have to do both.
        this.setMetaData(t.getMetaData());
        databaseAPI.terminate();
        time = (System.currentTimeMillis()) - time;
        //System.out.println("Time to download file (in milliseconds): " + time);
        //System.out.println(time);

        // We actually discard the mean and total values by calling this
        // But, we need to compute other statistics anyway
        //TODO Deal with derived data.  Most of it will be saved in the DB?
        generateDerivedData();
    }

}
