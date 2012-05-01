package edu.uoregon.tau.perfdmf;


import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;

import edu.uoregon.tau.perfdmf.database.DB;

/**
 * Reads a single trial from the database
 */
public class TAUdbDataSource extends DataSource {

    private DatabaseAPI databaseAPI;  

    public TAUdbDataSource(DatabaseAPI dbAPI) {
        super();
        this.setMetrics(new Vector<Metric>());
        this.databaseAPI = dbAPI;
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
				db.getSchemaPrefix() +" timer_callpath cp on v.timer = cp.timer and v.thread = cp.thread where t.trial = "+trialID;



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
        List<IntervalEvent> intervalEvents = databaseAPI.getIntervalEvents();
        ListIterator<IntervalEvent> lIE = intervalEvents.listIterator();
        while (lIE.hasNext()) {
            IntervalEvent ie = lIE.next();
            Function function = this.addFunction(ie.getName(), numberOfMetrics);
            addGroups(ie.getGroup(), function);
            ieMap.put(new Integer(ie.getID()), function);
        }
        databaseAPI.getTrial().setFunctionMap(ieMap);

        //getIntervalEventData(ieMap);
        fastGetIntervalEventData(trialID,ieMap, metricMap);

        // map Interval Event ID's to Function objects
        Map<Integer, UserEvent> aeMap = new HashMap<Integer, UserEvent>();
        ListIterator<AtomicEvent> lAE = databaseAPI.getAtomicEvents().listIterator();
        while (lAE.hasNext()) {
            AtomicEvent atomicEvent = lAE.next();
            UserEvent userEvent = addUserEvent(atomicEvent.getName());
            aeMap.put(new Integer(atomicEvent.getID()), userEvent);
        }

        ListIterator<AtomicLocationProfile> lAD = databaseAPI.getAtomicEventData().listIterator();
        while (lAD.hasNext()) {
            AtomicLocationProfile alp = lAD.next();
            Thread thread = addThread(alp.getNode(), alp.getContext(), alp.getThread());
            UserEvent userEvent = aeMap.get(new Integer(alp.getAtomicEventID()));
            UserEventProfile userEventProfile = thread.getUserEventProfile(userEvent);

            if (userEventProfile == null) {
                userEventProfile = new UserEventProfile(userEvent);
                thread.addUserEventProfile(userEventProfile);
            }

            userEventProfile.setNumSamples(alp.getSampleCount());
            userEventProfile.setMaxValue(alp.getMaximumValue());
            userEventProfile.setMinValue(alp.getMinimumValue());
            userEventProfile.setMeanValue(alp.getMeanValue());
            userEventProfile.setSumSquared(alp.getSumSquared());
            userEventProfile.updateMax();
        }

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
//    public static List<Trial> getTrialsForTAUdbView (List<RMIView> views, DB db) {
//		//PerfExplorerOutput.println("getTrialsForView()...");
//		List<Trial> trials = new ArrayList<Trial>();
//		try {
//			
//			StringBuilder sql = new StringBuilder();
//			sql.append("select conjoin, taudb_view, table_name, column_name, operator, value from taudb_view left outer join taudb_view_parameter on taudb_view.id = taudb_view_parameter.taudb_view where taudb_view.id in (");
//			for (int i = 0 ; i < views.size(); i++) {
//				if (i > 0)
//					sql.append(",?");
//				else
//					sql.append("?");
//			}
//			sql.append(") order by taudb_view.id");
//			PreparedStatement statement = db.prepareStatement(sql.toString());
//			int i = 1;
//			for (RMIView view : views) {
//				statement.setInt(1, Integer.valueOf(view.getField("ID")));
//				i++;
//			}
//			ResultSet results = statement.executeQuery();
//			
//			StringBuilder whereClause = new StringBuilder();
//			StringBuilder joinClause = new StringBuilder();
//			int currentView = 0;
//			int alias = 0;
//			String conjoin = " where ";
//			while (results.next() != false) {
//				int viewid = results.getInt(2);
//				String tableName = results.getString(3);
//				if (tableName == null) 
//					break;
//				String columnName = results.getString(4);
//				String operator = results.getString(5);
//				String value = results.getString(6);
//				if ((currentView > 0) && (currentView != viewid)) {
//					conjoin = " and ";
//				} else if (currentView == viewid) {
//					conjoin = " " + results.getString(1) + " ";
//				}
//				if (tableName.equalsIgnoreCase("trial")) {
//					whereClause.append(conjoin + tableName + "." + columnName + " " + operator + " " + "'" + value + "'");
//				} else {
//					// otherwise, we have primary_metadata or secondary_metadata
//					joinClause.append(" left outer join " + tableName + " t" + alias + " on t.id = t" + alias + ".trial");
//					whereClause.append(conjoin + "t" + alias + "." + columnName);
//					if (db.getDBType().compareTo("db2") == 0) {
//						whereClause.append(" as varchar(256)) ");
//					}
//					whereClause.append(" " + operator + " " + "'" + value + "'");
//				}
//				alias++;
//				currentView = viewid;
//			}
//			statement.close();
//			
//			//PerfExplorerOutput.println(whereClause.toString());
//			trials = Trial.getTrialList(db, joinClause.toString() + " " + whereClause.toString(), false);
//		} catch (Exception e) {
//			String error = "ERROR: Couldn't select views from the database!";
//			System.err.println(error);
//			e.printStackTrace();
//		}
//		return trials;
//	}


}
