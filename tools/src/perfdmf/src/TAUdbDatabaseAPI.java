package edu.uoregon.tau.perfdmf;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import edu.uoregon.tau.perfdmf.database.DB;

public class TAUdbDatabaseAPI {

	public static int uploadTrial(DB db, Trial trial) {

        DataSource dataSource = trial.getDataSource();

        try {
            db.setAutoCommit(false);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: couldn't set AutoCommit to false", e);
        }

        int newTrialID = -1;

        try {
            // save the trial metadata (which returns the new id)
            newTrialID = trial.saveTrial(db);
            trial.setID(newTrialID);
            
            //TODO: Deal with upload size, I think this is for the progress bar.
          //  computeUploadSize(dataSource);
            
            // upload the metrics and get a map that maps the metrics 0 -> n-1 to their unique DB IDs (e.g. 83, 84)
            Map<Metric, Integer> metricMap = uploadMetrics(newTrialID, dataSource.getMetrics(), db);

            Map<Function, Integer> functionMap = uploadFunctions(newTrialID, dataSource);

            uploadFunctionProfiles(newTrialID, dataSource, functionMap, metricMap, summaryOnly);

            Map<UserEvent, Integer> userEventMap = uploadUserEvents(newTrialID, dataSource);

            uploadUserEventProfiles(newTrialID, dataSource, userEventMap);

            if (this.cancelUpload) {
                db.rollback();
                deleteTrial(newTrialID);
                return -1;
            }

        } catch (SQLException e) {
            try {
                db.rollback();
                e.printStackTrace();
                throw new DatabaseException("Saving Trial Failed, rollbacks successful", e);
            } catch (SQLException e2) {
                throw new DatabaseException("Saving Trial Failed, rollbacks failed!", e2);
            }

        }

        try {
            db.commit();
            db.setAutoCommit(true);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: commit failed!", e);
        }

        //long stop = System.currentTimeMillis();
        //long elapsedMillis = stop - start;
        //double elapsedSeconds = (double) (elapsedMillis) / 1000.0;
        //        System.out.println("Elapsed time: " + elapsedSeconds + " seconds.");
        return newTrialID;
		return 0;
	}
	  private static Map<Metric, Integer> uploadMetrics(int trialID,
			List<Metric> metrics, DB db) throws SQLException {
		  Map<Metric, Integer> map = new HashMap<Metric, Integer>();
		for (Metric metric: metrics){
	            PreparedStatement stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
	                    + "metric (name, trial) VALUES (?, ?)");
	            stmt.setString(1, metric.getName());
	            stmt.setInt(2, trialID);
	            stmt.executeUpdate();
	            stmt.close();

	            String tmpStr = new String();
	            if (db.getDBType().compareTo("mysql") == 0)
	                tmpStr = "select LAST_INSERT_ID();";
	            else if (db.getDBType().compareTo("db2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
	            else if (db.getDBType().compareTo("derby") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
	            else if (db.getDBType().compareTo("h2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
	            else if (db.getDBType().compareTo("oracle") == 0)
	                tmpStr = "select " + db.getSchemaPrefix() + "metric_id_seq.currval FROM dual";
	            else
	                tmpStr = "select currval('metric_id_seq');";
	            int dbMetricID = Integer.parseInt(db.getDataItem(tmpStr));
	            map .put(metric, new Integer(dbMetricID));
	            metric.setDbMetricID(dbMetricID);
	        }
	        return map;
	    }
	  // fills the timer table
	    private Map<Function, Integer> uploadFunctions(int trialID, DataSource dataSource) throws SQLException {
	        Map<Function, Integer> map = new HashMap<Function, Integer>();

	        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
	        for (Iterator<Function> it = dataSource.getFunctions(); it.hasNext();) {
	            Function f = it.next();
	            if (f.isGroupMember(derived)) {
	                continue;
	            }

	            String group = null;
	            List<Group> groups = f.getGroups();
	            StringBuffer allGroups = new StringBuffer();
	            if (groups != null) {
	                for (int i = 0; i < groups.size(); i++) {
	                    if (i > 0)
	                        allGroups.append("|");
	                    allGroups.append(groups.get(i).getName());
	                }
	                if (groups.size() > 0)
	                    group = allGroups.toString();
	            }

	            PreparedStatement statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
	                    + "timer (trial, name, group_name) VALUES (?, ?, ?)");
	            statement.setInt(1, trialID);
	            statement.setString(2, f.getName());
	            statement.setString(3, group);
	            statement.executeUpdate();
	            statement.close();

	            String tmpStr = new String();
	            if (db.getDBType().compareTo("mysql") == 0)
	                tmpStr = "select LAST_INSERT_ID();";
	            else if (db.getDBType().compareTo("db2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
	            else if (db.getDBType().compareTo("derby") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
	            else if (db.getDBType().compareTo("h2") == 0)
	                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
	            else if (db.getDBType().compareTo("oracle") == 0)
	                tmpStr = "select " + db.getSchemaPrefix() + "interval_event_id_seq.currval FROM dual";
	            else
	                tmpStr = "select currval('interval_event_id_seq');";
	            int newIntervalEventID = Integer.parseInt(db.getDataItem(tmpStr));

	            map.put(f, new Integer(newIntervalEventID));

	            this.itemsDone++;
	        }
	        return map;
	    }

}
