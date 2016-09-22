package edu.uoregon.tau.perfdmf;

import java.sql.DatabaseMetaData;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import edu.uoregon.tau.perfdmf.database.DB;

/**
 * Holds all the data for an interval_event in the database.
 * This object is returned by the DataSession class and all of its subtypes.
 * The IntervalEvent object contains all the information associated with
 * an intervalEvent from which the TAU performance data has been generated.
 * A intervalEvent is associated with one trial, experiment and application, and has one or more
 * IntervalLocationProfile objects (one for each node/context/thread location in the trial) associated with it.  
 * <p>
 * An interval event has information
 * related to one particular interval event in the application, including the name of the interval event,
 * the TAU group it belongs to, and all of the total and mean data for the interval event. 
 * In order to see particular measurements for a node/context/thread/metric instance,
 * get the IntervalLocationProfile(s) for this IntervalEvent.  In order to access the total
 * or mean data, getTotalSummary() and getMeanSummary() methods are provided.  The
 * index of the metric in the Trial object should be used to indicate which total/mean
 * summary object to return.
 *
 * <P>CVS $Id: IntervalEvent.java,v 1.8 2009/11/13 00:11:41 amorris Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getIntervalEvents
 * @see		DataSession#setIntervalEvent
 * @see		Application
 * @see		Experiment
 * @see		Trial
 * @see		IntervalLocationProfile
 */
public class IntervalEvent {
	// can't instantiate this class!
	private IntervalEvent() {
    }

    // returns a Vector of IntervalEvents
    public static Map<Integer, Function> getIntervalEvents(DatabaseAPI dataSession, DB db, String whereClause, DataSource dataSource, int numberOfMetrics) {
    	if(db.getSchemaVersion() >0) return getTAUdbIntervalEvents(dataSession,db,  whereClause, dataSource, numberOfMetrics);
        Map<Integer, Function> events = new HashMap<Integer, Function>();
        // create a string to hit the database
        StringBuffer buf = new StringBuffer();
        buf.append("select id, name, group_name, trial ");
        buf.append("from " + db.getSchemaPrefix() + "interval_event ");
        buf.append(whereClause);

        if (db.getDBType().compareTo("oracle") == 0) {
            buf.append(" order by dbms_lob.substr(name) asc");
        } else if (db.getDBType().compareTo("derby") == 0) {
            buf.append(" order by cast (name as varchar(4000)) asc");
        } else if (db.getDBType().compareTo("db2") == 0) {
            buf.append(" order by cast (name as varchar(256)) asc");
        } else {
            buf.append(" order by name asc ");
        }

        // System.out.println(buf.toString());

        // get the results
        try {
            ResultSet resultSet = db.executeQuery(buf.toString());

            //IntervalEvent tmpIntervalEvent = null;
            while (resultSet.next() != false) {
                int id = resultSet.getInt(1);
                String name = resultSet.getString(2);
                String groups = resultSet.getString(3);
                int trialID = resultSet.getInt(4);
                Function function = dataSource.addFunction(name, numberOfMetrics);
//                function.setID(id);
                dataSource.addGroups(groups, function);
                events.put(id, function);
            }
            resultSet.close();
        } catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }

        return events;
    }

    private static Map<Integer, Function> getTAUdbIntervalEvents(
			DatabaseAPI dataSession, DB db, String whereClause, DataSource dataSource, int numberOfMetrics) {
//    	SELECT timer.id, timer.trial, timer.name, timer_group.group_name 
//    	FROM timer
//    	LEFT JOIN timer_group
//    	ON timer.id=timer_group.timer
    	
    	 Map<Integer, Function> events = new HashMap<Integer, Function>();
         // create a string to hit the database
         StringBuffer buf = new StringBuffer();
         buf.append("with recursive cp (id, parent, timer, name) as (");
         /* flat timer part */
         buf.append("SELECT tc.id, tc.parent, tc.timer, t.name FROM ");
         buf.append(db.getSchemaPrefix());
         buf.append("timer_callpath tc INNER JOIN ");
         buf.append(db.getSchemaPrefix());
         buf.append("timer t on tc.timer = t.id WHERE ");
         if (dataSession.getTrial() != null) {
        	 buf.append(" t.trial = " + dataSession.getTrial().getID() + " AND ");
         }
         buf.append(" tc.parent is null ");
         buf.append("UNION ALL ");
         /* recursive part */
         buf.append("SELECT d.id, d.parent, d.timer, ");
		        if (db.getDBType().compareTo("h2") == 0) {
					buf.append("concat (cp.name, ' => ', dt.name) FROM ");
		        } else {
					buf.append("cp.name || ' => ' || dt.name FROM ");
		        }
				buf.append(db.getSchemaPrefix());
         buf.append("timer_callpath AS d JOIN cp on (d.parent = cp.id) JOIN ");
         buf.append(db.getSchemaPrefix());
         buf.append("timer dt on d.timer = dt.id ");
         if (dataSession.getTrial() != null) {
        	 buf.append("where dt.trial = " + dataSession.getTrial().getID() + " ");
         }
         buf.append(") ");
         buf.append("SELECT distinct cp.id, cp.timer, cp.name, t.short_name, t.source_file, t.line_number, ");
         buf.append("t.line_number_end, t.column_number, t.column_number_end, g.group_name, t.trial FROM cp join ");
         buf.append(db.getSchemaPrefix());
         buf.append("timer t on cp.timer = t.id join ");
         buf.append(db.getSchemaPrefix());
         buf.append("timer_group g on t.id = g.timer ");
         buf.append(whereClause);

         if (db.getDBType().compareTo("oracle") == 0) {
             buf.append(" order by dbms_lob.substr(name) asc");
         } else if (db.getDBType().compareTo("derby") == 0) {
             buf.append(" order by cast (name as varchar(4000)) asc");
         } else if (db.getDBType().compareTo("db2") == 0) {
             buf.append(" order by cast (name as varchar(256)) asc");
         } else {
             buf.append(" order by name asc ");
         }

         // get the results
         try {
        	 //System.out.println(buf.toString());
             ResultSet resultSet = db.executeQuery(buf.toString());
             //IntervalEvent tmpIntervalEvent = null;
             Function last = null;
             while (resultSet.next() != false) {
                 int id = resultSet.getInt(1);
                 String name = resultSet.getString(3);
                 String group = resultSet.getString(10);
                 int trialID = resultSet.getInt(11);
                 last = events.get(id);
                 
                 boolean isCallpath = false;
                 if (name!=null&&name.contains(" => ")) {
                 	isCallpath = true;
                 	if(group!=null&&!group.contains("TAU_CALLPATH")){
                 		group = group+"|TAU_CALLPATH";
                 	}
                 }
                 
                 if (last != null) {
                	 dataSource.addGroups(group, last);
                 } else {
                    Function function = dataSource.addFunction(name, numberOfMetrics);
                    function.setDatabaseID(id);
                    dataSource.addGroups(group, function);
                    events.put(id, function);
                    SourceRegion sourceRegion = new SourceRegion();
                    sourceRegion.setFilename(resultSet.getString(5));
                    sourceRegion.setStartLine(resultSet.getInt(6));
                    sourceRegion.setEndLine(resultSet.getInt(7));
                    sourceRegion.setStartColumn(resultSet.getInt(8));
                    sourceRegion.setEndColumn(resultSet.getInt(9));
                    function.callpathFunction = isCallpath;
                    function.setShortName(resultSet.getString(4));
                    function.setSourceRegion(sourceRegion);
                 }
             }
             resultSet.close();
         } catch (Exception ex) {
             ex.printStackTrace();
             return null;
         }

         return events;
	}

	public static int saveIntervalEvent(DB db, int newTrialID, Function function, Hashtable<Integer, Integer> newMetHash, int saveMetricIndex)
            throws SQLException {
    	if(db.getSchemaVersion() >0) return saveTAUdbIntervalEvent(db, newTrialID, function, newMetHash, saveMetricIndex);
        int newIntervalEventID = -1;

        PreparedStatement statement = null;
        if (saveMetricIndex < 0) {
            //		statement = db.prepareStatement("INSERT INTO interval_event (trial, name, group_name) VALUES (?, ?, ?)");
            statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                    + "interval_event (trial, name, group_name) VALUES (?, ?, ?)");
            statement.setInt(1, newTrialID);
            statement.setString(2, function.getName());
            statement.setString(3, function.getGroupString());
            statement.executeUpdate();
            statement.close();

            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0)
                tmpStr = "select LAST_INSERT_ID();";
            else if (db.getDBType().compareTo("db2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
            else if (db.getDBType().compareTo("sqlite") == 0)
                tmpStr = "select seq from sqlite_sequence where name = 'interval_event'";
            else if (db.getDBType().compareTo("derby") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
            else if (db.getDBType().compareTo("h2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
            else if (db.getDBType().compareTo("oracle") == 0)
                tmpStr = "select " + db.getSchemaPrefix() + "interval_event_id_seq.currval FROM dual";
            else // postgres
                tmpStr = "select currval('interval_event_id_seq');";
            newIntervalEventID = Integer.parseInt(db.getDataItem(tmpStr));
        } else {

            if (db.getDBType().compareTo("oracle") == 0)
                statement = db.prepareStatement("SELECT id FROM " + db.getSchemaPrefix()
                        + "interval_event where dbms_lob.instr(name, ?) > 0 and trial = ?");
            else if (db.getDBType().compareTo("derby") == 0)
                statement = db.prepareStatement("SELECT id FROM " + db.getSchemaPrefix()
                        + "interval_event where cast(name as varchar(4000)) = ? and trial = ?");
            else
                statement = db.prepareStatement("SELECT id FROM " + db.getSchemaPrefix()
                        + "interval_event where name = ? and trial = ?");

            statement.setString(1, function.getName());
            statement.setInt(2, newTrialID);
            ResultSet resultSet = statement.executeQuery();
            while (resultSet.next() != false) {
                newIntervalEventID = resultSet.getInt(1);
            }
            resultSet.close();
            statement.close();
        }
        
        if (newIntervalEventID == -1) {
            throw new RuntimeException("Unable to find event in database, event: " + function.getName(), null);
        }

        // save the intervalEvent mean summary
        if (function.getMeanProfile() != null) {
        	System.out.println("TODO! SAVE THE MEAN SUMMARY!");
            //meanSummary.saveMeanSummary(db, newIntervalEventID, newMetHash, saveMetricIndex);
        }

        // save the intervalEvent total summary
        if (function.getTotalProfile() != null) {
        	System.out.println("TODO! SAVE THE TOTAL SUMMARY!");
            //totalSummary.saveTotalSummary(db, newIntervalEventID, newMetHash, saveMetricIndex);
        }
        return newIntervalEventID;
    }
    
	public static int saveTAUdbIntervalEvent(DB db, int newTrialID, Function function, Hashtable<Integer, Integer> newMetHash, int saveMetricIndex)
            throws SQLException {
        int newIntervalEventID = -1;

        PreparedStatement statement = null;
        if (saveMetricIndex < 0) {
            statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                    + "timer (trial, name) VALUES (?, ?)");
            statement.setInt(1, newTrialID);
            statement.setString(2, function.getName());
            // TODO: What about short_name, file, column, line, etc?
            statement.executeUpdate();
            statement.close();

            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0)
                tmpStr = "select LAST_INSERT_ID();";
            else if (db.getDBType().compareTo("db2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
            else if (db.getDBType().compareTo("sqlite") == 0)
                tmpStr = "select seq from sqlite_sequence where name = 'interval_event'";
            else if (db.getDBType().compareTo("derby") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
            else if (db.getDBType().compareTo("h2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
            else if (db.getDBType().compareTo("oracle") == 0)
                tmpStr = "select " + db.getSchemaPrefix() + "interval_event_id_seq.currval FROM dual";
            else // postgres
                tmpStr = "select currval('interval_event_id_seq');";
            newIntervalEventID = Integer.parseInt(db.getDataItem(tmpStr));

			// save the groups
            List<Group> groups = function.getGroups();
            statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
               	+ "timer_group (timer, group_name) VALUES (?, ?)");
			for (int i = 0 ; i < groups.size() ; i++) {
				statement.setInt(1,newIntervalEventID);
				statement.setString(2,groups.get(i).getName().trim());
				statement.addBatch();
			}
			statement.executeBatch();
			statement.close();
        } else {
        	StringBuilder query = new StringBuilder();
        	query.append("with recursive cp (id, parent, timer, name) as " +
        			"(SELECT tc.id, tc.parent, tc.timer, t.name FROM timer_callpath tc " + 
        			"INNER JOIN timer t on tc.timer = t.id WHERE t.trial = " + 
        			newTrialID + " and tc.parent is null " +
        			"UNION ALL SELECT d.id, d.parent, d.timer, ");
		        if (db.getDBType().compareTo("h2") == 0) {
					query.append("concat (cp.name, ' => ', dt.name) FROM ");
		        } else {
					query.append("cp.name || ' => ' || dt.name FROM ");
		        }
				query.append("timer_callpath AS d JOIN cp on (d.parent = cp.id) " +
        			"JOIN timer dt on d.timer = dt.id " +
					"where dt.trial = " + newTrialID + " ) " +
        			"SELECT distinct cp.id, cp.timer, cp.name, t.trial " +
        			"FROM cp join timer t on cp.timer = t.id " +
        			"join timer_group g on t.id = g.timer WHERE trial = ? and ");
            if (db.getDBType().compareTo("oracle") == 0)
                statement = db.prepareStatement(query.toString() + "dbms_lob.instr(cp.name, ?) > 0");
            else if (db.getDBType().compareTo("derby") == 0)
                statement = db.prepareStatement(query.toString() + "cast(cp.name as varchar(4000)) = ?");
            else
            	statement = db.prepareStatement(query.toString() + "cp.name = ?");

            statement.setInt(1, newTrialID);
            statement.setString(2, function.getName());
            ResultSet resultSet = statement.executeQuery();
            while (resultSet.next() != false) {
                newIntervalEventID = resultSet.getInt(1);
            }
            resultSet.close();
            statement.close();
        }
        
        if (newIntervalEventID == -1) {
            throw new RuntimeException("Unable to find event in database, event: " + function.getName(), null);
        }

        // TODO: mean, total. etc should all be saved when aggregate threads are saved.
/*        // save the intervalEvent mean summary
        if (meanSummary != null) {
            meanSummary.saveMeanSummary(db, newIntervalEventID, newMetHash, saveMetricIndex);
        }

        // save the intervalEvent total summary
        if (totalSummary != null) {
            totalSummary.saveTotalSummary(db, newIntervalEventID, newMetHash, saveMetricIndex);
        }
*/
        return newIntervalEventID;
    }
    
    public static void getMetaData(DB db) {
        // see if we've already have them
        // need to load each time in case we are working with a new database. 
        //        if (Trial.fieldNames != null)
        //            return;

        try {
            ResultSet resultSet = null;

            //String trialFieldNames[] = null;
            //int trialFieldTypes[] = null;

            DatabaseMetaData dbMeta = db.getMetaData();

            if ((db.getDBType().compareTo("oracle") == 0) || (db.getDBType().compareTo("derby") == 0)
                    || (db.getDBType().compareTo("h2") == 0)
                    || (db.getDBType().compareTo("db2") == 0)) {
                resultSet = dbMeta.getColumns(null, null, "INTERVAL_EVENT", "%");
            } else {
                resultSet = dbMeta.getColumns(null, null, "interval_event", "%");
            }

            Vector<String> nameList = new Vector<String>();
            Vector<Integer> typeList = new Vector<Integer>();
            List<String> typeNames = new ArrayList<String>();
            List<Integer> columnSizes = new ArrayList<Integer>();
            boolean seenID = false;

            ResultSetMetaData md = resultSet.getMetaData();
            for (int i = 0 ; i < md.getColumnCount() ; i++) {
            	//System.out.println(md.getColumnName(i));
            }

            while (resultSet.next() != false) {

                int ctype = resultSet.getInt("DATA_TYPE");
                String cname = resultSet.getString("COLUMN_NAME");
                String typename = resultSet.getString("TYPE_NAME");
                Integer size = new Integer(resultSet.getInt("COLUMN_SIZE"));

                // this code is because of a bug in derby...
                if (cname.equals("ID")) {
                    if (!seenID)
                        seenID = true;
                    else
                        break;
                }

                nameList.add(resultSet.getString("COLUMN_NAME"));
                typeList.add(new Integer(ctype));
                typeNames.add(typename);
                columnSizes.add(size);
            }
            resultSet.close();

            String[] fieldNames = new String[nameList.size()];
            int[] fieldTypes = new int[typeList.size()];
            String[] fieldTypeNames = new String[typeList.size()];
            for (int i = 0; i < typeList.size(); i++) {
                fieldNames[i] = nameList.get(i);
                fieldTypes[i] = typeList.get(i).intValue();
                if (columnSizes.get(i).intValue() > 255) {
                    fieldTypeNames[i] = typeNames.get(i) + "(" + columnSizes.get(i).toString() + ")";
                } else {
                    fieldTypeNames[i] = typeNames.get(i);
                }
            }

            db.getDatabase().setIntervalEventFieldNames(fieldNames);
            db.getDatabase().setIntervalEventFieldTypeNames(fieldTypeNames);
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

}
