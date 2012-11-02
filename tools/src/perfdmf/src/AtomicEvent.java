package edu.uoregon.tau.perfdmf;

import java.sql.DatabaseMetaData;
import java.sql.ResultSetMetaData;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;
import java.util.List;
import java.util.ArrayList;

import edu.uoregon.tau.perfdmf.database.DB;

/**
 * Holds all the data for a atomic event in the database.
 * This object is returned by the DataSession class and all of its subtypes.
 * The AtomicEvent object contains all the information associated with
 * an atomic event from which the TAU performance data has been generated.
 * A atomic event is associated with one trial, experiment and application, and has one or more
 * AtomicEventData objects (one for each metric in the trial) associated with it.  
 * <p>
 * A atomic event has particular information, including the name of the atomic event, 
 * the TAU group, and the application, experiment and trial IDs.
 *
 * <P>CVS $Id: AtomicEvent.java,v 1.4 2008/03/13 23:15:15 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getAtomicEvents
 * @see		DataSession#setAtomicEvent
 * @see		Application
 * @see		Experiment
 * @see		Trial
 * @see		AtomicLocationProfile
 */
public class AtomicEvent {

	// this class is only static methods!
    private AtomicEvent() {
    }

    // returns a Vector of UserEvents
    public static Map<Integer, UserEvent> getAtomicEvents(DatabaseAPI dataSession, DB db, String whereClause) {
    	if(db.getSchemaVersion()>0) return getTAUdbAtomicEvents(dataSession,db, whereClause);
        Map<Integer,UserEvent> atomicEvents = new HashMap<Integer,UserEvent>();
        // create a string to hit the database
        StringBuffer buf = new StringBuffer();
        buf.append("select u.id, u.trial, u.name ");
        buf.append("from " + db.getSchemaPrefix() + "atomic_event u inner join "
                + db.getSchemaPrefix() + "trial t on u.trial = t.id ");
        buf.append("inner join " + db.getSchemaPrefix() + "experiment e on t.experiment = e.id ");
        buf.append(whereClause);
        buf.append(" order by u.id ");
        // System.out.println(buf.toString());

        // get the results
        try {
            ResultSet resultSet = db.executeQuery(buf.toString());
            //UserEvent tmpUserEvent = null;
            while (resultSet.next() != false) {
            	int id = resultSet.getInt(1);
            	String name = resultSet.getString(3);
            	// this is UGLY
            	UserEvent ue = dataSession.getTrial().getDataSource().addUserEvent(name);
                atomicEvents.put(id, ue);
            }
            resultSet.close();
        } catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }

        return atomicEvents;
    }

    private static Map<Integer,UserEvent> getTAUdbAtomicEvents(
			DatabaseAPI datasource, DB db, String whereClause) {
        Map<Integer,UserEvent> atomicEvents = new HashMap<Integer,UserEvent>();
        // create a string to hit the database
        StringBuffer buf = new StringBuffer();
        buf.append("select u.id, u.trial, u.name ");
        buf.append("from " + db.getSchemaPrefix() + "counter u ");
        buf.append(whereClause);
        buf.append(" order by u.id ");
        //System.out.println(buf.toString());

        // get the results
        try {
            ResultSet resultSet = db.executeQuery(buf.toString());
            //UserEvent tmpUserEvent = null;
            while (resultSet.next() != false) {
                int id = resultSet.getInt(1);
                String name = resultSet.getString(3);
            	UserEvent ue = datasource.getTrial().getDataSource().addUserEvent(name);
//                UserEvent ue = new UserEvent(name, id);
            	/*
                int parent = resultSet.getInt(4);
                if (parent > 0) {
            		// TODO - SET THE PARENT! But we can't because we don't
            		// yet have a map of parents to Function/IntervalEvent objects.
                	Function f = datasource.getTrial().getFunctionMap().get(parent);
                	if (f != null) {
                		ue.setParent(f);
                	}
                }
               	*/
                atomicEvents.put(id,ue);
            }
            resultSet.close();
        } catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }

        return atomicEvents;
	}
	
	public static int saveAtomicEvent(DB db, int newTrialID, UserEvent userEvent) {
		// for the new schema
		if(db.getSchemaVersion()>0) return saveCounter(db, newTrialID, userEvent);
    	
        int newAtomicEventID = 0;
        try {
            PreparedStatement statement = null;
            statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                    + "atomic_event (trial, name) VALUES (?, ?)");
            statement.setInt(1, newTrialID);
            statement.setString(2, userEvent.getName());
            statement.executeUpdate();
            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0)
                tmpStr = "select LAST_INSERT_ID();";
            else if (db.getDBType().compareTo("derby") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM atomic_event";
            else if (db.getDBType().compareTo("sqlite") == 0)
                tmpStr = "select seq from sqlite_sequence where name = 'atomic_event'";
            else if (db.getDBType().compareTo("h2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM atomic_event";
            else if (db.getDBType().compareTo("db2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM atomic_event";
            else if (db.getDBType().compareTo("oracle") == 0)
                tmpStr = "select " + db.getSchemaPrefix() + "atomic_event_id_seq.currval FROM dual";
            else
                tmpStr = "select currval('atomic_event_id_seq');";
            newAtomicEventID = Integer.parseInt(db.getDataItem(tmpStr));
            //userEvent.setID(newAtomicEventID);
        } catch (SQLException e) {
            System.out.println("An error occurred while saving the trial.");
            e.printStackTrace();
        }
        return newAtomicEventID;
    }

	public static int saveCounter(DB db, int newTrialID, UserEvent userEvent) {
    	
        int newCounterID = 0;
        try {
            PreparedStatement statement = null;
            statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                    + "counter (trial, name, parent) VALUES (?, ?, ?)");
            statement.setInt(1, newTrialID);
            statement.setString(2, userEvent.getName());
            System.out.println("TODO! NEED TO SAVE COUNTER CONTEXT");
//            if (parentTimer == null) {
                statement.setNull(3, java.sql.Types.INTEGER);
//            } else {
//                statement.setInt(3, parentTimer.getID());	
//            }
            statement.executeUpdate();
            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0)
                tmpStr = "select LAST_INSERT_ID();";
            else if (db.getDBType().compareTo("derby") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM counter";
            else if (db.getDBType().compareTo("sqlite") == 0)
                tmpStr = "select seq from sqlite_sequence where name = 'counter'";
            else if (db.getDBType().compareTo("h2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM counter";
            else if (db.getDBType().compareTo("db2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM counter";
            else if (db.getDBType().compareTo("oracle") == 0)
                tmpStr = "select " + db.getSchemaPrefix() + "counter_id_seq.currval FROM dual";
            else
                tmpStr = "select currval('counter_id_seq');";
            newCounterID = Integer.parseInt(db.getDataItem(tmpStr));
            //userEvent.setID(newCounterID);
        } catch (SQLException e) {
            System.out.println("An error occurred while saving the trial.");
            e.printStackTrace();
        }
        return newCounterID;
    }

    public static void getMetaData(DB db) {
        // see if we've already have them
        // need to load each time in case we are working with a new database. 
        //        if (Trial.fieldNames != null)
        //            return;

        try {
            ResultSet resultSet = null;

            String tableName = "atomic_event";
    		if(db.getSchemaVersion()>0) {
    			tableName = "counter";
    		}
    		
            DatabaseMetaData dbMeta = db.getMetaData();

            if ((db.getDBType().compareTo("oracle") == 0) || (db.getDBType().compareTo("derby") == 0)
                    || (db.getDBType().compareTo("db2") == 0)) {
                resultSet = dbMeta.getColumns(null, null, tableName.toUpperCase(), "%");
            } else {
                resultSet = dbMeta.getColumns(null, null, tableName.toLowerCase(), "%");
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

            db.getDatabase().setAtomicEventFieldNames(fieldNames);
            db.getDatabase().setAtomicEventFieldTypeNames(fieldTypeNames);
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
