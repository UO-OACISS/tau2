package edu.uoregon.tau.perfdmf;

import java.sql.DatabaseMetaData;
import java.sql.ResultSetMetaData;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
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
    private int atomicEventID;
    private String name;
    private String group;
    private int trialID;
    private int experimentID;
    private int applicationID;
    private AtomicLocationProfile meanSummary = null;
    private AtomicLocationProfile totalSummary = null;
    private DatabaseAPI dataSession = null;

    public AtomicEvent(DatabaseAPI dataSession) {
        this.dataSession = dataSession;
    }

    /**
     * Gets the unique identifier of this atomic event object.
     *
     * @return	the unique identifier of the atomic event
     */
    public int getID() {
        return this.atomicEventID;
    }

    /**
     * Gets the name of the atomic event object.
     *
     * @return	the name of the atomic event
     */
    public String getName() {
        return this.name;
    }

    /**
     * Gets the group of the atomic event object.
     *
     * @return	the group of the atomic event
     */
    public String getGroup() {
        return this.group;
    }

    /**
     * Gets the trial ID of the atomic event object.
     *
     * @return	the trial ID of the atomic event
     */
    public int getTrialID() {
        return this.trialID;
    }

    /**
     * Gets the experiment ID of the atomic event object.
     *
     * @return	the experiment ID of the atomic event
     */
    public int getExperimentID() {
        return this.experimentID;
    }

    /**
     * Gets the application ID of the atomic event object.
     *
     * @return	the application ID of the atomic event
     */
    public int getApplicationID() {
        return this.applicationID;
    }

    /**
     * Gets mean summary data for the AtomicEvent object.
     * The mean data is averaged across all locations, defined as any combination
     * of node/context/thread.
     *
     * @return	the AtomicLocationProfile containing the mean data for this AtomicEvent.
     * @see		Trial
     * @see		AtomicLocationProfile
     * @see		DataSession#getAtomicEvents
     */
    public AtomicLocationProfile getMeanSummary() {
        if (this.meanSummary == null)
            dataSession.getAtomicEventDetail(this);
        return (this.meanSummary);
    }

    /**
     * Gets total summary data for the AtomicEvent object.
     * The total data is summed across all locations, defined as any combination
     * of node/context/thread.
     *
     * @return	the AtomicLocationProfile containing the total data for this AtomicEvent.
     * @see		Trial
     * @see		AtomicLocationProfile
     * @see		DataSession#getAtomicEvents
     */
    public AtomicLocationProfile getTotalSummary() {
        if (this.totalSummary == null)
            dataSession.getAtomicEventDetail(this);
        return (this.totalSummary);
    }

    /**
     * Sets the unique ID associated with this atomic event.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	id unique ID associated with this atomic event
     */
    public void setID(int id) {
        this.atomicEventID = id;
    }

    /**
     * Sets the atomic event name.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	name the name of the atomic event
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Sets the TAU group of this atomic event object.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	group the TAU group the atomic event is in.
     */
    public void setGroup(String group) {
        this.group = group;
    }

    /**
     * Sets the trial ID of this atomic event object.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	id the trial ID for the atomic event.
     */
    public void setTrialID(int id) {
        this.trialID = id;
    }

    /**
     * Sets the experiment ID of this atomic event object.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	id the experiment ID for the atomic event.
     */
    public void setExperimentID(int id) {
        this.experimentID = id;
    }

    /**
     * Sets the application ID of this atomic event object.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	id the application ID for the atomic event.
     */
    public void setApplicationID(int id) {
        this.applicationID = id;
    }

    /**
     * Adds a AtomicLocationProfile to the AtomicEvent as a mean summary.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	meanSummary the mean summary object for the AtomicEvent.
     */
    public void setMeanSummary(AtomicLocationProfile meanSummary) {
        this.meanSummary = meanSummary;
    }

    /**
     * Adds a AtomicLocationProfile to the AtomicEvent as a total summary.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	totalSummary the total summary object for the AtomicEvent.
     */
    public void setTotalSummary(AtomicLocationProfile totalSummary) {
        this.totalSummary = totalSummary;
    }

    // returns a Vector of AtomicEvents
    public static Vector<AtomicEvent> getAtomicEvents(DatabaseAPI dataSession, DB db, String whereClause) {
        Vector<AtomicEvent> atomicEvents = new Vector<AtomicEvent>();
        // create a string to hit the database
        StringBuffer buf = new StringBuffer();
        buf.append("select u.id, u.trial, u.name, ");
        buf.append("u.group_name ");
        buf.append("from " + db.getSchemaPrefix() + "atomic_event u inner join "
                + db.getSchemaPrefix() + "trial t on u.trial = t.id ");
        buf.append("inner join " + db.getSchemaPrefix() + "experiment e on t.experiment = e.id ");
        buf.append(whereClause);
        buf.append(" order by id ");
        // System.out.println(buf.toString());

        // get the results
        try {
            ResultSet resultSet = db.executeQuery(buf.toString());
            //AtomicEvent tmpAtomicEvent = null;
            while (resultSet.next() != false) {
                AtomicEvent ue = new AtomicEvent(dataSession);
                ue.setID(resultSet.getInt(1));
                ue.setTrialID(resultSet.getInt(2));
                ue.setName(resultSet.getString(3));
                ue.setGroup(resultSet.getString(4));
                atomicEvents.addElement(ue);
            }
            resultSet.close();
        } catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }

        return atomicEvents;
    }

    public int saveAtomicEvent(DB db, int newTrialID) {
        int newAtomicEventID = 0;
        try {
            PreparedStatement statement = null;
            statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                    + "atomic_event (trial, name, group_name) VALUES (?, ?, ?)");
            statement.setInt(1, newTrialID);
            statement.setString(2, name);
            statement.setString(3, group);
            statement.executeUpdate();
            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0)
                tmpStr = "select LAST_INSERT_ID();";
            else if (db.getDBType().compareTo("derby") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM atomic_event";
            else if (db.getDBType().compareTo("db2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM atomic_event";
            else if (db.getDBType().compareTo("oracle") == 0)
                tmpStr = "select " + db.getSchemaPrefix() + "atomic_event_id_seq.currval FROM dual";
            else
                tmpStr = "select currval('atomic_event_id_seq');";
            newAtomicEventID = Integer.parseInt(db.getDataItem(tmpStr));
        } catch (SQLException e) {
            System.out.println("An error occurred while saving the trial.");
            e.printStackTrace();
        }
        return newAtomicEventID;
    }

    public static void getMetaData(DB db) {
        // see if we've already have them
        // need to load each time in case we are working with a new database. 
        //        if (Trial.fieldNames != null)
        //            return;

        try {
            ResultSet resultSet = null;

            DatabaseMetaData dbMeta = db.getMetaData();

            if ((db.getDBType().compareTo("oracle") == 0) || (db.getDBType().compareTo("derby") == 0)
                    || (db.getDBType().compareTo("db2") == 0)) {
                resultSet = dbMeta.getColumns(null, null, "ATOMIC_EVENT", "%");
            } else {
                resultSet = dbMeta.getColumns(null, null, "atomic_event", "%");
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
