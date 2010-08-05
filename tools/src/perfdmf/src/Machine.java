package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.database.DBConnector;

/**
 * Holds all the data for a machine/thread_map in the database.  
 * This object is returned by the DataSession class and all of its subtypes.
 * The Machine object contains all the information associated with
 * a node of a parallel machine from which the TAU performance data has been generated.
 * A machine is associated with a trial, and has one or more
 * interval_location_profiles associated with it.
 *
 * <P>CVS $Id: Machine.java,v 1.2 2007/05/02 19:43:28 amorris Exp $</P>
 * @author	Kevin Huck
 * @version	0.2
 * @since	0.2
 * @see		Trial
 * @see     IntervalLocationProfile
 */
public class Machine implements Serializable {

    private static String fieldNames[];
    private static int fieldTypes[];

    private int machineID;
    private int trialID;
    private int nodeID;
    private int contextID;
    private int threadID;
    private String fields[];
    
/**
 * Default Constructor
 *
 * @return new Machine object
 */
    public Machine() {
        if (Machine.fieldNames == null) {
            this.fields = new String[0];
        } else {
            this.fields = new String[Machine.fieldNames.length];
        }
    }

/**
 * Copy Constructor
 *
 * @return new Machine object
 */
    public Machine(Machine machine) {
        this.machineID = machine.getID();
        this.nodeID = machine.getNodeID();
        this.contextID = machine.getContextID();
        this.threadID = machine.getThreadID();
        this.fields = (String[]) machine.fields.clone();
    }

/**
 * Reallocate space for storage
 * 
 * Reallocate the storage space for the attributes
 *
 */
    public void reallocMetaData() {
        if (Machine.fieldNames == null) {
            this.fields = new String[0];
        } else {
            this.fields = new String[Machine.fieldNames.length];
        }
    }

/**
 * Get the array of attributes
 * 
 * @return the array of attributes
 */
    public String[] getFields() {
        return fields;
    }

/**
 * Set the array of fields
 * 
 * @param fields the array of strings which are the table attributes
 */
    public void setFields(String[] fields) {
        this.fields = fields;
    }

/**
 * Get the table information from the database
 * 
 * Other than the id, trial, node, context and thread columns, this code
 * makes no other assumptions about what columns might be in the machine_thread_map
 * table.  Therefore, we have to use the JDBC connection to query the DBMS
 * and ask what columns are in the machine_thread_map table.
 * 
 * @param db the DB connection object
 */
    public static void getMetaData(DB db) {
        // see if we've already have them
        if (Machine.fieldNames != null)
            return;

        try {
            ResultSet resultSet = null;

            DatabaseMetaData dbMeta = db.getMetaData();

            if ((db.getDBType().compareTo("oracle") == 0) || 
				(db.getDBType().compareTo("derby") == 0) || 
				(db.getDBType().compareTo("db2") == 0)) {
                resultSet = dbMeta.getColumns(null, null, "MACHINE_THREAD_MAP", "%");
            } else {
                resultSet = dbMeta.getColumns(null, null, "machine_thread_map", "%");
            }

            List<String> nameList = new ArrayList<String>();
            List<Integer> typeList = new ArrayList<Integer>();
			boolean seenID = false;

            while (resultSet.next() != false) {

                int ctype = resultSet.getInt("DATA_TYPE");
                String cname = resultSet.getString("COLUMN_NAME");
                String typename = resultSet.getString("TYPE_NAME");
                //System.out.println ("column: " + cname + ", type: " + ctype + ", typename: " + typename);

                // only integer and string types (for now)
                // don't do id, trial, node, context or thread - we already know about them

				// this code is because of a bug in derby...
				if (cname.equals("ID")) {
					if (!seenID)
						seenID = true;
					else
						break;
				}

                if (DBConnector.isReadAbleType(ctype) && cname.toUpperCase().compareTo("ID") != 0
                        && cname.toUpperCase().compareTo("TRIAL") != 0
                        && cname.toUpperCase().compareTo("NODE") != 0
                        && cname.toUpperCase().compareTo("CONTEXT") != 0
                        && cname.toUpperCase().compareTo("THREAD") != 0) {

                    nameList.add(resultSet.getString("COLUMN_NAME"));
                    typeList.add(new Integer(ctype));
                }

            }
            resultSet.close();

            Machine.fieldNames = new String[nameList.size()];
            Machine.fieldTypes = new int[typeList.size()];

            for (int i = 0; i < typeList.size(); i++) {
                Machine.fieldNames[i] = nameList.get(i);
                Machine.fieldTypes[i] = typeList.get(i).intValue();
            }

        } catch (SQLException e) {
            System.err.println(e.getMessage());
            e.printStackTrace();
        }
    }

    public int getNumFields() {
        return fields.length;
    }

    public String getFieldName(int idx) {
        return Machine.fieldNames[idx];
    }

    public String getField(int idx) {
        return fields[idx];
    }

    public int getFieldType(int idx) {
        return Machine.fieldTypes[idx];
    }

    public void setField(int idx, String field) {

        if (DBConnector.isIntegerType(fieldTypes[idx]) && field != null) {
            try {
                int test = Integer.parseInt(field);
            } catch (java.lang.NumberFormatException e) {
                return;
            }
        }

        if (DBConnector.isFloatingPointType(fieldTypes[idx]) && field != null) {
            try {
                double test = Double.parseDouble(field);
            } catch (java.lang.NumberFormatException e) {
                return;
            }
        }

        fields[idx] = field;
    }

    /**
     * Gets the unique identifier for the trial associated with this Machine.
     *
     * @return	the unique identifier of the trial
     */
    public int getID() {
        return machineID;
    }

    /**
     * Gets the unique identifier for the trial associated with this Machine.
     *
     * @return	the unique identifier of the trial
     */
    public int getTrialID() {
        return trialID;
    }

    /**
     * Gets the identifier for the node associated with this Machine.
     *
     * @return	the identifier of the node
     */
    public int getNodeID() {
        return nodeID;
    }

    /**
     * Gets the identifier for the context associated with this Machine.
     *
     * @return	the identifier of the context
     */
    public int getContextID() {
        return contextID;
    }

    /**
     * Gets the identifier for the thread associated with this Machine.
     *
     * @return	the identifier of the thread
     */
    public int getThreadID() {
        return threadID;
    }

    public String toString() {
        return "machine: " + machineID;
    }

    /**
     * Sets the unique ID associated with this Machine.
     * <i>NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	id unique ID associated with this Machine
     */
    public void setID(int id) {
        this.machineID = id;
    }

    /**
     * Sets the trial ID associated with this Machine.
     * <i>NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	trialID trial ID associated with this Machine
     */
    public void setTrialID(int trialID) {
        this.trialID = trialID;
    }

    /**
     * Sets the node ID associated with this Machine.
     * <i>NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	nodeID node ID associated with this Machine
     */
    public void setNodeID(int nodeID) {
        this.nodeID = nodeID;
    }

    /**
     * Sets the context ID associated with this Machine.
     * <i>NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	contextID context ID associated with this Machine
     */
    public void setContextID(int contextID) {
        this.contextID = contextID;
    }

    /**
     * Sets the thread ID associated with this Machine.
     * <i>NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	threadID thread ID associated with this Machine
     */
    public void setThreadID(int threadID) {
        this.threadID = threadID;
    }

    /**
     * Returns the column names for the Machine table
     *
     * @param	db	the database connection
     * @return	String[] an array of String objects
     */
    public static String[] getFieldNames(DB db) throws DatabaseException {
        getMetaData(db);
        return fieldNames;
    }

    public static List<Machine> getMachineList(DB db, String whereClause) throws DatabaseException {
        try {
            Machine.getMetaData(db);

            // create a string to hit the database
            StringBuffer buf = new StringBuffer();
            buf.append("select id, trial, node, context, thread");

            for (int i = 0; i < Machine.fieldNames.length; i++) {
                buf.append(", " + Machine.fieldNames[i]);
            }

            buf.append(" from ");
            buf.append(db.getSchemaPrefix());
            buf.append("machine_thread_map ");

            buf.append(whereClause);

            buf.append(" order by trial, node, context, thread asc ");

            // get the results
            List<Machine> Machines = new ArrayList<Machine>();

            ResultSet resultSet = db.executeQuery(buf.toString());
            while (resultSet.next() != false) {
                Machine machine = new Machine();
                machine.setID(resultSet.getInt(1));
                machine.setTrialID(resultSet.getInt(2));
                machine.setNodeID(resultSet.getInt(3));
                machine.setContextID(resultSet.getInt(4));
                machine.setThreadID(resultSet.getInt(5));

                for (int i = 0; i < Machine.fieldNames.length; i++) {
                    machine.setField(i, resultSet.getString(i + 4));
                }

                Machines.add(machine);
            }
            resultSet.close();

            return Machines;

        } catch (SQLException e) {
            throw new DatabaseException("Error getting Machine list", e);
        }
    }

    public int saveMachine(DB db) throws SQLException {
        boolean itExists = exists(db);
        int newMachineID = 0;

        StringBuffer buf = new StringBuffer();

        if (itExists) {
            buf.append("UPDATE " + db.getSchemaPrefix() + "Machine SET trial = ?, node = ?, context = ?, thread = ?");
            for (int i = 0; i < this.getNumFields(); i++) {
                if (DBConnector.isWritableType(this.getFieldType(i)))
                    buf.append(", " + this.getFieldName(i) + " = ?");
            }
            buf.append(" WHERE id = ?");
        } else {
            buf.append("INSERT INTO " + db.getSchemaPrefix() + "Machine (trial, node, context, thread");
            for (int i = 0; i < this.getNumFields(); i++) {
                if (DBConnector.isWritableType(this.getFieldType(i)))
                    buf.append(", " + this.getFieldName(i));
            }
            buf.append(") VALUES (?, ?");
            for (int i = 0; i < this.getNumFields(); i++) {
                if (DBConnector.isWritableType(this.getFieldType(i)))
                    buf.append(", ?");
            }
            buf.append(")");
        }

        PreparedStatement statement = db.prepareStatement(buf.toString());

        int pos = 1;
        statement.setInt(pos++, trialID);
        statement.setInt(pos++, nodeID);
        statement.setInt(pos++, contextID);
        statement.setInt(pos++, threadID);

        for (int i = 0; i < this.getNumFields(); i++) {
            if (DBConnector.isWritableType(this.getFieldType(i)))
                statement.setString(pos++, this.getField(i));
        }

        if (itExists) {
            statement.setInt(pos++, this.getID());
        }
        statement.executeUpdate();
        statement.close();
        if (itExists) {
            newMachineID = machineID;
        } else {
            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0) {
                tmpStr = "select LAST_INSERT_ID();";
            } else if (db.getDBType().compareTo("db2") == 0) {
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM Machine";
            } else if (db.getDBType().compareTo("derby") == 0) {
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM Machine";
            } else if (db.getDBType().compareTo("oracle") == 0) {
                tmpStr = "select " + db.getSchemaPrefix() + "Machine_id_seq.currval FROM dual";
            } else {
                tmpStr = "select currval('Machine_id_seq');";
            }
            newMachineID = Integer.parseInt(db.getDataItem(tmpStr));
        }
        return newMachineID;
    }

    private boolean exists(DB db) throws SQLException {
        boolean retval = false;
        PreparedStatement statement = db.prepareStatement("SELECT trial FROM " + db.getSchemaPrefix()
                + "Machine WHERE id = ?");
        statement.setInt(1, machineID);
        ResultSet results = statement.executeQuery();
        while (results.next() != false) {
            retval = true;
            break;
        }
        results.close();
        return retval;
    }

    public static void deleteMachine(DB db, int MachineID) throws DatabaseException {
        try {
            PreparedStatement statement = null;
            statement = db.prepareStatement("delete from " + db.getSchemaPrefix() + "Machine where id = ?");
            statement.setInt(1, MachineID);
            statement.execute();
            statement.close();
        } catch (SQLException e) {
            throw new DatabaseException("Error deleting Machine", e);
        }
    }

    private void readObject(ObjectInputStream aInputStream) throws ClassNotFoundException, IOException {
        // always perform the default de-serialization first
        aInputStream.defaultReadObject();
        if (fieldNames == null)
            fieldNames = (String[]) aInputStream.readObject();
        if (fieldTypes == null)
            fieldTypes = (int[]) aInputStream.readObject();
    }

    private void writeObject(ObjectOutputStream aOutputStream) throws IOException {
        // always perform the default serialization first
        aOutputStream.defaultWriteObject();
        aOutputStream.writeObject(fieldNames);
        aOutputStream.writeObject(fieldTypes);
    }
    /**
     *  hack - needed to delete meta so that it is reloaded each time a new database is created.
     */
    public void removeMetaData()
    {
    	fieldNames = null;
    	fieldTypes = null;
    }
}
