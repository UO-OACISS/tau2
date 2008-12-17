package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.sql.*;
import java.util.Collections;
import java.util.Vector;

import edu.uoregon.tau.common.AlphanumComparator;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.database.DBConnector;

/**
 * Holds all the data for an experiment in the database.  
 * This object is returned by the DataSession class and all of its subtypes.
 * The Experiment object contains all the information associated with
 * an experiment from which the TAU performance data has been generated.
 * An experiment is associated with an application, and has one or more
 * trials associated with it.
 *
 * <P>CVS $Id: Experiment.java,v 1.9 2008/12/17 19:19:55 amorris Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DatabaseAPI#getExperimentList
 * @see		DatabaseAPI#setExperiment
 * @see		Application
 * @see		Trial
 */
public class Experiment implements Serializable, Comparable {

    private int experimentID;
    private int applicationID;
    private String name;
    private String fields[];
    private Database database;

    private static AlphanumComparator alphanum = new AlphanumComparator();

    public Experiment() {
        this.fields = new String[0];
    }

    // copy constructor
    public Experiment(Experiment exp) {
        this.name = exp.getName();
        this.applicationID = exp.getApplicationID();
        this.experimentID = exp.getID();
        this.fields = (String[]) exp.fields.clone();
        this.database = exp.database;
    }

    public String[] getFields() {
        return fields;
    }

    public void setFields(String[] fields) {
        this.fields = fields;
    }

    public static void getMetaData(DB db) {
        Database database = db.getDatabase();

        //see if we've already have them
        if (database.getExpFieldNames() != null)
            return;

        try {
            ResultSet resultSet = null;

            String expFieldNames[] = null;
            int expFieldTypes[] = null;

            DatabaseMetaData dbMeta = db.getMetaData();

            if ((db.getDBType().compareTo("oracle") == 0) || (db.getDBType().compareTo("derby") == 0)
                    || (db.getDBType().compareTo("db2") == 0)) {
                resultSet = dbMeta.getColumns(null, null, "EXPERIMENT", "%");
            } else {
                resultSet = dbMeta.getColumns(null, null, "experiment", "%");
            }

            Vector nameList = new Vector();
            Vector typeList = new Vector();
            boolean seenID = false;

            while (resultSet.next() != false) {

                int ctype = resultSet.getInt("DATA_TYPE");
                String cname = resultSet.getString("COLUMN_NAME");
                String typename = resultSet.getString("TYPE_NAME");
                //System.out.println ("column: " + cname + ", type: " + ctype + ", typename: " + typename);

                // only integer and string types (for now)
                // don't do name and id, we already know about them

                // this code is because of a bug in derby...
                if (cname.equals("ID")) {
                    if (!seenID)
                        seenID = true;
                    else
                        break;
                }

                if (DBConnector.isReadAbleType(ctype) && cname.toUpperCase().compareTo("ID") != 0
                        && cname.toUpperCase().compareTo("NAME") != 0 && cname.toUpperCase().compareTo("APPLICATION") != 0) {

                    nameList.add(resultSet.getString("COLUMN_NAME"));
                    typeList.add(new Integer(ctype));
                }

            }
            resultSet.close();

            String[] fieldNames = new String[nameList.size()];
            int[] fieldTypes = new int[typeList.size()];

            for (int i = 0; i < typeList.size(); i++) {
                fieldNames[i] = (String) nameList.get(i);
                fieldTypes[i] = ((Integer) typeList.get(i)).intValue();
            }

            database.setExpFieldNames(fieldNames);
            database.setExpFieldTypes(fieldTypes);

        } catch (SQLException e) {
            e.printStackTrace();
        }
        //        } catch (SQLException e) {
        //            throw new DatabaseException("Error retrieving Experiment metadata", e);
        //        }
    }

    public int getNumFields() {
        return fields.length;
    }

    public String getFieldName(int idx) {
        return database.getExpFieldNames()[idx];
    }

    public String getField(int idx) {
        return fields[idx];
    }

    public int getFieldType(int idx) {
        return database.getExpFieldTypes()[idx];
    }

    public void setField(int idx, String field) {

        if (DBConnector.isIntegerType(database.getExpFieldTypes()[idx]) && field != null) {
            try {
                int test = Integer.parseInt(field);
            } catch (java.lang.NumberFormatException e) {
                return;
            }
        }

        if (DBConnector.isFloatingPointType(database.getExpFieldTypes()[idx]) && field != null) {
            try {
                double test = Double.parseDouble(field);
            } catch (java.lang.NumberFormatException e) {
                return;
            }
        }

        fields[idx] = field;
    }

    /**
     * Gets the unique identifier of the current experiment object.
     *
     * @return	the unique identifier of the experiment
     */
    public int getID() {
        return experimentID;
    }

    /**
     * Gets the unique identifier for the application associated with this experiment.
     *
     * @return	the unique identifier of the application
     */
    public int getApplicationID() {
        return applicationID;
    }

    /**
     * Gets the name of the current experiment object.
     *
     * @return	the name of the experiment
     */
    public String getName() {
        return name;
    }

    public String toString() {
        return name;
    }

    /**
     * Sets the unique ID associated with this experiment.
     * <i>NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	id unique ID associated with this experiment
     */
    public void setID(int id) {
        this.experimentID = id;
    }

    /**
     * Sets the application ID associated with this experiment.
     * <i>NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	applicationID application ID associated with this experiment
     */
    public void setApplicationID(int applicationID) {
        this.applicationID = applicationID;
    }

    /**
     * Sets the name of the current experiment object.
     * <i>Note: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	name the experiment name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Returns the column names for the Experiment table
     *
     * @param	db	the database connection
     * @return	String[] an array of String objects
     */
    public static String[] getFieldNames(DB db) throws DatabaseException {
        getMetaData(db);
        return db.getDatabase().getExpFieldNames();
    }

    public static Vector getExperimentList(DB db, String whereClause) throws DatabaseException {
        try {
            Experiment.getMetaData(db);
            Database database = db.getDatabase();
            // create a string to hit the database
            StringBuffer buf = new StringBuffer();
            buf.append("select id, application, name");

            for (int i = 0; i < database.getExpFieldNames().length; i++) {
                buf.append(", " + database.getExpFieldNames()[i]);
            }

            buf.append(" from ");
            buf.append(db.getSchemaPrefix());
            buf.append("experiment ");

            buf.append(whereClause);

            if (db.getDBType().compareTo("oracle") == 0) {
                buf.append(" order by dbms_lob.substr(name) asc");
            } else if (db.getDBType().compareTo("derby") == 0) {
                buf.append(" order by cast(name as varchar(256)) asc");
            } else if (db.getDBType().compareTo("db2") == 0) {
                buf.append(" order by cast(name as varchar(256)) asc");
            } else {
                buf.append(" order by name asc ");
            }

            // get the results
            Vector experiments = new Vector();

            ResultSet resultSet = db.executeQuery(buf.toString());
            while (resultSet.next() != false) {
                Experiment exp = new Experiment();
                exp.setDatabase(db.getDatabase());
                exp.setID(resultSet.getInt(1));
                exp.setApplicationID(resultSet.getInt(2));
                exp.setName(resultSet.getString(3));

                for (int i = 0; i < database.getExpFieldNames().length; i++) {
                    exp.setField(i, resultSet.getString(i + 4));
                }

                experiments.addElement(exp);
            }
            resultSet.close();

            Collections.sort(experiments);

            return experiments;

        } catch (SQLException e) {
            throw new DatabaseException("Error getting experiment list", e);
        }
    }

    public int saveExperiment(DB db) throws SQLException {
        boolean itExists = exists(db);
        int newExperimentID = 0;

        StringBuffer buf = new StringBuffer();

        if (itExists) {
            buf.append("UPDATE " + db.getSchemaPrefix() + "experiment SET application = ?, name = ?");
            for (int i = 0; i < this.getNumFields(); i++) {
                if (DBConnector.isWritableType(this.getFieldType(i)))
                    buf.append(", " + this.getFieldName(i) + " = ?");
            }
            buf.append(" WHERE id = ?");
        } else {
            buf.append("INSERT INTO " + db.getSchemaPrefix() + "experiment (application, name");
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
        statement.setInt(pos++, applicationID);
        statement.setString(pos++, name);

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
            newExperimentID = experimentID;
        } else {
            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0) {
                tmpStr = "select LAST_INSERT_ID();";
            } else if (db.getDBType().compareTo("db2") == 0) {
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM experiment";
            } else if (db.getDBType().compareTo("derby") == 0) {
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM experiment";
            } else if (db.getDBType().compareTo("oracle") == 0) {
                tmpStr = "select " + db.getSchemaPrefix() + "experiment_id_seq.currval FROM dual";
            } else {
                tmpStr = "select currval('experiment_id_seq');";
            }
            newExperimentID = Integer.parseInt(db.getDataItem(tmpStr));
        }
        return newExperimentID;
    }

    private boolean exists(DB db) throws SQLException {
        boolean retval = false;
        PreparedStatement statement = db.prepareStatement("SELECT application FROM " + db.getSchemaPrefix()
                + "experiment WHERE id = ?");
        statement.setInt(1, experimentID);
        ResultSet results = statement.executeQuery();
        while (results.next() != false) {
            retval = true;
            break;
        }
        results.close();
        return retval;
    }

    public static void deleteExperiment(DB db, int experimentID) throws DatabaseException {
        try {
            PreparedStatement statement = null;
            statement = db.prepareStatement("delete from " + db.getSchemaPrefix() + "experiment where id = ?");
            statement.setInt(1, experimentID);
            statement.execute();
            statement.close();
        } catch (SQLException e) {
            throw new DatabaseException("Error deleting experiment", e);
        }
    }

    private void readObject(ObjectInputStream aInputStream) throws ClassNotFoundException, IOException {
        // always perform the default de-serialization first
        aInputStream.defaultReadObject();
        //        if (fieldNames == null)
        //            fieldNames = (String[]) aInputStream.readObject();
        //        if (fieldTypes == null)
        //            fieldTypes = (int[]) aInputStream.readObject();
    }

    private void writeObject(ObjectOutputStream aOutputStream) throws IOException {
        // always perform the default serialization first
        aOutputStream.defaultWriteObject();
        //        aOutputStream.writeObject(fieldNames);
        //        aOutputStream.writeObject(fieldTypes);
    }

    /**
     *  hack - needed to delete meta so that it is reloaded each time a new database is created.
     */
    //    public void removeMetaData() {
    //        fieldNames = null;
    //        fieldTypes = null;
    //    }
    public Database getDatabase() {
        return database;
    }

    public void setDatabase(Database database) {
        this.database = database;
        fields = new String[database.getExpFieldNames().length];
    }

    public int compareTo(Object arg0) {
        return alphanum.compare(this.getName(), ((Experiment)arg0).getName());
    }
}
