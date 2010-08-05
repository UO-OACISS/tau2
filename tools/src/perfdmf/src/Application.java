package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.sql.*;
import java.util.Vector;

import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.database.DBConnector;

/**
 * Holds all the data for an application in the database.  This 
 * object is returned by the DatabaseAPI object.
 * The Application object contains all the information associated with
 * an application from which the TAU performance data has been generated.
 * An application has zero or more experiments associated with it.
 *
 * <P>CVS $Id: Application.java,v 1.15 2009/08/19 11:14:50 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version 0.1
 * @since   0.1
 * @see		DatabaseAPI#getApplicationList
 * @see		DatabaseAPI#setApplication
 * @see		Experiment
 */
public class Application implements Serializable {

    private int applicationID;
    private String name;
    private String fields[];

    private Database database;

    public Database getDatabase() {
        return database;
    }

    public void setDatabase(Database database) {
        this.database = database;
        fields = new String[database.getAppFieldNames().length];
    }

    public Application() {
        fields = new String[0];
    }

    // copy constructor
    public Application(Application app) {
        this.name = app.getName();
        this.applicationID = app.getID();
        this.fields = (String[]) app.fields.clone();
        this.database = app.database;
    }

    public String[] getFields() {
        return fields;
    }

    public void setFields(String[] fields) {
        this.fields = fields;
    }

    /**
     * Returns the column names for the Application table
     *
     * @param	db	the database connection
     * @return	String[] an array of String objects
     */
    public static String[] getFieldNames(DB db) throws DatabaseException {
        getMetaData(db);
        return db.getDatabase().getAppFieldNames();
    }

    public static void getMetaData(DB db) {
        // see if we've already have them
        //       if (Application.fieldNames != null)
        //            return;

        try {
            Database database = db.getDatabase();
            ResultSet resultSet = null;

            String appFieldNames[] = null;
            int appFieldTypes[] = null;

            DatabaseMetaData dbMeta = db.getMetaData();

            if ((db.getDBType().compareTo("oracle") == 0) || (db.getDBType().compareTo("derby") == 0)
                    || (db.getDBType().compareTo("db2") == 0)) {
                resultSet = dbMeta.getColumns(null, null, "APPLICATION", "%");
            } else {
                resultSet = dbMeta.getColumns(null, null, "application", "%");
            }

            Vector<String> nameList = new Vector<String>();
            Vector<Integer> typeList = new Vector<Integer>();
            boolean seenID = false;

            while (resultSet.next() != false) {

                int ctype = resultSet.getInt("DATA_TYPE");
                String cname = resultSet.getString("COLUMN_NAME");
                String typename = resultSet.getString("TYPE_NAME");

                // this code is because of a bug in derby...
                if (cname.equals("ID")) {
                    if (!seenID)
                        seenID = true;
                    else
                        break;
                }

                // only integer and string types (for now)
                // don't do name and id, we already know about them
                if (DBConnector.isReadAbleType(ctype) && cname.toUpperCase().compareTo("ID") != 0
                        && cname.toUpperCase().compareTo("NAME") != 0) {

                    nameList.add(resultSet.getString("COLUMN_NAME"));
                    typeList.add(new Integer(ctype));
                }
            }
            resultSet.close();

            String[] fieldNames = new String[nameList.size()];
            int[] fieldTypes = new int[typeList.size()];

            for (int i = 0; i < typeList.size(); i++) {
                fieldNames[i] = nameList.get(i);
                fieldTypes[i] = typeList.get(i).intValue();
            }

            database.setAppFieldNames(fieldNames);
            database.setAppFieldTypes(fieldTypes);

        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public int getNumFields() {
        return fields.length;
    }

    public String getFieldName(int idx) {
        return database.getAppFieldNames()[idx];
    }

    public int getFieldType(int idx) {
        return database.getAppFieldTypes()[idx];
    }

    // These two are here to handle the copy constructor for making ParaProfApplications
    public String[] getFieldNames() {
        return database.getAppFieldNames();
    }

    public int[] getFieldTypes() {
        return database.getAppFieldTypes();
    }

    /**
     * Gets the unique identifier of the current application object.
     *
     * @return	the unique identifier of the application
     */
    public int getID() {
        return applicationID;
    }

    /**
     * Gets the name of the current application object.
     *
     * @return	the name of the application
     */
    public String getName() {
        return name;
    }

    public String toString() {
        return name;
    }

    public String getField(int idx) {
        return fields[idx];
    }

    public String getField(String name) {
        if (database.getAppFieldNames() == null)
            return null;
        for (int i = 0; i < database.getAppFieldNames().length; i++) {
            if (name.toUpperCase().equals(database.getAppFieldNames()[i].toUpperCase())) {
                if (i < fields.length)
                    return fields[i];
            }
        }
        return null;
    }

    public void setField(int idx, String field) {
        if (DBConnector.isIntegerType(database.getAppFieldTypes()[idx]) && field != null) {
            try {
                int test = Integer.parseInt(field);
            } catch (java.lang.NumberFormatException e) {
                return;
            }
        }

        if (DBConnector.isFloatingPointType(database.getAppFieldTypes()[idx]) && field != null) {
            try {
                double test = Double.parseDouble(field);
            } catch (java.lang.NumberFormatException e) {
                return;
            }
        }
        fields[idx] = field;
    }

    /**
     * Sets the unique identifier of the current application object.
     * <i>Note: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	id a unique application identifier
     */
    public void setID(int id) {
        this.applicationID = id;
    }

    /**
     * Sets the name of the current application object.
     * <i>Note: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	name the application name
     */
    public void setName(String name) {
        this.name = name;
    }

    public static Vector<Application> getApplicationList(DB db, String whereClause) {
        StringBuffer buf = null;
        try {
            Database database = db.getDatabase();
            Application.getMetaData(db);

            ResultSet resultSet = null;
            Vector<Application> applications = new Vector<Application>();

            buf = new StringBuffer("select id, name");

            for (int i = 0; i < database.getAppFieldNames().length; i++) {
                buf.append(", " + database.getAppFieldNames()[i]);
            }

            buf.append(" from " + db.getSchemaPrefix() + "application");

            buf.append(whereClause);

            if (db.getDBType().compareTo("oracle") == 0) {
                buf.append(" order by dbms_lob.substr(name), id asc");
            } else if (db.getDBType().compareTo("derby") == 0) {
                buf.append(" order by cast (name as varchar(256)), id asc");
            } else if (db.getDBType().compareTo("db2") == 0) {
                buf.append(" order by cast (name as varchar(256)), id asc");
            } else {
                buf.append(" order by name, id asc ");
            }

            resultSet = db.executeQuery(buf.toString());
            while (resultSet.next() != false) {
                Application application = new Application();
                application.setDatabase(database);

                application.setID(resultSet.getInt(1));
                application.setName(resultSet.getString(2));

                String tmp = resultSet.getString(3);

                for (int i = 0; i < database.getAppFieldNames().length; i++) {
                    application.setField(i, resultSet.getString(i + 3));
                }

                //Add the application.
                applications.addElement(application);
            }
            //Cleanup resources.
            resultSet.close();

            return applications;
        } catch (SQLException e) {
            if (buf != null)
                System.out.println(buf.toString());
			System.err.println(e.getMessage());
            throw new DatabaseException("", e);
        }

    }

    public int saveApplication(DB db) throws SQLException {

        boolean itExists = false;

        // First, determine whether it exists already (whether we are doing an insert or update)
        PreparedStatement statement = db.prepareStatement("SELECT name FROM " + db.getSchemaPrefix() + "application WHERE id = ?");
        statement.setInt(1, this.getID());
        ResultSet resultSet = statement.executeQuery();
        while (resultSet.next() != false) {
            itExists = true;
            break;
        }
        resultSet.close();
        statement.close();

        StringBuffer buf = new StringBuffer();
        if (itExists) {
            buf.append("UPDATE " + db.getSchemaPrefix() + "application SET name = ?");
            for (int i = 0; i < this.getNumFields(); i++) {
                if (DBConnector.isWritableType(this.getFieldType(i)))
                    buf.append(", " + this.getFieldName(i) + " = ?");
            }
            buf.append(" WHERE id = ?");
        } else {
            buf.append("INSERT INTO " + db.getSchemaPrefix() + "application (name");
            for (int i = 0; i < this.getNumFields(); i++) {
                if (DBConnector.isWritableType(this.getFieldType(i)))
                    buf.append(", " + this.getFieldName(i));
            }
            buf.append(") VALUES (?");
            for (int i = 0; i < this.getNumFields(); i++) {
                if (DBConnector.isWritableType(this.getFieldType(i)))
                    buf.append(", ?");
            }
            buf.append(")");
        }

        statement = db.prepareStatement(buf.toString());

        int pos = 1;
        statement.setString(pos++, this.getName());

        for (int i = 0; i < this.getNumFields(); i++) {
            if (DBConnector.isWritableType(this.getFieldType(i)))
                statement.setString(pos++, this.getField(i));
        }

        if (itExists) {
            statement.setInt(pos++, this.getID());
        }
        statement.executeUpdate();
        statement.close();

        int newApplicationID = 0;

        if (itExists) {
            newApplicationID = this.getID();
        } else {
            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0) {
                tmpStr = "select LAST_INSERT_ID();";
            } else if (db.getDBType().compareTo("db2") == 0) {
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM application";
            } else if (db.getDBType().compareTo("derby") == 0) {
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM application";
            } else if (db.getDBType().compareTo("oracle") == 0) {
                tmpStr = "SELECT " + db.getSchemaPrefix() + "application_id_seq.currval FROM DUAL";
            } else { // postgresql 
                tmpStr = "select currval('application_id_seq');";
            }
            newApplicationID = Integer.parseInt(db.getDataItem(tmpStr));
        }
        return newApplicationID;

    }

    private boolean exists(DB db) {
        boolean retval = false;
        try {
            PreparedStatement statement = db.prepareStatement("SELECT name FROM application WHERE id = ?");
            statement.setInt(1, applicationID);
            ResultSet results = statement.executeQuery();
            while (results.next() != false) {
                retval = true;
                break;
            }
            results.close();
        } catch (SQLException e) {
            System.out.println("An error occurred while saving the application.");
            e.printStackTrace();
        }
        return retval;
    }

    public static void deleteApplication(DB db, int applicationID) {
        try {
            PreparedStatement statement = null;
            statement = db.prepareStatement("delete from " + db.getSchemaPrefix() + "application where id = ?");
            statement.setInt(1, applicationID);
            statement.execute();
            statement.close();
        } catch (SQLException e) {
            System.out.println("An error occurred while deleting the application.");
            e.printStackTrace();
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

    //    /**
    //     *  hack - needed to delete meta so that it is reloaded each time a new database is created.
    //     */
    //    public void removeMetaData() {
    //        fieldNames = null;
    //        fieldTypes = null;
    //    }

}
