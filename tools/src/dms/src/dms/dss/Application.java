package dms.dss;

import dms.perfdb.DB;
import java.sql.SQLException;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.Vector;

/**
 * Holds all the data for an application in the database.  This 
 * object is returned by the DataSession object and all of its subtypes.
 * The Application object contains all the information associated with
 * an application from which the TAU performance data has been generated.
 * An application has one or more experiments associated with it.
 *
 * <P>CVS $Id: Application.java,v 1.2 2004/04/07 17:36:57 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version 0.1
 * @since 0.1
 * @see		DataSession#getApplicationList
 * @see		DataSession#setApplication
 * @see		Experiment
 */
public class Application {
	private int applicationID;
	private String name;
	private String version;
	private String description;
	private String language;
	private String paradigm;
	private String usage;
	private String executableOptions;
	private String userData;

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

/**
 * Gets the version of the current application object.
 *
 * @return	the version of the application
 */
	public String getVersion() {
		return version;
	}

/**
 * Gets the description of the current application object.
 *
 * @return	the description of the application
 */
	public String getDescription() {
		return description;
	}

/**
 * Gets the language of the current application object.
 *
 * @return	the language of the application
 */
	public String getLanguage() {
		return language;
	}

/**
 * Gets the para diag of the current application object.
 *
 * @return	the para diag of the application
 */
	public String getParaDiag() {
		return paradigm;
	}

/**
 * Gets the usage of the current application object.
 *
 * @return	the usage of the application
 */
	public String getUsage() {
		return usage;
	}

/**
 * Gets the executable options of the current application object.
 *
 * @return	the executable options of the application
 */
	public String getExecutableOptions() {
		return executableOptions;
	}

/**
 * Gets the user data of the current application object.
 *
 * @return	the user data of the application
 */
	public String getUserData() {
		return userData;
	}

/*
	public String getExperimentTableName() {
		return experimentTableName;
	}

	public void setExperimentTableName(String experimentTableName) {
		this.experimentTableName = experimentTableName;
	}
 */

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

/**
 * Sets the version of the current application object.
 * <i>Note: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	version the application version
 */
	public void setVersion(String version) {
		this.version = version;
	}

/**
 * Sets the description of the current application object.
 * <i>Note: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	description the application description
 */
	public void setDescription(String description) {
		this.description = description;
	}

/**
 * Sets the language of the current application object.
 * <i>Note: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	language the application language
 */
	public void setLanguage(String language) {
		this.language = language;
	}

/**
 * Sets the para diag of the current application object.
 * <i>Note: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	paradigm the application para diag
 */
	public void setParaDiag(String paradigm) {
		this.paradigm = paradigm;
	}

/**
 * Sets the usage of the current application object.
 * <i>Note: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	usage the application usage
 */
	public void setUsage(String usage) {
		this.usage = usage;
	}

/**
 * Sets the executable options of the current application object.
 * <i>Note: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	executableOptions the application executable options
 */
	public void setExecutableOptions(String executableOptions) {
		this.executableOptions = executableOptions;
	}
 
/**
 * Sets the user data of the current application object.
 * <i>Note: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	userData the application user data
 */
	public void setUserData(String userData) {
		this.userData = userData;
	}

	public static Vector getApplicationList(DB db, String whereClause) {
		Vector applications = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select * from application ");
		buf.append(whereClause);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
	    	while (resultSet.next() != false) {
				Application app = new Application();
				app.setID(resultSet.getInt(1));
				app.setName(resultSet.getString(2));
				app.setVersion(resultSet.getString(3));
				app.setDescription(resultSet.getString(4));
				app.setLanguage(resultSet.getString(5));
				app.setParaDiag(resultSet.getString(6));
				app.setUsage(resultSet.getString(7));
				app.setExecutableOptions(resultSet.getString(8));
				app.setUserData(resultSet.getString(9));
				applications.addElement(app);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return applications;
	}

	public void saveApplication(DB db) {
		boolean itExists = exists(db);
		try {
			PreparedStatement statement = null;
			if (itExists) {
				statement = db.prepareStatement("UPDATE APPLICATION SET name = ?, version = ?, description = ?, language = ?, paradigm = ?, usage_text = ?, execution_options = ?, userdata = ? WHERE id = ?");
			} else {
				statement = db.prepareStatement("INSERT INTO APPLICATION (name, version, description, language, paradigm, usage_text, execution_options, userdata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)");
			}
			statement.setString(1, name);
			statement.setString(2, version);
			statement.setString(3, description);
			statement.setString(4, language);
			statement.setString(5, paradigm);
			statement.setString(6, usage);
			statement.setString(7, executableOptions);
			statement.setString(8, userData);
			if (itExists) {
				statement.setInt(9, applicationID);
			}
			statement.executeUpdate();
			String tmpStr = new String();
			if (db.getDBType().compareTo("mysql") == 0)
				tmpStr = "select LAST_INSERT_ID();";
			if (db.getDBType().compareTo("db2") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM application";
			else // postgresql
				tmpStr = "select currval('application_id_seq');";
			applicationID = Integer.parseInt(db.getDataItem(tmpStr));
		} catch (SQLException e) {
			System.out.println("An error occurred while saving the application.");
			e.printStackTrace();
			System.exit(0);
		}
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
			System.exit(0);
		}
		return retval;
	}
}

