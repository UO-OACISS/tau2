package dms.dss;

/**
 * Holds all the data for an application in the database.  This 
 * object is returned by the DataSession object and all of its subtypes.
 * The Application object contains all the information associated with
 * an application from which the TAU performance data has been generated.
 * An application has one or more experiments associated with it.
 *
 * <P>CVS $Id: Application.java,v 1.8 2003/08/27 17:07:37 khuck Exp $</P>
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
	private String paraDiag;
	private String usage;
	private String executableOptions;
	private String experimentTableName;

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
		return paraDiag;
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
 * @param	paraDiag the application para diag
 */
	public void setParaDiag(String paraDiag) {
		this.paraDiag = paraDiag;
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
}

