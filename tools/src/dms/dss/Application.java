package dms.dss;

/**
 * Holds all the data for an application in the database.  This 
 * object is returned by the DataSession object and all of its subtypes.
 *
 * <P>CVS $Id: Application.java,v 1.6 2003/08/01 21:38:21 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version 0.1
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

/**
 * Gets the experiment table name of the current application object.
 *
 * @return	the experiment table name of the application
 */
	public String getExperimentTableName() {
		return experimentTableName;
	}

/**
 * Sets the unique identifier of the current application object.
 *
 * @param	id a unique application identifier
 */
	public void setID(int id) {
		this.applicationID = id;
	}

/**
 * Sets the name of the current application object.
 *
 * @param	id the application name
 */
	public void setName(String name) {
		this.name = name;
	}

/**
 * Sets the version of the current application object.
 *
 * @param	id the application version
 */
	public void setVersion(String version) {
		this.version = version;
	}

/**
 * Sets the description of the current application object.
 *
 * @param	id the application description
 */
	public void setDescription(String description) {
		this.description = description;
	}

/**
 * Sets the language of the current application object.
 *
 * @param	id the application language
 */
	public void setLanguage(String language) {
		this.language = language;
	}

/**
 * Sets the para diag of the current application object.
 *
 * @param	id the application para diag
 */
	public void setParaDiag(String paraDiag) {
		this.paraDiag = paraDiag;
	}

/**
 * Sets the usage of the current application object.
 *
 * @param	id the application usage
 */
	public void setUsage(String usage) {
		this.usage = usage;
	}

/**
 * Sets the executable options of the current application object.
 *
 * @param	id the application executable options
 */
	public void setExecutableOptions(String executableOptions) {
		this.executableOptions = executableOptions;
	}

/**
 * Sets the experiment table name of the current application object.
 *
 * @param	id the application experiment table name
 */
	public void setExperimentTableName(String experimentTableName) {
		this.experimentTableName = experimentTableName;
	}
}

