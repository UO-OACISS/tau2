package dms.dss;

/**
 * Holds all the data for an experiment in the database.  
 * This object is returned by the DataSession class and all of its subtypes.
 * The Experiment object contains all the information associated with
 * an experiment from which the TAU performance data has been generated.
 * An experiment is associated with an application, and has one or more
 * trials associated with it.
 *
 * <P>CVS $Id: Experiment.java,v 1.7 2003/10/13 17:45:35 bertie Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getExperimentList
 * @see		DataSession#setExperiment
 * @see		Application
 * @see		Trial
 */
public class Experiment {
	private int experimentID;
	private int applicationID;
        private String name;
	private String systemInfo;
	private String configurationInfo;
	private String instrumentationInfo;
	private String compilerInfo;
	private String trialTableName;

/**
 * Gets the unique identifier of the current experiment object.
 *
 * @return	the unique identifier of the experiment
 */
	public int getID () {
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
	
/**
 * Gets the System Info associated with this experiment.
 *
 * @return	the System Info for the experiment
 */
	public String getSystemInfo() {
		return systemInfo;
	}
	
/**
 * Gets the Configuration Info associated with this experiment.
 *
 * @return	the Configuration Info for the experiment
 */
	public String getConfigInfo() {
		return configurationInfo;
	}
	
/**
 * Gets the Instrumentation Info associated with this experiment.
 *
 * @return	the Instrumentation Info for the experiment
 */
	public String getInstrumentationInfo() {
		return instrumentationInfo;
	}
	
/**
 * Gets the Compiler Info associated with this experiment.
 *
 * @return	the Compiler Info for the experiment
 */
	public String getCompilerInfo() {
		return compilerInfo;
	}
	
/*
	public String getTrialTableName() {
		return trialTableName;
	}

	public void setTrialTableName (String trialTableName) {
		this.trialTableName = trialTableName;
	}
 */
	
/**
 * Sets the unique ID associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	id unique ID associated with this experiment
 */
	public void setID (int id) {
		this.experimentID = id;
	}

/**
 * Sets the application ID associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	applicationID application ID associated with this experiment
 */
	public void setApplicationID (int applicationID) {
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
 * Sets the System Info associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	systemInfo System Info associated with this experiment
 */
	public void setSystemInfo (String systemInfo) {
		this.systemInfo = systemInfo;
	}

/**
 * Sets the Configuration Info associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	configurationInfo Configuration Info associated with this experiment
 */
	public void setConfigurationInfo (String configurationInfo) {
		this.configurationInfo = configurationInfo;
	}

/**
 * Sets the Instrumentation Info associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	instrumentationInfo Instrumentation Info associated with this experiment
 */
	public void setInstrumentationInfo (String instrumentationInfo) {
		this.instrumentationInfo = instrumentationInfo;
	}

/**
 * Sets the Compiler Info associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	compilerInfo Compiler Info associated with this experiment
 */
	public void setCompilerInfo (String compilerInfo) {
		this.compilerInfo = compilerInfo;
	}

}


