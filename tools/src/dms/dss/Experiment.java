package dms.dss;

/**
 * Holds all the data for an experiment in the database.  
 * This object is returned by the DataSession class and all of its subtypes.
 * The Experiment object contains all the information associated with
 * an experiment from which the TAU performance data has been generated.
 * An experiment is associated with an application, and has one or more
 * trials associated with it.
 *
 * <P>CVS $Id: Experiment.java,v 1.8 2003/10/17 18:46:53 khuck Exp $</P>
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
	private String userData;
	private String systemName;
	private String systemMachineType;
	private String systemArch;
	private String systemOS;
	private String systemMemorySize;
	private String systemProcessorAmount;
	private String systemL1CacheSize;
	private String systemL2CacheSize;
	private String systemUserData;
	private String configurationPrefix;
	private String configurationArchitecture;
	private String configurationCpp;
	private String configurationCc;
	private String configurationJdk;
	private String configurationProfile;
	private String configurationUserData;
	private String compilerCppName;
	private String compilerCppVersion;
	private String compilerCcName;
	private String compilerCcVersion;
	private String compilerJavaDirpath;
	private String compilerJavaVersion;
	private String compilerUserData;
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
 * Gets the user data from the current experiment object.
 *
 * @return	the user data from the experiment
 */
	public String getUserData() {
		return userData;
	}
	
/**
 * Gets the System Name associated with this experiment.
 *
 * @return	the System Name for the experiment
 */
	public String getSystemName() {
		return systemName;
	}
	
/**
 * Gets the System MachineType associated with this experiment.
 *
 * @return	the System MachineType for the experiment
 */
	public String getSystemMachineType() {
		return systemMachineType;
	}
	
/**
 * Gets the System Arch associated with this experiment.
 *
 * @return	the System Arch for the experiment
 */
	public String getSystemArch() {
		return systemArch;
	}
	
/**
 * Gets the System OS associated with this experiment.
 *
 * @return	the System OS for the experiment
 */
	public String getSystemOS() {
		return systemOS;
	}
	
/**
 * Gets the System MemorySize associated with this experiment.
 *
 * @return	the System MemorySize for the experiment
 */
	public String getSystemMemorySize() {
		return systemMemorySize;
	}
	
/**
 * Gets the System ProcessorAmount associated with this experiment.
 *
 * @return	the System ProcessorAmount for the experiment
 */
	public String getSystemProcessorAmount() {
		return systemProcessorAmount;
	}
	
/**
 * Gets the System L1CacheSize associated with this experiment.
 *
 * @return	the System L1CacheSize for the experiment
 */
	public String getSystemL1CacheSize() {
		return systemL1CacheSize;
	}
	
/**
 * Gets the System L2CacheSize associated with this experiment.
 *
 * @return	the System L2CacheSize for the experiment
 */
	public String getSystemL2CacheSize() {
		return systemL2CacheSize;
	}
	
/**
 * Gets the System UserData associated with this experiment.
 *
 * @return	the System UserData for the experiment
 */
	public String getSystemUserData() {
		return systemUserData;
	}
	
/**
 * Gets the Configuration Prefix associated with this experiment.
 *
 * @return	the Configuration Prefix for the experiment
 */
	public String getConfigPrefix() {
		return configurationPrefix;
	}
	
/**
 * Gets the Configuration Architecture associated with this experiment.
 *
 * @return	the Configuration Architecture for the experiment
 */
	public String getConfigArchitecture() {
		return configurationArchitecture;
	}
	
/**
 * Gets the Configuration Cpp associated with this experiment.
 *
 * @return	the Configuration Cpp for the experiment
 */
	public String getConfigCpp() {
		return configurationCpp;
	}
	
/**
 * Gets the Configuration Cc associated with this experiment.
 *
 * @return	the Configuration Cc for the experiment
 */
	public String getConfigCc() {
		return configurationCc;
	}
	
/**
 * Gets the Configuration Jdk associated with this experiment.
 *
 * @return	the Configuration Jdk for the experiment
 */
	public String getConfigJdk() {
		return configurationJdk;
	}
	
/**
 * Gets the Configuration Profile associated with this experiment.
 *
 * @return	the Configuration Profile for the experiment
 */
	public String getConfigProfile() {
		return configurationProfile;
	}
	
/**
 * Gets the Configuration UserData associated with this experiment.
 *
 * @return	the Configuration UserData for the experiment
 */
	public String getConfigUserData() {
		return configurationUserData;
	}
	
/**
 * Gets the Compiler CppName associated with this experiment.
 *
 * @return	the Compiler CppName for the experiment
 */
	public String getCompilerCppName() {
		return compilerCppName;
	}
	
/**
 * Gets the Compiler CppVersion associated with this experiment.
 *
 * @return	the Compiler CppVersion for the experiment
 */
	public String getCompilerCppVersion() {
		return compilerCppVersion;
	}
	
/**
 * Gets the Compiler CcName associated with this experiment.
 *
 * @return	the Compiler CcName for the experiment
 */
	public String getCompilerCcName() {
		return compilerCcName;
	}
	
/**
 * Gets the Compiler CcVersion associated with this experiment.
 *
 * @return	the Compiler CcVersion for the experiment
 */
	public String getCompilerCcVersion() {
		return compilerCcVersion;
	}
	
/**
 * Gets the Compiler JavaDirpath associated with this experiment.
 *
 * @return	the Compiler JavaDirpath for the experiment
 */
	public String getCompilerJavaDirpath() {
		return compilerJavaDirpath;
	}
	
/**
 * Gets the Compiler JavaVersion associated with this experiment.
 *
 * @return	the Compiler JavaVersion for the experiment
 */
	public String getCompilerJavaVersion() {
		return compilerJavaVersion;
	}
	
/**
 * Gets the Compiler UserData associated with this experiment.
 *
 * @return	the Compiler UserData for the experiment
 */
	public String getCompilerUserData() {
		return compilerUserData;
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
 * Sets the user data of the current experiment object.
 * <i>Note: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	userData the experiment user data
 */
	public void setUserData(String userData) {
		this.userData = userData;
	}

/**
 * Sets the System Name associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	systemName System Name associated with this experiment
 */
	public void setSystemName (String systemName) {
		this.systemName = systemName;
	}

/**
 * Sets the System MachineType associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	systemMachineType System MachineType associated with this experiment
 */
	public void setSystemMachineType (String systemMachineType) {
		this.systemMachineType = systemMachineType;
	}

/**
 * Sets the System Arch associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	systemArch System Arch associated with this experiment
 */
	public void setSystemArch (String systemArch) {
		this.systemArch = systemArch;
	}

/**
 * Sets the System OS associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	systemOS System OS associated with this experiment
 */
	public void setSystemOS (String systemOS) {
		this.systemOS = systemOS;
	}

/**
 * Sets the System MemorySize associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	systemMemorySize System MemorySize associated with this experiment
 */
	public void setSystemMemorySize (String systemMemorySize) {
		this.systemMemorySize = systemMemorySize;
	}

/**
 * Sets the System ProcessorAmount associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	systemProcessorAmount System ProcessorAmount associated with this experiment
 */
	public void setSystemProcessorAmount (String systemProcessorAmount) {
		this.systemProcessorAmount = systemProcessorAmount;
	}

/**
 * Sets the System L1CacheSize associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	systemL1CacheSize System L1CacheSize associated with this experiment
 */
	public void setSystemL1CacheSize (String systemL1CacheSize) {
		this.systemL1CacheSize = systemL1CacheSize;
	}

/**
 * Sets the System L2CacheSize associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	systemL2CacheSize System L2CacheSize associated with this experiment
 */
	public void setSystemL2CacheSize (String systemL2CacheSize) {
		this.systemL2CacheSize = systemL2CacheSize;
	}

/**
 * Sets the System UserData associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	systemUserData System UserData associated with this experiment
 */
	public void setSystemUserData (String systemUserData) {
		this.systemUserData = systemUserData;
	}

/**
 * Sets the Configuration Prefix associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	configurationPrefix Configuration Prefix associated with this experiment
 */
	public void setConfigurationPrefix (String configurationPrefix) {
		this.configurationPrefix = configurationPrefix;
	}

/**
 * Sets the Configuration Architecture associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	configurationArchitecture Configuration Architecture associated with this experiment
 */
	public void setConfigurationArchitecture (String configurationArchitecture) {
		this.configurationArchitecture = configurationArchitecture;
	}

/**
 * Sets the Configuration Cpp associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	configurationCpp Configuration Cpp associated with this experiment
 */
	public void setConfigurationCpp (String configurationCpp) {
		this.configurationCpp = configurationCpp;
	}

/**
 * Sets the Configuration Cc associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	configurationCc Configuration Cc associated with this experiment
 */
	public void setConfigurationCc (String configurationCc) {
		this.configurationCc = configurationCc;
	}

/**
 * Sets the Configuration Jdk associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	configurationJdk Configuration Jdk associated with this experiment
 */
	public void setConfigurationJdk (String configurationJdk) {
		this.configurationJdk = configurationJdk;
	}

/**
 * Sets the Configuration Profile associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	configurationProfile Configuration Profile associated with this experiment
 */
	public void setConfigurationProfile (String configurationProfile) {
		this.configurationProfile = configurationProfile;
	}

/**
 * Sets the Configuration UserData associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	configurationUserData Configuration UserData associated with this experiment
 */
	public void setConfigurationUserData (String configurationUserData) {
		this.configurationUserData = configurationUserData;
	}

/**
 * Sets the Compiler CppName associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	compilerCppName Compiler CppName associated with this experiment
 */
	public void setCompilerCppName (String compilerCppName) {
		this.compilerCppName = compilerCppName;
	}

/**
 * Sets the Compiler CppVersion associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	compilerCppVersion Compiler CppVersion associated with this experiment
 */
	public void setCompilerCppVersion (String compilerCppVersion) {
		this.compilerCppVersion = compilerCppVersion;
	}

/**
 * Sets the Compiler CcName associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	compilerCcName Compiler CcName associated with this experiment
 */
	public void setCompilerCcName (String compilerCcName) {
		this.compilerCcName = compilerCcName;
	}

/**
 * Sets the Compiler CcVersion associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	compilerCcVersion Compiler CcVersion associated with this experiment
 */
	public void setCompilerCcVersion (String compilerCcVersion) {
		this.compilerCcVersion = compilerCcVersion;
	}

/**
 * Sets the Compiler JavaDirpath associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	compilerJavaDirpath Compiler JavaDirpath associated with this experiment
 */
	public void setCompilerJavaDirpath (String compilerJavaDirpath) {
		this.compilerJavaDirpath = compilerJavaDirpath;
	}

/**
 * Sets the Compiler JavaVersion associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	compilerJavaVersion Compiler JavaVersion associated with this experiment
 */
	public void setCompilerJavaVersion (String compilerJavaVersion) {
		this.compilerJavaVersion = compilerJavaVersion;
	}

/**
 * Sets the Compiler UserData associated with this experiment.
 * <i>NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	compilerUserData Compiler UserData associated with this experiment
 */
	public void setCompilerUserData (String compilerUserData) {
		this.compilerUserData = compilerUserData;
	}

}


