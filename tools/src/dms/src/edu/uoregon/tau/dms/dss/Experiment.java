package edu.uoregon.tau.dms.dss;

import edu.uoregon.tau.dms.database.DB;
import java.sql.SQLException;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.Vector;
import java.io.Serializable;

/**
 * Holds all the data for an experiment in the database.  
 * This object is returned by the DataSession class and all of its subtypes.
 * The Experiment object contains all the information associated with
 * an experiment from which the TAU performance data has been generated.
 * An experiment is associated with an application, and has one or more
 * trials associated with it.
 *
 * <P>CVS $Id: Experiment.java,v 1.5 2004/10/27 21:34:30 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getExperimentList
 * @see		DataSession#setExperiment
 * @see		Application
 * @see		Trial
 */
public class Experiment implements Serializable {
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
	
    public String toString() {
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

    public static Vector getExperimentList (DB db, String whereClause) {
	Vector experiments = new Vector();
	// create a string to hit the database
	StringBuffer buf = new StringBuffer();
	buf.append("select id, application, name, system_name, ");
	buf.append("system_machine_type, system_arch, system_os, ");
	buf.append("system_memory_size, system_processor_amt, ");
	buf.append("system_l1_cache_size, system_l2_cache_size, ");
	buf.append("system_userdata, ");
	buf.append("configure_prefix, configure_arch, configure_cpp, ");
	buf.append("configure_cc, configure_jdk, configure_profile, ");
	buf.append("configure_userdata, ");
	buf.append("compiler_cpp_name, compiler_cpp_version, ");
	buf.append("compiler_cc_name, compiler_cc_version, ");
	buf.append("compiler_java_dirpath, compiler_java_version, ");
	buf.append("compiler_userdata, userdata from experiment ");
	buf.append(whereClause);

	if (db.getDBType().compareTo("oracle") == 0) {
	    buf.append(" order by dbms_lob.substr(name) asc");
	} else {
	    buf.append(" order by name asc ");
	}
	// System.out.println(buf.toString());

	// get the results
	try {
	    ResultSet resultSet = db.executeQuery(buf.toString());	
	    while (resultSet.next() != false) {
		Experiment exp = new Experiment();
		exp.setID(resultSet.getInt(1));
		exp.setApplicationID(resultSet.getInt(2));
		exp.setName(resultSet.getString(3));
		exp.setSystemName(resultSet.getString(4));
		exp.setSystemMachineType(resultSet.getString(5));
		exp.setSystemArch(resultSet.getString(6));
		exp.setSystemOS(resultSet.getString(7));
		exp.setSystemMemorySize(resultSet.getString(8));
		exp.setSystemProcessorAmount(resultSet.getString(9));
		exp.setSystemL1CacheSize(resultSet.getString(10));
		exp.setSystemL2CacheSize(resultSet.getString(11));
		exp.setSystemUserData(resultSet.getString(12));
		exp.setConfigurationPrefix(resultSet.getString(13));
		exp.setConfigurationArchitecture(resultSet.getString(14));
		exp.setConfigurationCpp(resultSet.getString(15));
		exp.setConfigurationCc(resultSet.getString(16));
		exp.setConfigurationJdk(resultSet.getString(17));
		exp.setConfigurationProfile(resultSet.getString(18));
		exp.setConfigurationUserData(resultSet.getString(19));
		exp.setCompilerCppName(resultSet.getString(20));
		exp.setCompilerCppVersion(resultSet.getString(21));
		exp.setCompilerCcName(resultSet.getString(22));
		exp.setCompilerCcVersion(resultSet.getString(23));
		exp.setCompilerJavaDirpath(resultSet.getString(24));
		exp.setCompilerJavaVersion(resultSet.getString(25));
		exp.setCompilerUserData(resultSet.getString(26));
		exp.setUserData(resultSet.getString(27));
		experiments.addElement(exp);
	    }
	    resultSet.close(); 
	}catch (Exception ex) {
	    ex.printStackTrace();
	    return null;
	}
		
	return experiments;
    }

    public int saveExperiment(DB db) {
	boolean itExists = exists(db);
	int newExperimentID = 0;
	try {
	    PreparedStatement statement = null;
	    if (itExists) {
		statement = db.prepareStatement("UPDATE experiment SET application = ?, name = ?, system_name = ?, system_machine_type = ?, system_arch = ?, system_os = ?, system_memory_size = ?, system_processor_amt = ?, system_l1_cache_size = ?, system_l2_cache_size = ?, system_userdata = ?, compiler_cpp_name = ?, compiler_cpp_version = ?, compiler_cc_name = ?, compiler_cc_version = ?, compiler_java_dirpath = ?, compiler_java_version = ?, compiler_userdata = ?, configure_prefix = ?, configure_arch = ?, configure_cpp = ?, configure_cc = ?, configure_jdk = ?, configure_profile = ?, configure_userdata = ?, userdata = ? WHERE id = ?");
	    } else {
		statement = db.prepareStatement("INSERT INTO experiment (application, name, system_name, system_machine_type, system_arch, system_os, system_memory_size, system_processor_amt, system_l1_cache_size, system_l2_cache_size, system_userdata, compiler_cpp_name, compiler_cpp_version, compiler_cc_name, compiler_cc_version, compiler_java_dirpath, compiler_java_version, compiler_userdata, configure_prefix, configure_arch, configure_cpp, configure_cc, configure_jdk, configure_profile, configure_userdata, userdata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
	    }
	    statement.setInt(1, applicationID);
	    statement.setString(2, name);
	    statement.setString(3, systemName);
	    statement.setString(4, systemMachineType);
	    statement.setString(5, systemArch);
	    statement.setString(6, systemOS);
	    statement.setString(7, systemMemorySize);
	    statement.setString(8, systemProcessorAmount);
	    statement.setString(9, systemL1CacheSize);
	    statement.setString(10, systemL2CacheSize);
	    statement.setString(11, systemUserData);
	    statement.setString(12, compilerCppName);
	    statement.setString(13, compilerCppVersion);
	    statement.setString(14, compilerCcName);
	    statement.setString(15, compilerCcVersion);
	    statement.setString(16, compilerJavaDirpath);
	    statement.setString(17, compilerJavaVersion);
	    statement.setString(18, compilerUserData);
	    statement.setString(19, configurationPrefix);
	    statement.setString(20, configurationArchitecture);
	    statement.setString(21, configurationCpp);
	    statement.setString(22, configurationCc);
	    statement.setString(23, configurationJdk);
	    statement.setString(24, configurationProfile);
	    statement.setString(25, configurationUserData);
	    statement.setString(26, userData);
	    if (itExists) {
		statement.setInt(27, experimentID);
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
		} else if (db.getDBType().compareTo("oracle") == 0) {
		    tmpStr = "select experiment_id_seq.currval FROM dual";
		} else {
		    tmpStr = "select currval('experiment_id_seq');";
		}
		newExperimentID = Integer.parseInt(db.getDataItem(tmpStr));
	    }
	} catch (SQLException e) {
	    System.out.println("An error occurred while saving the experiment.");
	    e.printStackTrace();
	    System.exit(0);
	}
	return newExperimentID;
    }

    private boolean exists(DB db) {
	boolean retval = false;
	try {
	    PreparedStatement statement = db.prepareStatement("SELECT application FROM experiment WHERE id = ?");
	    statement.setInt(1, experimentID);
	    ResultSet results = statement.executeQuery();
	    while (results.next() != false) {
		retval = true;
		break;
	    }
	    results.close();
	} catch (SQLException e) {
	    System.out.println("An error occurred while saving the experiment.");
	    e.printStackTrace();
	    System.exit(0);
	}
	return retval;
    }

    public static void deleteExperiment(DB db, int experimentID) {
	try {
	    PreparedStatement statement = null;
	    statement = db.prepareStatement("delete from experiment where id = ?");
	    statement.setInt(1, experimentID);
	    statement.execute();
	    statement.close();
	} catch (SQLException e) {
	    System.out.println("An error occurred while deleting the experiment.");
	    e.printStackTrace();
	    System.exit(0);
	}
    }
}


