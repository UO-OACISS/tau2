/* 
   Name:       DatabaseAPI.java
   Author:     Robert Bell
  
  
   Description: API to the PerfDMF (Performance Database Management Framework).

   Things to do:
*/

package edu.uoregon.tau.dms.dss;

import java.util.*;
import java.sql.*;
import java.util.Date;
import edu.uoregon.tau.dms.database.DB;

public class DatabaseAPI{

    //####################################
    //Contructor(s).
    //####################################
    public DatabaseAPI(){}

    public DatabaseAPI(DB db){
	this.db = db;}
    //####################################
    //End - Contructor(s).
    //####################################

    //####################################
    //Public section.
    //####################################
    public void setDB(DB db){
	this.db = db;}

    public DB getDB(){
	return this.db;}

    public DataSessionIterator getApplicationList() throws DatabaseAPIException{
	ResultSet resultSet = null;
	Vector applications = new Vector();
	
	try{
	    StringBuffer buf = new StringBuffer();
	    buf.append("select id, name, version, description, language, ");
	    buf.append("paradigm, usage_text, execution_options, userdata from ");
	    buf.append(this.db.getSchemaPrefix());
	    buf.append("application order by name asc ");
	    
	    resultSet = db.executeQuery(buf.toString());	
	    while(resultSet.next() != false){
		Application application = new Application();
		application.setID(resultSet.getInt(1));
		application.setName(resultSet.getString(2));
		application.setVersion(resultSet.getString(3));
		application.setDescription(resultSet.getString(4));
		application.setLanguage(resultSet.getString(5));
		application.setParaDiag(resultSet.getString(6));
		application.setUsage(resultSet.getString(7));
		application.setExecutableOptions(resultSet.getString(8));
		application.setUserData(resultSet.getString(9));
		
		//Add the application.
		applications.addElement(application);
	    }
	    //Cleanup resources.
	    resultSet.close();
	}
	catch(Exception e){
	    throw new DatabaseAPIException(e);
	}
	return new DataSessionIterator(applications);
    }

    public DataSessionIterator getExperimentList(int applicationID) throws DatabaseAPIException{
	ResultSet resultSet = null;
	Vector experiments = new Vector();

	try{
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
	    buf.append("where application = " + applicationID);
	    buf.append(" order by name asc ");
	    
	
	    resultSet = db.executeQuery(buf.toString());	
	    while (resultSet.next() != false) {
		Experiment experiment = new Experiment();
		experiment.setID(resultSet.getInt(1));
		experiment.setApplicationID(resultSet.getInt(2));
		experiment.setName(resultSet.getString(3));
		experiment.setSystemName(resultSet.getString(4));
		experiment.setSystemMachineType(resultSet.getString(5));
		experiment.setSystemArch(resultSet.getString(6));
		experiment.setSystemOS(resultSet.getString(7));
		experiment.setSystemMemorySize(resultSet.getString(8));
		experiment.setSystemProcessorAmount(resultSet.getString(9));
		experiment.setSystemL1CacheSize(resultSet.getString(10));
		experiment.setSystemL2CacheSize(resultSet.getString(11));
		experiment.setSystemUserData(resultSet.getString(12));
		experiment.setConfigurationPrefix(resultSet.getString(13));
		experiment.setConfigurationArchitecture(resultSet.getString(14));
		experiment.setConfigurationCpp(resultSet.getString(15));
		experiment.setConfigurationCc(resultSet.getString(16));
		experiment.setConfigurationJdk(resultSet.getString(17));
		experiment.setConfigurationProfile(resultSet.getString(18));
		experiment.setConfigurationUserData(resultSet.getString(19));
		experiment.setCompilerCppName(resultSet.getString(20));
		experiment.setCompilerCppVersion(resultSet.getString(21));
		experiment.setCompilerCcName(resultSet.getString(22));
		experiment.setCompilerCcVersion(resultSet.getString(23));
		experiment.setCompilerJavaDirpath(resultSet.getString(24));
		experiment.setCompilerJavaVersion(resultSet.getString(25));
		experiment.setCompilerUserData(resultSet.getString(26));
		experiment.setUserData(resultSet.getString(27));

		//Add the experiment.
		experiments.addElement(experiment);
	    }
	    //Cleanup resources.
	    resultSet.close();
	}
	catch(Exception e){
	    throw new DatabaseAPIException(e);
	}
	return new DataSessionIterator(experiments);
    }

    public DataSessionIterator getTrialList(int experiementID) throws DatabaseAPIException{
	ResultSet resultSet = null;
	Vector trials = new Vector();

	try{
	    StringBuffer buf = new StringBuffer();
	    buf.append("select t.id, t.experiment, e.application, ");
	    buf.append("t.time, t.problem_definition, t.node_count, ");
	    buf.append("t.contexts_per_node, t.threads_per_context, ");
	    buf.append("t.name, t.userdata ");
	    buf.append("from trial t inner join experiment e ");
	    buf.append("on t.experiment = e.id ");
	    buf.append("where t.experiment = " + experiementID);
	    buf.append(" order by t.node_count, t.contexts_per_node, t.threads_per_context, t.id ");
	    
	    resultSet = db.executeQuery(buf.toString());	
	    while(resultSet.next() != false){
		Trial trial = new Trial();
		trial.setID(resultSet.getInt(1));
		trial.setExperimentID(resultSet.getInt(2));
		trial.setApplicationID(resultSet.getInt(3));
		trial.setTime(resultSet.getString(4));
		trial.setProblemDefinition(resultSet.getString(5));
		trial.setNodeCount(resultSet.getInt(6));
		trial.setNumContextsPerNode(resultSet.getInt(7));
		trial.setNumThreadsPerContext(resultSet.getInt(8));
		trial.setName(resultSet.getString(9));
		trial.setUserData(resultSet.getString(10));

		//Add the trial.
		trials.addElement(trial);
	    }
	    //Cleanup resources.
	    resultSet.close();
	}
	catch(Exception e){
	    throw new DatabaseAPIException(e);
	}
	return new DataSessionIterator(trials);
    }

    // gets the metric data for the trial
    private DataSessionIterator getMetricList(int trialID) throws DatabaseAPIException{
	ResultSet resultSet = null;
	Vector metrics = new Vector();

	try{
	    StringBuffer buf = new StringBuffer();
	    buf.append("select id, name ");
	    buf.append("from metric where trial = " + trialID);
	    buf.append(" order by id ");

	    while (resultSet.next() != false) {
		Metric metric = new Metric();
		metric.setID(resultSet.getInt(1));
		metric.setName(resultSet.getString(2));
		metric.setTrialID(trialID);
		
		//Add the metric.
		metrics.addElement(metric);
	    }
	    //Cleanup resources.
	    resultSet.close();
	}
	catch(Exception e){
	    throw new DatabaseAPIException(e);
	}
	return new DataSessionIterator(metrics);
    }

    public DataSessionIterator getIntervalEvents() throws DatabaseAPIException{
	ResultSet resultSet = null;
	Vector intervalEvents = new Vector();

	try{
	    while (resultSet.next() != false) {
		//Add the interval event.
		//intervalEvents.addElement(intervalEvent);
	    }
	    //Cleanup resources.
	    resultSet.close();
	}
	catch(Exception e){
	    throw new DatabaseAPIException(e);
	}
	return new DataSessionIterator(intervalEvents);
    }


    //####################################
    //End - Public section.
    //####################################

    //####################################
    //Protected section.
    //####################################
    //####################################
    //End - Protected section.
    //####################################

    //####################################
    //Private section.
    //####################################
    //####################################
    //End - Private section.
    //####################################


    //####################################
    //Instance data.
    //####################################
    private DB db = null;
    //####################################
    //End - Instance data.
    //####################################
}

class DatabaseAPIException extends Exception{
    
    public DatabaseAPIException(String exceptionString){
	super(exceptionString);}

    public DatabaseAPIException(Exception e){
	super(e.toString());}
}
