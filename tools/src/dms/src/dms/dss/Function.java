package dms.dss;

import java.util.Vector;
import java.util.Enumeration;
import dms.perfdb.DB;
import java.sql.SQLException;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
/**
 * Holds all the data for a function in the database.
 * This object is returned by the DataSession class and all of its subtypes.
 * The Function object contains all the information associated with
 * an function from which the TAU performance data has been generated.
 * A function is associated with one trial, experiment and application, and has one or more
 * FunctionDataObject objects (one for each node/context/thread location in the trial) associated with it.  
 * <p>
 * A function has information
 * related to one particular function in the application, including the name of the function,
 * the TAU group it belongs to, and all of the total and mean data for the function. 
 * In order to see particular measurements for a node/context/thread/metric instance,
 * get the FunctionDataObject(s) for this Function.  In order to access the total
 * or mean data, getTotalSummary() and getMeanSummary() methods are provided.  The
 * index of the metric in the Trial object should be used to indicate which total/mean
 * summary object to return.
 *
 * <P>CVS $Id: Function.java,v 1.4 2004/04/02 23:28:17 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getFunctions
 * @see		DataSession#setFunction
 * @see		Application
 * @see		Experiment
 * @see		Trial
 * @see		FunctionDataObject
 */
public class Function {
	private int functionID;
	private String name;
	private String group;
	private int trialID;
	private Vector meanSummary = null;
	private Vector totalSummary = null;
	private DataSession dataSession = null;

	public Function (DataSession dataSession) {
		this.dataSession = dataSession;
	}

/**
 * Gets the unique identifier of this function object.
 *
 * @return	the unique identifier of the function
 */
	public int getID () {
		return this.functionID;
	}

/**
 * Gets the function name.
 *
 * @return	the name of the function
 */
	public String getName () {
		return this.name;
	}

/**
 * Gets the TAU group of this function object.
 *
 * @return	the TAU group the function is in.
 */
	public String getGroup () {
		return this.group;
	}

/**
 * Gets the trial ID of this function object.
 *
 * @return	the trial ID for the function.
 */
	public int getTrialID () {
		return this.trialID;
	}

/**
 * Gets mean summary data for the function object.
 * The Trial object associated with this function has a vector of metric
 * names that represent the metric data stored for each function in the
 * application.  Multiple metrics can be recorded for each trial.  The index
 * of the metric desired should be passed into this function to get the
 * mean data for this function/trial/experiment/application combination.
 * The mean data is averaged across all locations, defined as any combination
 * of node/context/thread.
 *
 * @param	metricIndex the metric index for the desired metric.
 * @return	the FunctionDataObject containing the mean data for this function/metric combination.
 * @see		Trial
 * @see		FunctionDataObject
 */
	public FunctionDataObject getMeanSummary (int metricIndex) {
		if (this.meanSummary == null) {
			Vector metrics = dataSession.getMetrics();
			dataSession.setMetrics(null);
			dataSession.getFunctionDetail(this);
			dataSession.setMetrics(metrics);
		}
		return (FunctionDataObject)(this.meanSummary.elementAt(metricIndex));
	}

/**
 * Gets mean summary data for the function object.
 * The Trial object associated with this function has a vector of metric
 * names that represent the metric data stored for each function in the
 * application.  Multiple metrics can be recorded for each trial.  If the
 * user has selected a metric before calling getFunctions(), then this method
 * can be used to return the mean metric data for this 
 * function/trial/experiment/application combination.
 * The mean data is averaged across all locations, defined as any combination
 * of node/context/thread.
 *
 * @return	the FunctionDataObject containing the mean data for this function/metric combination.
 * @see		Trial
 * @see		FunctionDataObject
 * @see		DataSession#getFunctions
 * @see		DataSession#setMetric(String)
 */
	public FunctionDataObject getMeanSummary () {
		if (this.meanSummary == null)
			dataSession.getFunctionDetail(this);
		return (FunctionDataObject)(this.meanSummary.elementAt(0));
	}

/**
 * Gets total summary data for the function object.
 * The Trial object associated with this function has a vector of metric
 * names that represent the metric data stored for each function in the
 * application.  Multiple metrics can be recorded for each trial.  The index
 * of the metric desired should be passed into this function to get the
 * total data for this function/trial/experiment/application combination.
 * The total data is summed across all locations, defined as any combination
 * of node/context/thread.
 *
 * @param	metricIndex the metric index for the desired metric.
 * @return	the FunctionDataObject containing the total data for this function/metric combination.
 * @see		Trial
 * @see		FunctionDataObject
 */
	public FunctionDataObject getTotalSummary (int metricIndex) {
		if (this.totalSummary == null) {
			Vector metrics = dataSession.getMetrics();
			dataSession.setMetrics(null);
			dataSession.getFunctionDetail(this);
			dataSession.setMetrics(metrics);
		}
		return (FunctionDataObject)(this.totalSummary.elementAt(metricIndex));
	}

/**
 * Gets total summary data for the function object.
 * The Trial object associated with this function has a vector of metric
 * names that represent the metric data stored for each function in the
 * application.  Multiple metrics can be recorded for each trial.  If the
 * user has selected a metric before calling getFunctions(), then this method
 * can be used to return the total metric data for this 
 * function/trial/experiment/application combination.
 * The total data is summed across all locations, defined as any combination
 * of node/context/thread.
 *
 * @return	the FunctionDataObject containing the total data for this function/metric combination.
 * @see		Trial
 * @see		FunctionDataObject
 * @see		DataSession#getFunctions
 */
	public FunctionDataObject getTotalSummary () {
		if (this.totalSummary == null)
			dataSession.getFunctionDetail(this);
		return (FunctionDataObject)(this.totalSummary.elementAt(0));
	}

/**
 * Sets the unique ID associated with this funciton.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	id unique ID associated with this function
 */
	public void setID (int id) {
		this.functionID = id;
	}

/**
 * Sets the function name.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	name the name of the function
 */
	public void setName (String name) {
		this.name = name;
	}

/**
 * Sets the TAU group of this function object.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	group the TAU group the function is in.
 */
	public void setGroup (String group) {
		this.group = group;
	}

/**
 * Sets the trial ID of this function object.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	id the trial ID for the function.
 */
	public void setTrialID (int id) {
		this.trialID = id;
	}

/**
 * Adds a FunctionDataObject to the function as a mean summary.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	meanSummary the mean summary object for the function.
 */
	public void addMeanSummary (FunctionDataObject meanSummary) {
		if (this.meanSummary == null)
			this.meanSummary = new Vector();
		this.meanSummary.addElement(meanSummary);
	}

/**
 * Clears the function's vector of mean summary objects.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 */
	public void clearMeanSummary () {
		this.meanSummary = null;
	}

/**
 * Adds a FunctionDataObject to the function as a total summary.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	totalSummary the total summary object for the function.
 */
	public void addTotalSummary (FunctionDataObject totalSummary) {
		if (this.totalSummary == null)
			this.totalSummary = new Vector();
		this.totalSummary.addElement(totalSummary);
	}

/**
 * Clears the function's vector of total summary objects.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 */
	public void clearTotalSummary () {
		this.totalSummary = null;
	}

	// returns a Vector of Functions
	public static Vector getFunctions(DataSession dataSession, DB db, String whereClause) {
		Vector funs = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select distinct id, name, group_name, trial ");
		buf.append("from function ");
		buf.append(whereClause);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
			Function tmpFunction = null;
	    	while (resultSet.next() != false) {
				Function fun = new Function(dataSession);
				fun.setID(resultSet.getInt(1));
				fun.setName(resultSet.getString(2));
				fun.setGroup(resultSet.getString(3));
				fun.setTrialID(resultSet.getInt(4));
				funs.addElement(fun);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return funs;
	}

	public int saveFunction(DB db, int newTrialID, Vector metricID) {
		int newFunctionID = 0;
		try {
			PreparedStatement statement = null;
			statement = db.prepareStatement("INSERT INTO function (trial, name, group_name) VALUES (?, ?, ?)");
			statement.setInt(1, newTrialID);
			statement.setString(2, name);
			statement.setString(3, group);
			statement.executeUpdate();
			String tmpStr = new String();
			if (db.getDBType().compareTo("mysql") == 0)
				tmpStr = "select LAST_INSERT_ID();";
			else
				tmpStr = "select currval('function_id_seq');";
			newFunctionID = Integer.parseInt(db.getDataItem(tmpStr));
		} catch (SQLException e) {
			System.out.println("An error occurred while saving the function.");
			e.printStackTrace();
			System.exit(0);
		}
		// save the function mean summaries
		if (meanSummary != null) {
			Enumeration enum = meanSummary.elements();
			FunctionDataObject fdo;
			while (enum.hasMoreElements()) {
				fdo = (FunctionDataObject)enum.nextElement();
				fdo.saveMeanSummary(db, newFunctionID, metricID);
			}
		}

		// save the function total summaries
		if (totalSummary != null) {
			Enumeration enum = totalSummary.elements();
			FunctionDataObject fdo;
			while (enum.hasMoreElements()) {
				fdo = (FunctionDataObject)enum.nextElement();
				fdo.saveTotalSummary(db, newFunctionID, metricID);
			}
		}
			return newFunctionID;
	}
}

