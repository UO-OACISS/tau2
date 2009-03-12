/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.Vector;


import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;

/**
 * @author khuck
 *
 */
public class SaveResultOperation extends AbstractPerformanceOperation {
	
	private boolean forceOverwrite = false;
	private int maxNodes = 1;
	private int maxContexts = 1;
	private int maxThreads = 1;
	private int curNode = 0;
	private int curContext = 0;
	private int curThread = -1;
	private DB db = null;
	private Trial trial;
	private StringBuilder buf = null;
	private PreparedStatement statement = null;
	private double mainInclusive = 0.0;
	private double[] accumulators = {0,0,0,0,0,0,0};

	/**
	 * @param input
	 */
	public SaveResultOperation(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param trial
	 */
	public SaveResultOperation(Trial trial) {
		super(trial);
		this.trial = trial;
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param inputs
	 */
	public SaveResultOperation(List<PerformanceResult> inputs) {
		super(inputs);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceAnalysisOperation#processData()
	 */
	public List<PerformanceResult> processData() {
		
		for (PerformanceResult input : inputs) {
			trial = input.getTrial();
			// first, check to see if the trial ID is set, or if the Trial is set.
			if (trial == null) {
				System.err.println("SaveResultOperation.processData() not implemented for " + input.getClass().getName());
				System.err.println("The Trial reference is null - no idea where this data is supposed to go.");
			} else {
				db = PerfExplorerServer.getServer().getDB();
				try {
					db.setAutoCommit(false);

					// first, get the node/context/trial combinations
					getNodeContextThreadInfo();
					
					// get the list of metrics - we are likely to have derived some metrics
					Set<String> oldMetricSet = new HashSet<String>();
					Iterator iter = trial.getMetrics().iterator();
					while (iter.hasNext()) {
						Metric tmpMetric = (Metric)iter.next();
						oldMetricSet.add(tmpMetric.getName());
					}
					// for each metric in the trial, check to see if it exists already.
					for (String metric : input.getMetrics()) {
						if (oldMetricSet.contains(metric)) {
							// check for force, otherwise continue
							if (forceOverwrite) {
								// do update
							} else {
								// output an error
							}
						} else {
							// get the main inclusive value, we need it.
							mainInclusive = input.getInclusive(0, input.getMainEvent(), metric);

							// need to insert a new metric
							int metricID = insertMetric(metric);
							for (String event : input.getEvents()) {
								// check for / insert the event
								int eventID = insertEvent(event);
								curNode = 0;
								curContext = 0;
								curThread = -1;
								for (int i = 0 ; i < accumulators.length ; i++) 
									accumulators[i] = 0.0;
								for (Integer thread : input.getThreads()) {
									// insert the Interval Location Profile
									insertILP(input, metricID, eventID, thread, event, metric);
								}
								insertTotalAndAverage(input, metricID, eventID, event, metric);
							}
							db.commit();
						}
					}
				} catch (Exception exception) {
					System.err.println(exception.getMessage());
					if (db.getDBType().equals("postgresql"))
						System.err.println(statement.toString());
					else
						System.err.println(buf.toString());
					exception.printStackTrace();
					try {
						db.rollback();
						db.setAutoCommit(true);
					} catch (Exception e2) {}
				}
			}
		}
			
		// return the input files
		return this.inputs;
	}

	private void getNodeContextThreadInfo() throws SQLException {
		buf = new StringBuilder();
		//event = event + ".kevin";
		buf.append("select node_count, contexts_per_node, threads_per_context from trial where id = ?");
		statement = db.prepareStatement(buf.toString());
		statement.setInt(1, trial.getID());
		//System.out.println(statement);
		ResultSet results = statement.executeQuery();
		if (results.next() != false) {
			this.maxNodes = results.getInt(1);
			this.maxContexts = results.getInt(2);
			this.maxThreads = results.getInt(3);
		}
		//System.out.println("Max Nodes: " + maxNodes);
		//System.out.println("Max Contexts: " + maxContexts);
		//System.out.println("Max Threads: " + maxThreads);
		results.close();
		statement.close();		
	}

	private void insertILP(PerformanceResult input, int metricID, int eventID, Integer thread, String event, String metric) throws SQLException {
		curThread++;
		if (curThread >= maxThreads) {
			curThread = 0;
			curContext++;
		}
		if (curContext >= maxContexts) {
			curContext = 0;
			curNode++;
		}
		buf = new StringBuilder();
		buf.append("insert into interval_location_profile (interval_event, node, ");
		buf.append("context, thread, metric, inclusive_percentage, inclusive, ");
        if (db.getDBType().compareTo("oracle") == 0) {
			buf.append("exclusive_percentage, excl, ");
		} else {
			buf.append("exclusive_percentage, exclusive, ");
		}
		if (db.getDBType().compareTo("derby") == 0) {
			buf.append("num_calls, subroutines, ");
		} else {
			buf.append("call, subroutines, ");
		}
		buf.append("inclusive_per_call) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
		statement = db.prepareStatement(buf.toString());
		statement.setInt(1, eventID);
		statement.setInt(2, curNode);
		statement.setInt(3, curContext);
		statement.setInt(4, curThread);
		statement.setInt(5, metricID);
		if (mainInclusive == 0.0) {
			statement.setDouble(6, 0.0);
		} else {
			statement.setDouble(6, input.getInclusive(thread, event, metric)/mainInclusive);
		}
		statement.setDouble(7, input.getInclusive(thread, event, metric));
		if (mainInclusive == 0.0) {
			statement.setDouble(8, 0.0);
		} else {
			statement.setDouble(8, input.getExclusive(thread, event, metric)/mainInclusive);
		}
		statement.setDouble(9, input.getExclusive(thread, event, metric));
		statement.setDouble(10, input.getCalls(thread, event));
		statement.setDouble(11, input.getSubroutines(thread, event));
		if (input.getCalls(thread, event) == 0) {
			statement.setDouble(12, 0.0);
		} else {
			statement.setDouble(12, input.getInclusive(thread, event, metric)/input.getCalls(thread, event));
			accumulators[6] += input.getInclusive(thread, event, metric)/input.getCalls(thread, event);
		}
		//System.out.println(statement);
		statement.execute();
		statement.close();		
		if (mainInclusive == 0.0) {
		} else {
			accumulators[0] += input.getInclusive(thread, event, metric)/mainInclusive;
		}
		accumulators[1] += input.getInclusive(thread, event, metric);
		if (mainInclusive == 0.0) {
		} else {
			accumulators[2] += input.getExclusive(thread, event, metric)/mainInclusive;
		}
		accumulators[3] += input.getExclusive(thread, event, metric);
		accumulators[4] += input.getCalls(thread, event);
		accumulators[5] += input.getSubroutines(thread, event);
	}

	private void insertTotalAndAverage(PerformanceResult input, int metricID, int eventID, String event, String metric) throws SQLException {
		buf = new StringBuilder();
		buf.append("insert into interval_total_summary (interval_event, metric, ");
		buf.append("inclusive_percentage, inclusive, ");
        if (db.getDBType().compareTo("oracle") == 0) {
			buf.append("exclusive_percentage, excl, ");
		} else {
			buf.append("exclusive_percentage, exclusive, ");
		}
		if (db.getDBType().compareTo("derby") == 0) {
			buf.append("num_calls, subroutines, ");
		} else {
			buf.append("call, subroutines, ");
		}
		buf.append("inclusive_per_call) values (?, ?, ?, ?, ?, ?, ?, ?, ?)");
		statement = db.prepareStatement(buf.toString());
		statement.setInt(1, eventID);
		statement.setInt(2, metricID);
		statement.setDouble(3, accumulators[0]);
		statement.setDouble(4, accumulators[1]);
		statement.setDouble(5, accumulators[2]);
		statement.setDouble(6, accumulators[3]);
		statement.setDouble(7, accumulators[4]);
		statement.setDouble(8, accumulators[5]);
		statement.setDouble(9, accumulators[6]);
		//System.out.println(statement);
		statement.execute();
		statement.close();		

		buf = new StringBuilder();
		buf.append("insert into interval_mean_summary (interval_event, metric, ");
		buf.append("inclusive_percentage, inclusive, ");
        if (db.getDBType().compareTo("oracle") == 0) {
			buf.append("exclusive_percentage, excl, ");
		} else {
			buf.append("exclusive_percentage, exclusive, ");
		}
		if (db.getDBType().compareTo("derby") == 0) {
			buf.append("num_calls, subroutines, ");
		} else {
			buf.append("call, subroutines, ");
		}
		buf.append("inclusive_per_call) values (?, ?, ?, ?, ?, ?, ?, ?, ?)");
		statement = db.prepareStatement(buf.toString());
		statement.setInt(1, eventID);
		statement.setInt(2, metricID);
		int totalThreads = input.getThreads().size();
		statement.setDouble(3, accumulators[0]/totalThreads);
		statement.setDouble(4, accumulators[1]/totalThreads);
		statement.setDouble(5, accumulators[2]/totalThreads);
		statement.setDouble(6, accumulators[3]/totalThreads);
		statement.setDouble(7, accumulators[4]/totalThreads);
		statement.setDouble(8, accumulators[5]/totalThreads);
		statement.setDouble(9, accumulators[6]/totalThreads);
		//System.out.println(statement);
		statement.execute();
		statement.close();		
	}

	private int insertEvent(String event) throws SQLException {
		int eventID = 0;
		buf = new StringBuilder();
		//event = event + ".kevin";
		buf.append("select id from interval_event where trial = ? and name = ?");
		statement = db.prepareStatement(buf.toString());
		statement.setInt(1, trial.getID());
		statement.setString(2, event);
		//System.out.println(statement);
		ResultSet results = statement.executeQuery();
		if (results.next() != false) {
			eventID = results.getInt(1);
		}
		results.close();
		statement.close();
		// do we need to insert a new event?
		if (eventID == 0) {
			buf = new StringBuilder();
			buf.append("insert into interval_event (trial, name) values (?, ?)");
			statement = db.prepareStatement(buf.toString());
			statement.setInt(1, trial.getID());
			statement.setString(2, event);
			//System.out.println(statement);
			statement.execute();			
			statement.close();		
		}
		return eventID;
	}

	/**
	 * @param trial
	 * @param metric
	 * @param db
	 * @param buf
	 * @return
	 * @throws SQLException
	 * @throws NumberFormatException
	 */
	private int insertMetric(String metric) throws SQLException, NumberFormatException {
		
		int metricID;
		buf = new StringBuilder();
		buf.append("insert into metric (trial, name) values (?, ?) ");
		statement = db.prepareStatement(buf.toString());
		statement.setInt(1, trial.getID());
		statement.setString(2, metric);
		//System.out.println(statement);
		statement.execute();
		statement.close();
		statement.close();
		String tmpStr = new String();
		if (db.getDBType().compareTo("mysql") == 0) {
		tmpStr = "select LAST_INSERT_ID();";
		} else if (db.getDBType().compareTo("db2") == 0) {
		tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
		} else if (db.getDBType().compareTo("derby") == 0) {
		tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
		} else if (db.getDBType().compareTo("oracle") == 0) {
		tmpStr = "SELECT metric_id_seq.currval FROM DUAL";
		} else { // postgresql 
		tmpStr = "select currval('metric_id_seq');";
		}
		metricID = Integer.parseInt(db.getDataItem(tmpStr));
		return metricID;
	}

	/**
	 * @return the forceOverwrite
	 */
	public boolean isForceOverwrite() {
		return forceOverwrite;
	}

	/**
	 * @param forceOverwrite the forceOverwrite to set
	 */
	public void setForceOverwrite(boolean forceOverwrite) {
		this.forceOverwrite = forceOverwrite;
	}

}
