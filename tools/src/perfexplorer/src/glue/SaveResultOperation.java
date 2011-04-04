/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;

/**
 * @author khuck
 *
 */
public class SaveResultOperation extends AbstractPerformanceOperation {

    /**
     * 
     */
    private static final long serialVersionUID = 1285952458769047610L;
    private boolean forceOverwrite = false;
    //private int maxNodes = 1;
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

		    Map<String,Integer> events= insertEvents( input.getEvents());		 
		    events= insertEvents( input.getEvents());


		    // first, get the node/context/trial combinations
		    getNodeContextThreadInfo();

		    // get the list of metrics - we are likely to have derived some metrics
		    Set<String> oldMetricSet = new HashSet<String>();
		    Iterator<Metric> iter = trial.getMetrics().iterator();
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
			    PreparedStatement statementILP = getStatementILP();
			    PreparedStatement statementTA1 = getStatementTA1();
			    PreparedStatement statementTA2 = getStatementTA2();

			    // need to insert a new metric
			    int metricID = insertMetric(metric);
			    for (String event : input.getEvents()) {
				// check for / insert the event
				int eventID = events.get(event);
				curNode = 0;
				curContext = 0;
				curThread = -1;
				for (int i = 0 ; i < accumulators.length ; i++) 
				    accumulators[i] = 0.0;
				for (Integer thread : input.getThreads()) {
				    // insert the Interval Location Profile
				    double exclusive = input.getExclusive(thread, event, metric);
				    double inclusive =input.getInclusive(thread, event, metric);
				    double calls = input.getCalls(thread, event);
				    double sub = input.getSubroutines(thread, event);
				    insertILP(statementILP, metricID, eventID, inclusive, exclusive, calls, sub);
				}
				insertTotalAndAverage(statementTA1, statementTA2, input.getThreads().size(), metricID, eventID);
			    }
			    statementILP.executeBatch();
			    statementILP.close();
			    statementTA1.executeBatch();
			    statementTA1.close();
			    statementTA2.executeBatch();
			    statementTA2.close();
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
    private PreparedStatement getStatementTA2() throws SQLException {
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
	}else if (db.getDBType().compareTo("mysql") == 0) {
	    buf.append("`call`, subroutines, ");
	} else {
	    buf.append("call, subroutines, ");
	}
	buf.append("inclusive_per_call) values (?, ?, ?, ?, ?, ?, ?, ?, ?)");
	return db.prepareStatement(buf.toString());
    }

    private PreparedStatement getStatementTA1() throws SQLException {
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
	}else if (db.getDBType().compareTo("mysql") == 0) {
	    buf.append("`call`, subroutines, ");

	} else {
	    buf.append("call, subroutines, ");
	}
	buf.append("inclusive_per_call) values (?, ?, ?, ?, ?, ?, ?, ?, ?)");
	return db.prepareStatement(buf.toString());
    }

    private PreparedStatement getStatementILP() throws SQLException {
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
	}else if (db.getDBType().compareTo("mysql") == 0) {
	    buf.append("`call`, subroutines, ");
	} else {

	    buf.append("call, subroutines, ");
	}
	buf.append("inclusive_per_call) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
	return db.prepareStatement(buf.toString());
    }

    private Map<String, Integer> insertEvents(Set<String> events) throws SQLException {
	HashMap<String, Integer> map = new HashMap<String, Integer>();

	buf = new StringBuilder();
	buf.append("insert into interval_event (trial, name) values (?, ?)");
	PreparedStatement preStatement = db.prepareStatement(buf.toString());

	for(String event: events){
	    int eventID = 0;
	    buf = new StringBuilder();
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
		preStatement.setInt(1, trial.getID());
		preStatement.setString(2, event);	
	    }
	    map.put(event, eventID);
	}
	preStatement.executeBatch();
	preStatement.close();	
	return map;
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
	    //this.maxNodes = results.getInt(1);
	    this.maxContexts = results.getInt(2);
	    this.maxThreads = results.getInt(3);
	}
	//System.out.println("Max Nodes: " + maxNodes);
	//System.out.println("Max Contexts: " + maxContexts);
	//System.out.println("Max Threads: " + maxThreads);
	results.close();
	statement.close();		
    }

    private void insertILP(PreparedStatement statementILP, int metricID, int eventID, double inclusive, double exclusive, double calls, double sub) throws SQLException {


	curThread++;
	if (curThread >= maxThreads) {
	    curThread = 0;
	    curContext++;
	}
	if (curContext >= maxContexts) {
	    curContext = 0;
	    curNode++;
	}

	statementILP.setInt(1, eventID);
	statementILP.setInt(2, curNode);
	statementILP.setInt(3, curContext);
	statementILP.setInt(4, curThread);
	statementILP.setInt(5, metricID);
	if (mainInclusive == 0.0) {
	    statementILP.setDouble(6, 0.0);
	} else {
	    statementILP.setDouble(6, inclusive/mainInclusive);
	}
	statementILP.setDouble(7, inclusive);
	if (mainInclusive == 0.0) {
	    statementILP.setDouble(8, 0.0);
	} else {
	    statementILP.setDouble(8, exclusive/mainInclusive);
	}
	statementILP.setDouble(9, exclusive);
	statementILP.setDouble(10, calls);
	statementILP.setDouble(11, sub);
	if (calls == 0) {
	    statementILP.setDouble(12, 0.0);
	} else {
	    statementILP.setDouble(12, inclusive/calls);
	    accumulators[6] += inclusive/calls;
	}
	statementILP.addBatch();
//	System.out.println(statementILP);
	//	statement.execute();
	//	statement.close();		
	if (mainInclusive == 0.0) {
	} else {
	    accumulators[0] += inclusive/mainInclusive;
	}
	accumulators[1] += inclusive;
	if (mainInclusive == 0.0) {
	} else {
	    accumulators[2] += exclusive/mainInclusive;
	}
	accumulators[3] += exclusive;
	accumulators[4] += calls;
	accumulators[5] += sub;
    }

    private void insertTotalAndAverage(PreparedStatement statementTA1,PreparedStatement statementTA2, int totalThreads, int metricID, int eventID) throws SQLException {

	statementTA1.setInt(1, eventID);
	statementTA1.setInt(2, metricID);
	statementTA1.setDouble(3, accumulators[0]);
	statementTA1.setDouble(4, accumulators[1]);
	statementTA1.setDouble(5, accumulators[2]);
	statementTA1.setDouble(6, accumulators[3]);
	statementTA1.setDouble(7, accumulators[4]);
	statementTA1.setDouble(8, accumulators[5]);
	statementTA1.setDouble(9, accumulators[6]);
	statementTA1.addBatch();
	//System.out.println(statement);
	//	statement.execute();
	//	statement.close();		


	statementTA2.setInt(1, eventID);
	statementTA2.setInt(2, metricID);
	statementTA2.setDouble(3, accumulators[0]/totalThreads);
	statementTA2.setDouble(4, accumulators[1]/totalThreads);
	statementTA2.setDouble(5, accumulators[2]/totalThreads);
	statementTA2.setDouble(6, accumulators[3]/totalThreads);
	statementTA2.setDouble(7, accumulators[4]/totalThreads);
	statementTA2.setDouble(8, accumulators[5]/totalThreads);
	statementTA2.setDouble(9, accumulators[6]/totalThreads);
	statementTA2.addBatch();
	//System.out.println(statement);
	//	statement.execute();
	//	statement.close();		
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
     * @return metric index
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
