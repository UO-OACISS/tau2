package edu.uoregon.tau.dms.analysis;

import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.database.*;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Vector;
import java.util.ListIterator;

abstract public class DistanceAnalysis {
	protected DB db = null;
	protected DistanceMatrix results = null;
	protected Trial trial = null;
	protected Metric metric = null;

	public DistanceAnalysis (PerfDMFSession session, Trial inTrial, Metric inMetric) {
		this.db = session.db();
		this.trial = inTrial;
		this.metric = inMetric;
	}

	private void getRawData() {
		// calculate the threadCount;
		int threadCount = trial.getNodeCount() * trial.getNumContextsPerNode() * trial.getNumThreadsPerContext();

		// get the event count from the database
		StringBuffer buf = new StringBuffer();
		buf.append("select count(id) from interval_event where trial = ");
		buf.append(trial.getID());
		int eventCount = Integer.parseInt(db.getDataItem(buf.toString()));
		System.out.println(threadCount + " " + eventCount);

		// initialize the matrix
		results = createTheMatrix(threadCount, eventCount);

		// get the event names from the database
		buf = new StringBuffer();
		buf.append("select name from interval_event where trial = ? order by id");
		try {
			PreparedStatement statement = db.prepareStatement(buf.toString());
			statement.setInt(1, trial.getID());
			ResultSet resultSet = statement.executeQuery();          
			int i = 0;
			while (resultSet.next() != false) {
				results.eventName[i++] = resultSet.getString(1);
				// System.out.println(results.eventName[i-1]);
        	}
        	resultSet.close(); 
		} catch (SQLException e) {
			e.printStackTrace();
			return;
		}

		// build the query to get total amounts
		buf = new StringBuffer();
		buf.append("select l.node, l.context, l.thread, sum(l.exclusive) ");
		buf.append("from interval_location_profile l ");
		buf.append("inner join interval_event e ");
		buf.append("on l.interval_event = e.id ");
		buf.append("where e.trial = ? and l.metric = ? ");
		buf.append("group by l.node, l.context, l.thread ");
		buf.append("order by l.node, l.context, l.thread ");

		// get the totals
		getTotals();

		// get the raw data
		getMatrixData();
	}

	abstract protected void getTotals();

	abstract protected void getMatrixData();

	abstract protected DistanceMatrix createTheMatrix(int threadCount, int eventCount);
	
	public double[][] getManhattanDistance() {
		if (results == null) getRawData();
		results.getManhattanDistance();
		return results.distanceMatrix;
	}

	public double[][] getEuclidianDistance() {
		if (results == null) getRawData();
		results.getEuclidianDistance();
		return results.distanceMatrix;
	}

	public String toString() { return results == null ? new String("") : results.toString(); }
}

