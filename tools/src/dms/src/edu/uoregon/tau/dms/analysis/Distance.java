package edu.uoregon.tau.dms.analysis;

import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.database.*;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Vector;
import java.util.ListIterator;

public class Distance {
	private DB db = null;
	private EventMatrix results = null;
	private Trial trial = null;
	private Metric metric = null;

	public Distance (PerfDMFSession session, Trial inTrial, Metric inMetric) {
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
		// System.out.println(threadCount + " " + eventCount);

		// initialize the matrix
		results = new EventMatrix(threadCount, eventCount);

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
		buf.append("select l.node, l.context, l.thread, sum(l.exclusive) from interval_location_profile l ");
		buf.append("inner join interval_event e on l.interval_event = e.id ");
		buf.append("where e.trial = ? and l.metric = ? ");
		buf.append("group by l.node, l.context, l.thread ");

		// hit the database, and get the totals for each thread
		try {
			PreparedStatement statement = db.prepareStatement(buf.toString());
			statement.setInt(1, trial.getID());
			statement.setInt(2, metric.getID());
			ResultSet resultSet = statement.executeQuery();          
			int i = 0;
			while (resultSet.next() != false) {
				results.threadTotal[i++] = resultSet.getDouble(4);
				// System.out.println(resultSet.getInt(1) + ":" + resultSet.getInt(2) + ":" + resultSet.getInt(3) + ":" + results.threadTotal[i-1]);
        	}
        	resultSet.close(); 
		} catch (SQLException e) {
			e.printStackTrace();
			return;
		}

		// build the query to get the raw data & percentages
		buf = new StringBuffer();
		int cFactor = trial.getNumThreadsPerContext();
		int nFactor = cFactor * trial.getNumContextsPerNode();
		buf.append("select (l.node * ");
		buf.append(nFactor);
		buf.append(") + (l.context * ");
		buf.append(cFactor);
		buf.append(") + l.thread, e.name, COALESCE(l.exclusive, 0.0) from interval_event e ");
		buf.append("left outer join interval_location_profile l on e.id = l.interval_event ");
		buf.append("where e.trial = ? and l.metric = ? order by e.id ");

		// hit the database, and get the normalized values for each event on each thread
		try {
			PreparedStatement statement = db.prepareStatement(buf.toString());
			statement.setInt(1, trial.getID());
			statement.setInt(2, metric.getID());
			ResultSet resultSet = statement.executeQuery();          
			int i = 0;
			int j = 0;
			while (resultSet.next() != false) {
				j = resultSet.getInt(1);
				results.threadMatrix[j][i++] = resultSet.getDouble(3) / results.threadTotal[j];
				i = i == eventCount ? 0 : i;
        	}
        	resultSet.close(); 
		} catch (SQLException e) {
			e.printStackTrace();
			return;
		}

	}

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

	public int getThreadCount() { return results == null ? 0 : results.threadCount; }

	public int getEventCount() { return results == null ? 0 : results.eventCount; }
}

