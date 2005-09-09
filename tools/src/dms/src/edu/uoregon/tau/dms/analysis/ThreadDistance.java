package edu.uoregon.tau.dms.analysis;

import edu.uoregon.tau.dms.dss.*;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class ThreadDistance extends DistanceAnalysis {

	public ThreadDistance (DatabaseAPI session, Trial inTrial, Metric inMetric) {
		super (session, inTrial, inMetric);
	}

	protected void getTotals() {

		// build the query to get total amounts
		// for each node, context, thread, get the total time in all events
		StringBuffer buf = new StringBuffer();
		buf.append("select l.node, l.context, l.thread, sum(l.exclusive) ");
		buf.append("from interval_event e ");
		buf.append("left outer join interval_location_profile l ");
		buf.append("on e.id = l.interval_event ");
		buf.append("where e.trial = ? and l.metric = ? ");
		buf.append("group by l.node, l.context, l.thread ");
		buf.append("order by l.node, l.context, l.thread ");

		// hit the database, and get the totals for each thread
		try {
			PreparedStatement statement = db.prepareStatement(buf.toString());
			statement.setInt(1, trial.getID());
			statement.setInt(2, metric.getID());
			ResultSet resultSet = statement.executeQuery();          
			int i = 0;
			while (resultSet.next() != false) {
				results.total[i++] = resultSet.getDouble(4);
        	}
        	resultSet.close(); 
		} catch (SQLException e) {
			e.printStackTrace();
			return;
		}
	}

	protected void getMatrixData() {
		// build the query to get the raw data
		StringBuffer buf = new StringBuffer();
		int cFactor = Integer.parseInt(trial.getField("threads_per_context"));
		int nFactor = cFactor * Integer.parseInt(trial.getField("contexts_per_node"));
		buf.append("select (l.node * ");
		buf.append(nFactor);
		buf.append(") + (l.context * ");
		buf.append(cFactor);
		buf.append(") + l.thread as idx, e.id, ");
		buf.append("COALESCE(l.exclusive, 0.0) from interval_event e ");
		buf.append("left outer join interval_location_profile l ");
		buf.append("on e.id = l.interval_event ");
		buf.append("where e.trial = ? and l.metric = ? ");
		buf.append("order by e.id, idx ");

		// hit the database, and get the normalized values for each 
		// event on each thread
		try {
			PreparedStatement statement = db.prepareStatement(buf.toString());
			statement.setInt(1, trial.getID());
			statement.setInt(2, metric.getID());
			ResultSet resultSet = statement.executeQuery();          
			int i = 0; // thread index
			int j = 0; // event index
			int lastEvent = 0;
			// do this once, outside the loop, to initialize lastEvent.
			if (resultSet.next() != false) {
				i = resultSet.getInt(1); // get the thread index
				lastEvent = resultSet.getInt(2); // get the first event id
				if (results.total[i] == 0.0)
					results.dataMatrix[i][j] = 0.0;
				else
					results.dataMatrix[i][j] = resultSet.getDouble(3) / results.total[i];
			}
			while (resultSet.next() != false) {
				i = resultSet.getInt(1); // get the thread index
				if (lastEvent != resultSet.getInt(2)) {
					j++; // increment the event index
					lastEvent = resultSet.getInt(2);
				}
				if (results.total[i] == 0.0)
					results.dataMatrix[i][j] = 0.0;
				else
					results.dataMatrix[i][j] = resultSet.getDouble(3) / results.total[i];
        	}
        	resultSet.close(); 
		} catch (SQLException e) {
			e.printStackTrace();
			return;
		}

	}

	public int getThreadCount() { return results == null ? 0 : results.matrixSize; }

	public int getEventCount() { return results == null ? 0 : results.dimensionCount; }

	protected DistanceMatrix createTheMatrix(int threadCount, int eventCount) {
		return new ThreadMatrix(threadCount, eventCount);
	}
}

