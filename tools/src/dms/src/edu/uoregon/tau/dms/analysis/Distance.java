package edu.uoregon.tau.dms.analysis;

import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.database.*;
import java.sql.*;
import java.util.Vector;
import java.util.ListIterator;

public class Distance {
	private DB db = null;

	public Distance (PerfDMFSession session) {
		this.db = session.db();
	}

	public double[][] getNodeDistance(Trial inTrial, Metric inMetric) {
		int nodeCount = inTrial.getNodeCount() * inTrial.getNumContextsPerNode() * inTrial.getNumThreadsPerContext();
		// get the event count from the database
		int eventCount = 0;
		StringBuffer buf = new StringBuffer();
		buf.append("select sum(id) from interval_event where trial = ");
		buf.append(inTrial.getID());
		// get the count thingy

		EventMatrix results = new EventMatrix(nodeCount, eventCount);

		// hit the database, and get the matrix of raw data
		/*
		StringBuffer buf = new StringBuffer();
		buf.append("select e.name, e.trial, t.node_count * ");
		buf.append("t.contexts_per_node * t.threads_per_context as threads, ");
		buf.append("min(i.");
		buf.append(measurement);
		buf.append("), avg(i.");
		buf.append(measurement);
		buf.append("), max(i.");
		buf.append(measurement);
		buf.append("), stddev(i.");
		buf.append(measurement);
		buf.append(") from interval_event e inner join interval_location_profile ");
		buf.append("i on e.id = i.interval_event inner join trial t on e.trial = ");
		buf.append("t.id where e.trial in (");

		// loop through the trials, and get their IDs for the select statement
		int i = 0;
		while (trials.hasNext()) {
			Trial trial = (Trial)trials.next();
			if (i++ > 0) {
				buf.append(",");
			}
			buf.append(trial.getID());
		}
		buf.append(") ");
		if (function != null)
			buf.append("and e.name like '" + function + "' ");
		buf.append("group by e.name, e.trial, threads order by ");
		buf.append("e.name, threads, e.trial");

		// System.out.println(buf.toString());
		try {
			ResultSet resultSet = db.executeQuery(buf.toString());          
			String currentFunction = new String("");
			ScalabilityResult current = null;
			int counter = 0;
			while (resultSet.next() != false) {
				if (!currentFunction.equals(resultSet.getString(1))) {
					// create a new function result
					current = new ScalabilityResult(resultSet.getString(1), inTrials.size());
					results.add(current);
					counter = 0;
					currentFunction = resultSet.getString(1);
				} else { counter++; }
				current.threadCount[counter] = resultSet.getInt(3);
				current.minimum[counter] = resultSet.getDouble(4);
				current.average[counter] = resultSet.getDouble(5);
				current.maximum[counter] = resultSet.getDouble(6);
				current.stddev[counter] = resultSet.getDouble(7);
        	}
        	resultSet.close(); 
		} catch (SQLException e) {
			e.printStackTrace();
			return null;
		}
		*/

		return results.distanceMatrix;
	}

}

