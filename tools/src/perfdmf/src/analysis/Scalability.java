package edu.uoregon.tau.perfdmf.analysis;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ListIterator;
import java.util.Vector;

import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.database.DB;

public class Scalability {
	private DB db = null;

	public Scalability (DatabaseAPI session) {
		this.db = session.db();
	}

	public ScalabilityResults exclusive(Vector<Trial> inTrials, String function) {
		return trials(inTrials, "exclusive", function);
	}

	public ScalabilityResults exclusivePercentage(Vector<Trial> inTrials, String function) {
		return trials(inTrials, "exclusive_percentage", function);
	}

	public ScalabilityResults inclusive(Vector<Trial> inTrials, String function) {
		return trials(inTrials, "inclusive", function);
	}

	public ScalabilityResults inclusivePercentage(Vector<Trial> inTrials, String function) {
		return trials(inTrials, "inclusive_percentage", function);
	}

	public ScalabilityResults trials (Vector<Trial> inTrials, String measurement, String function) {
		ScalabilityResults results = new ScalabilityResults();
		ListIterator<Trial> trials = inTrials.listIterator();

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
			Trial trial = trials.next();
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

		return results;
	}

}

