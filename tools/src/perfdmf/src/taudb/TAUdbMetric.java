/**
 * 
 */
package edu.uoregon.tau.perfdmf.taudb;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

import edu.uoregon.tau.perfdmf.Metric;

/**
 * @author khuck
 *
 */
public class TAUdbMetric extends Metric{
	//private int id = 0;
	private TAUdbTrial trial = null;
	private TAUdbSession session = null;

	/**
	 * 
	 */
	public TAUdbMetric(TAUdbSession session, int id, TAUdbTrial trial, String name, boolean derived) {
		super();
		this.metricID = id;
		this.dbMetricID = id;
		this.trialID = trial.getID();
		this.trial = trial;
		this.session = session;
		this.name = name;
		this.derivedMetric = derived;
	}

	public String toString() {
		return this.name;
	}

	

	/**
	 * @return the trial
	 */
	public TAUdbTrial getTrial() {
		return trial;
	}

	/**
	 * @param trial the trial to set
	 */
	public void setTrial(TAUdbTrial trial) {
		this.trial = trial;
	}

	/**
	 * @return the session
	 */
	public TAUdbSession getSession() {
		return session;
	}

	/**
	 * @param session the session to set
	 */
	public void setSession(TAUdbSession session) {
		this.session = session;
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @param name the name to set
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * @return the derived
	 */
	public boolean isDerived() {
		return derivedMetric;
	}

	/**
	 * @param derived the derived to set
	 */
	public void setDerived(boolean derived) {
		this.derivedMetric = derived;
	}

	public static Map<Integer, TAUdbMetric> getMetrics(TAUdbSession session, TAUdbTrial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getMetrics() != null && trial.getMetrics().size() > 0) {
			return trial.getTAUdbMetrics();
		}
		Map<Integer, TAUdbMetric> metrics = new HashMap<Integer, TAUdbMetric>();
		String query = "select id, name, derived from metric where trial = ?";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getID());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				String name = results.getString(2);
				boolean derived = results.getBoolean(3);
				TAUdbMetric metric = new TAUdbMetric (session, id, trial, name, derived);
				metrics.put(id, metric);
			}
			results.close();
			statement.close();
			trial.setMetrics(metrics);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return metrics;
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TAUdbSession session = new TAUdbSession("callpath", false);
		TAUdbTrial trial = TAUdbTrial.getTrial(session, 1, false);
		Map<Integer, TAUdbMetric> metrics = TAUdbMetric.getMetrics(session, trial);
		for (Integer id : metrics.keySet()) {
			TAUdbMetric metric = metrics.get(id);
			System.out.println(metric.toString());
		}
		session.close();
	}

}
