/**
 * 
 */
package edu.uoregon.tau.perfdmf.taudb;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author khuck
 *
 */
public class TAUdbTimerValue {
	private TAUdbSession session = null;
	private TAUdbTrial trial = null;
	private TAUdbTimerCall timerCall = null;
	private TAUdbMetric metric = null;
	private double inclusive = 0.0;
	private double exclusive = 0.0;
	private double inclusivePercent = 0.0;
	private double exclusivePercent = 0.0;
	private double sumExclusiveSquared = 0.0;

	/**
	 * 
	 */
	public TAUdbTimerValue(TAUdbSession session, TAUdbTrial trial, TAUdbTimerCall timerCall, TAUdbMetric metric, double inclusive, double exclusive, double inclusivePercent, double exclusivePercent, double sumExclusiveSquared) {
		this.session = session;
		this.trial = trial;
		this.timerCall = timerCall;
		this.metric = metric;
		this.inclusive = inclusive;
		this.inclusivePercent = inclusivePercent;
		this.exclusive = exclusive;
		this.exclusivePercent = exclusivePercent;
		this.sumExclusiveSquared = sumExclusiveSquared;
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
	 * @return the timerCall
	 */
	public TAUdbTimerCall getTimerCall() {
		return timerCall;
	}

	/**
	 * @param timerCall the timerCall to set
	 */
	public void setTimerCall(TAUdbTimerCall timerCall) {
		this.timerCall = timerCall;
	}

	/**
	 * @return the metric
	 */
	public TAUdbMetric getMetric() {
		return metric;
	}

	/**
	 * @param metric the metric to set
	 */
	public void setMetric(TAUdbMetric metric) {
		this.metric = metric;
	}

	/**
	 * @return the inclusive
	 */
	public double getInclusive() {
		return inclusive;
	}

	/**
	 * @param inclusive the inclusive to set
	 */
	public void setInclusive(double inclusive) {
		this.inclusive = inclusive;
	}

	/**
	 * @return the exclusive
	 */
	public double getExclusive() {
		return exclusive;
	}

	/**
	 * @param exclusive the exclusive to set
	 */
	public void setExclusive(double exclusive) {
		this.exclusive = exclusive;
	}

	/**
	 * @return the inclusivePercent
	 */
	public double getInclusivePercent() {
		return inclusivePercent;
	}

	/**
	 * @param inclusivePercent the inclusivePercent to set
	 */
	public void setInclusivePercent(double inclusivePercent) {
		this.inclusivePercent = inclusivePercent;
	}

	/**
	 * @return the exclusivePercent
	 */
	public double getExclusivePercent() {
		return exclusivePercent;
	}

	/**
	 * @param exclusivePercent the exclusivePercent to set
	 */
	public void setExclusivePercent(double exclusivePercent) {
		this.exclusivePercent = exclusivePercent;
	}

	/**
	 * @return the sumExclusiveSquared
	 */
	public double getSumExclusiveSquared() {
		return sumExclusiveSquared;
	}

	/**
	 * @param sumExclusiveSquared the sumExclusiveSquared to set
	 */
	public void setSumExclusiveSquared(double sumExclusiveSquared) {
		this.sumExclusiveSquared = sumExclusiveSquared;
	}

	public String toString() {
		StringBuilder b = new StringBuilder();
		b.append("Callpath: " + timerCall + " ");
		b.append(metric.getName() + ": " + inclusive + ", " + exclusive);
		return b.toString();
	}
	
	public static List<TAUdbTimerValue> getTimerValues(TAUdbSession session, TAUdbTrial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getTimerValues() != null && trial.getTimerValues().size() > 0) {
			return trial.getTimerValues();
		}
		List<TAUdbTimerValue> timerValues = new ArrayList<TAUdbTimerValue>();
		String query = "select v.timer_call_data, v.metric, v.inclusive_value, v.exclusive_value, v.inclusive_percent, v.exclusive_percent, v.sum_exclusive_squared from timer_value v join timer_call_data td on v.timer_call_data = td.id join thread t on td.thread = t.id where t.trial = ?";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getID());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer timerCallID = results.getInt(1);
				Integer metricID = results.getInt(2);
				double inclusive = results.getDouble(3);
				double exclusive = results.getDouble(4);
				double inclusivePercent = results.getDouble(5);
				double exclusivePercent = results.getDouble(6);
				double sumExclusiveSquared = results.getDouble(7);
				TAUdbTimerCall timerCall = trial.getTimerCalls().get(timerCallID);
				TAUdbMetric metric = trial.getTAUdbMetrics().get(metricID);
				TAUdbTimerValue timerValue = new TAUdbTimerValue (session, trial, timerCall, metric, inclusive, exclusive, inclusivePercent, exclusivePercent, sumExclusiveSquared);
				timerValues.add(timerValue);
			}
			results.close();
			statement.close();
			trial.setTimerValues(timerValues);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return timerValues;
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TAUdbSession session = new TAUdbSession("callpath", false);
		TAUdbTrial trial = TAUdbTrial.getTrial(session, 1, true);
		List<TAUdbTimerValue> timerValues = TAUdbTimerValue.getTimerValues(session, trial);
		for (TAUdbTimerValue timerValue : timerValues) {
			System.out.println(timerValue.toString());
		}
		session.close();
	}

}
