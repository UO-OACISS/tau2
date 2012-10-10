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
public class TAUdbCounterValue {
	private TAUdbSession session = null;
	private TAUdbTrial trial = null;
	private TAUdbCounter counter = null;
	private TAUdbTimerCallpath timerCallpath = null;
	private TAUdbThread thread = null;
	private double sampleCount = 0.0;
	private double maximumValue = 0.0;
	private double minimumValue = 0.0;
	private double meanValue = 0.0;
	private double standardDeviation = 0.0;

	/**
	 * 
	 */
	public TAUdbCounterValue(TAUdbSession session, TAUdbTrial trial, TAUdbCounter counter, TAUdbTimerCallpath timerCallpath, TAUdbThread thread, double sampleCount, double maximumValue, double minimumValue, double meanValue, double standardDeviation) {
		this.session = session;
		this.trial = trial;
		this.counter = counter;
		this.timerCallpath = timerCallpath;
		this.thread = thread;
		this.sampleCount = sampleCount;
		this.maximumValue = maximumValue;
		this.minimumValue = minimumValue;
		this.meanValue = meanValue;
		this.standardDeviation = standardDeviation;
	}

	public String toString() {
		StringBuilder b = new StringBuilder();
		b.append("Counter: " + counter);
		b.append(" Thread: " + thread.getThreadRank());
		b.append(" samples: " + this.sampleCount);
		b.append(" mean: " + this.meanValue);
		return b.toString();
	}

	public static List<TAUdbCounterValue> getCounterValues(TAUdbSession session, TAUdbTrial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getCounterValues() != null && trial.getCounterValues().size() > 0) {
			return trial.getCounterValues();
		}
		List<TAUdbCounterValue> counterValues = new ArrayList<TAUdbCounterValue>();
		String query = "select v.counter, v.timer_callpath, v.thread, v.sample_count, v.maximum_value, v.minimum_value, v.mean_value, v.standard_deviation from counter_value v join counter c on v.counter = c.id where c.trial = ?";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getID());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer counterID = results.getInt(1);
				Integer timerCallpathID = results.getInt(2);
				Integer threadID = results.getInt(3);
				double sampleCount = results.getDouble(4);
				double maximumValue = results.getDouble(5);
				double minimumValue = results.getDouble(6);
				double meanValue = results.getDouble(7);
				double standardDeviation = results.getDouble(8);
				TAUdbCounter counter = trial.getCounters().get(counterID);
				TAUdbTimerCallpath timerCallpath = trial.getTimerCallpaths().get(timerCallpathID);
				TAUdbThread thread = trial.getThreads().get(threadID);
				if (thread == null) {
					thread = trial.getDerivedThreads().get(threadID);
				}
				TAUdbCounterValue counterValue = new TAUdbCounterValue (session, trial, counter, timerCallpath, thread, sampleCount, maximumValue, minimumValue, meanValue, standardDeviation);
				counterValues.add(counterValue);
			}
			results.close();
			statement.close();
			trial.setCounterValues(counterValues);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return counterValues;
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
	 * @return the counter
	 */
	public TAUdbCounter getCounter() {
		return counter;
	}

	/**
	 * @param counter the counter to set
	 */
	public void setCounter(TAUdbCounter counter) {
		this.counter = counter;
	}

	/**
	 * @return the timerCallpath
	 */
	public TAUdbTimerCallpath getTimerCallpath() {
		return timerCallpath;
	}

	/**
	 * @param timerCallpath the timerCallpath to set
	 */
	public void setTimerCallpath(TAUdbTimerCallpath timerCallpath) {
		this.timerCallpath = timerCallpath;
	}

	/**
	 * @return the thread
	 */
	public TAUdbThread getThread() {
		return thread;
	}

	/**
	 * @param thread the thread to set
	 */
	public void setThread(TAUdbThread thread) {
		this.thread = thread;
	}

	/**
	 * @return the sampleCount
	 */
	public double getSampleCount() {
		return sampleCount;
	}

	/**
	 * @param sampleCount the sampleCount to set
	 */
	public void setSampleCount(double sampleCount) {
		this.sampleCount = sampleCount;
	}

	/**
	 * @return the maximumValue
	 */
	public double getMaximumValue() {
		return maximumValue;
	}

	/**
	 * @param maximumValue the maximumValue to set
	 */
	public void setMaximumValue(double maximumValue) {
		this.maximumValue = maximumValue;
	}

	/**
	 * @return the minimumValue
	 */
	public double getMinimumValue() {
		return minimumValue;
	}

	/**
	 * @param minimumValue the minimumValue to set
	 */
	public void setMinimumValue(double minimumValue) {
		this.minimumValue = minimumValue;
	}

	/**
	 * @return the meanValue
	 */
	public double getMeanValue() {
		return meanValue;
	}

	/**
	 * @param meanValue the meanValue to set
	 */
	public void setMeanValue(double meanValue) {
		this.meanValue = meanValue;
	}

	/**
	 * @return the standardDeviation
	 */
	public double getStandardDeviation() {
		return standardDeviation;
	}

	/**
	 * @param standardDeviation the standardDeviation to set
	 */
	public void setStandardDeviation(double standardDeviation) {
		this.standardDeviation = standardDeviation;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TAUdbSession session = new TAUdbSession("callpath", false);
		TAUdbTrial trial = TAUdbTrial.getTrial(session, 1, true);
		List<TAUdbCounterValue> counterValues = TAUdbCounterValue.getCounterValues(session, trial);
		for (TAUdbCounterValue val : counterValues) {
			System.out.println(val.toString());
		}
		session.close();
	}

}
