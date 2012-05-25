/**
 * 
 */
package edu.uoregon.tau.perfdmf.taudb;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

/**
 * @author khuck
 *
 */
public class TimerCall {
	private Session session = null;
	private Trial trial = null;
	private int id = 0;
	private TimerCallpath timerCallpath = null;
	private Thread thread = null;
	private double calls = 0.0;
	private double subroutines = 0.0;
	private double timestamp = 0.0;

	/**
	 * 
	 */
	public TimerCall(Session session, Trial trial, int id, TimerCallpath timerCallpath, Thread thread, double calls, double subroutines, double timestamp) {
		this.session = session;
		this.trial = trial;
		this.id = id;
		this.timerCallpath = timerCallpath;
		this.thread = thread;
		this.calls = calls;
		this.subroutines = subroutines;
		this.timestamp = timestamp;
	}

	/**
	 * @return the session
	 */
	public Session getSession() {
		return session;
	}

	/**
	 * @param session the session to set
	 */
	public void setSession(Session session) {
		this.session = session;
	}

	/**
	 * @return the trial
	 */
	public Trial getTrial() {
		return trial;
	}

	/**
	 * @param trial the trial to set
	 */
	public void setTrial(Trial trial) {
		this.trial = trial;
	}

	/**
	 * @return the id
	 */
	public int getId() {
		return id;
	}

	/**
	 * @param id the id to set
	 */
	public void setId(int id) {
		this.id = id;
	}

	/**
	 * @return the timerCallpath
	 */
	public TimerCallpath getTimerCallpath() {
		return timerCallpath;
	}

	/**
	 * @param timerCallpath the timerCallpath to set
	 */
	public void setTimerCallpath(TimerCallpath timerCallpath) {
		this.timerCallpath = timerCallpath;
	}

	/**
	 * @return the thread
	 */
	public Thread getThread() {
		return thread;
	}

	/**
	 * @param thread the thread to set
	 */
	public void setThread(Thread thread) {
		this.thread = thread;
	}

	/**
	 * @return the calls
	 */
	public double getCalls() {
		return calls;
	}

	/**
	 * @param calls the calls to set
	 */
	public void setCalls(double calls) {
		this.calls = calls;
	}

	/**
	 * @return the subroutines
	 */
	public double getSubroutines() {
		return subroutines;
	}

	/**
	 * @param subroutines the subroutines to set
	 */
	public void setSubroutines(double subroutines) {
		this.subroutines = subroutines;
	}

	/**
	 * @return the timestamp
	 */
	public double getTimestamp() {
		return timestamp;
	}

	/**
	 * @param timestamp the timestamp to set
	 */
	public void setTimestamp(double timestamp) {
		this.timestamp = timestamp;
	}

	public String toString() {
		StringBuilder b = new StringBuilder();
		b.append("Callpath: " + timerCallpath);
		b.append(" Thread: " + thread.getThreadRank());
		b.append(" Calls: " + this.calls);
		b.append(" Subroutines: " + this.subroutines);
		return b.toString();
	}
	public static Map<Integer, TimerCall> getTimerCalls(Session session, Trial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getTimerCalls() != null && trial.getTimerCalls().size() > 0) {
			return trial.getTimerCalls();
		}
		Map<Integer, TimerCall> timerCalls = new HashMap<Integer, TimerCall>();
		String query = "select tc.id, tc.timer_callpath, tc.thread, tc.calls, tc.subroutines, tc.timestamp from timer_call_data tc join thread t on tc.thread = t.id where t.trial = ?";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getId());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				Integer timerCallpathID = results.getInt(2);
				Integer threadID = results.getInt(3);
				double calls = results.getDouble(4);
				double subroutines = results.getDouble(5);
				double timestamp = results.getDouble(6);
				TimerCallpath timerCallpath = trial.getTimerCallpaths().get(timerCallpathID);
				Thread thread = trial.getThreads().get(threadID);
				if (thread == null) {
					thread = trial.getDerivedThreads().get(threadID);
				}
				TimerCall timerCallData = new TimerCall (session, trial, id, timerCallpath, thread, calls, subroutines, timestamp);
				timerCalls.put(id, timerCallData);
			}
			results.close();
			statement.close();
			trial.setTimerCalls(timerCalls);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return timerCalls;
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Session session = new Session("callpath", false);
		Trial trial = Trial.getTrial(session, 1, true);
		Map<Integer, TimerCall> timerCalls = TimerCall.getTimerCalls(session, trial);
		for (Integer cp : timerCalls.keySet()) {
			System.out.println(timerCalls.get(cp).toString());
		}
		session.close();
	}

}
