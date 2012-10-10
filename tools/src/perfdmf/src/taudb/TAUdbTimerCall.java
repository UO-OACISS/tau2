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
public class TAUdbTimerCall {
	private TAUdbSession session = null;
	private TAUdbTrial trial = null;
	private int id = 0;
	private TAUdbTimerCallpath timerCallpath = null;
	private TAUdbThread thread = null;
	private double calls = 0.0;
	private double subroutines = 0.0;
	private double timestamp = 0.0;

	/**
	 * 
	 */
	public TAUdbTimerCall(TAUdbSession session, TAUdbTrial trial, int id, TAUdbTimerCallpath timerCallpath, TAUdbThread thread, double calls, double subroutines, double timestamp) {
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
	public static Map<Integer, TAUdbTimerCall> getTimerCalls(TAUdbSession session, TAUdbTrial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getTimerCalls() != null && trial.getTimerCalls().size() > 0) {
			return trial.getTimerCalls();
		}
		Map<Integer, TAUdbTimerCall> timerCalls = new HashMap<Integer, TAUdbTimerCall>();
		String query = "select tc.id, tc.timer_callpath, tc.thread, tc.calls, tc.subroutines, tc.timestamp from timer_call_data tc join thread t on tc.thread = t.id where t.trial = ?";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getID());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				Integer timerCallpathID = results.getInt(2);
				Integer threadID = results.getInt(3);
				double calls = results.getDouble(4);
				double subroutines = results.getDouble(5);
				double timestamp = results.getDouble(6);
				TAUdbTimerCallpath timerCallpath = trial.getTimerCallpaths().get(timerCallpathID);
				TAUdbThread thread = trial.getThreads().get(threadID);
				if (thread == null) {
					thread = trial.getDerivedThreads().get(threadID);
				}
				TAUdbTimerCall timerCallData = new TAUdbTimerCall (session, trial, id, timerCallpath, thread, calls, subroutines, timestamp);
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
		TAUdbSession session = new TAUdbSession("callpath", false);
		TAUdbTrial trial = TAUdbTrial.getTrial(session, 1, true);
		Map<Integer, TAUdbTimerCall> timerCalls = TAUdbTimerCall.getTimerCalls(session, trial);
		for (Integer cp : timerCalls.keySet()) {
			System.out.println(timerCalls.get(cp).toString());
		}
		session.close();
	}

}
