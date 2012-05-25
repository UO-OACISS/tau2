/**
 * 
 */
package edu.uoregon.tau.perfdmf.taudb;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

import edu.uoregon.tau.perfdmf.database.DB;

/**
 * @author khuck
 *
 */
public class TimerCallpath {
	private int id = 0;
	private Timer timer = null;
	private TimerCallpath parent = null;
	private String name = null;

	/**
	 * 
	 */
	public TimerCallpath(Session session, int id, Timer timer, TimerCallpath parent, String name) {
		this.id = id;
		this.timer = timer;
		this.parent = parent;
		this.name = name;
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
	 * @return the timer
	 */
	public Timer getTimer() {
		return timer;
	}

	/**
	 * @param timer the timer to set
	 */
	public void setTimer(Timer timer) {
		this.timer = timer;
	}

	/**
	 * @return the parent
	 */
	public TimerCallpath getParent() {
		return parent;
	}

	/**
	 * @param parent the parent to set
	 */
	public void setParent(TimerCallpath parent) {
		this.parent = parent;
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

	public String toString() {
		return this.name;
	}
	
	public static Map<Integer, TimerCallpath> getTimerCallpaths(Session session, Trial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getTimerCallpaths() != null && trial.getTimerCallpaths().size() > 0) {
			return trial.getTimerCallpaths();
		}
		Map<Integer, TimerCallpath> timerCallpaths = new HashMap<Integer, TimerCallpath>();
		Map<Integer, Timer> timers = trial.getTimers();
		DB db = session.getDB();
		StringBuilder sb = new StringBuilder();
		// for some reason, this fails as a parameterized query. So, put the trial id in explicitly.
		sb.append("with recursive cp (id, parent, timer, name) as ( " +
				"SELECT tc.id, tc.parent, tc.timer, t.name FROM " +
				db.getSchemaPrefix() +
				"timer_callpath tc inner join " +
				db.getSchemaPrefix() +
				"timer t on tc.timer = t.id where " +
				"t.trial = " + trial.getId() + " and tc.parent is null " +
				"UNION ALL " +
				"SELECT d.id, d.parent, d.timer, " +
				"concat (cp.name, ' => ', dt.name) FROM " +
				db.getSchemaPrefix() +
				"timer_callpath AS d JOIN cp ON (d.parent = cp.id) join " +
				db.getSchemaPrefix() +
				"timer dt on d.timer = dt.id) " +
				"SELECT distinct cp.id, cp.parent, cp.timer, cp.name FROM cp order by parent ");
		try {
			PreparedStatement statement = session.getDB().prepareStatement(sb.toString());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				Integer parentID = results.getInt(2);
				Integer timerID = results.getInt(3);
				String name = results.getString(4);
				Timer timer = timers.get(timerID);
				TimerCallpath parent = timerCallpaths.get(parentID);
				TimerCallpath timerCallpath = new TimerCallpath (session, id, timer, parent, name);
				timerCallpaths.put(id, timerCallpath);
			}
			results.close();
			statement.close();
			trial.setTimerCallpaths(timerCallpaths);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return timerCallpaths;
	}


	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Session session = new Session("callpath", false);
		Trial trial = Trial.getTrial(session, 1, true);
		Map<Integer, TimerCallpath> timerCallpaths = TimerCallpath.getTimerCallpaths(session, trial);
		for (Integer cp : timerCallpaths.keySet()) {
			System.out.println(timerCallpaths.get(cp).toString());
		}
		session.close();
	}

}
