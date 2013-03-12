/**
 * 
 */
package edu.uoregon.tau.perfdmf.taudb;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.uoregon.tau.perfdmf.database.DB;

/**
 * @author khuck
 *
 */
public class TAUdbTimerCallpath {
	private int id = 0;
	private TAUdbTimer timer = null;
	private TAUdbTimerCallpath parent = null;
	private List<TAUdbTimerCallpath> children = null;
	private String name = null;

	/**
	 * 
	 */
	public TAUdbTimerCallpath(TAUdbSession session, int id, TAUdbTimer timer, TAUdbTimerCallpath parent, String name) {
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
	public TAUdbTimer getTimer() {
		return timer;
	}

	/**
	 * @param timer the timer to set
	 */
	public void setTimer(TAUdbTimer timer) {
		this.timer = timer;
	}

	/**
	 * @return the parent
	 */
	public TAUdbTimerCallpath getParent() {
		return parent;
	}

	/**
	 * @param parent the parent to set
	 */
	public void setParent(TAUdbTimerCallpath parent) {
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

	/**
	 * @return the children
	 */
	public List<TAUdbTimerCallpath> getChildren() {
		return children;
	}

	public void addChild(TAUdbTimerCallpath child) {
		if (this.children == null)
			this.children = new ArrayList<TAUdbTimerCallpath>();
		this.children.add(child);
	}

	public String toString() {
		return this.name;
	}
	
	public static Map<Integer, TAUdbTimerCallpath> getTimerCallpaths(TAUdbSession session, TAUdbTrial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getTimerCallpaths() != null && trial.getTimerCallpaths().size() > 0) {
			return trial.getTimerCallpaths();
		}
		Map<Integer, TAUdbTimerCallpath> timerCallpaths = new HashMap<Integer, TAUdbTimerCallpath>();
		Map<Integer, TAUdbTimer> timers = trial.getTimers();
		DB db = session.getDB();
		StringBuilder sb = new StringBuilder();
		// for some reason, this fails as a parameterized query. So, put the trial id in explicitly.
		sb.append("with recursive cp (id, parent, timer, name) as ( " +
				"SELECT tc.id, tc.parent, tc.timer, t.name FROM " +
				db.getSchemaPrefix() +
				"timer_callpath tc inner join " +
				db.getSchemaPrefix() +
				"timer t on tc.timer = t.id where " +
				"t.trial = " + trial.getID() + " and tc.parent is null " +
				"UNION ALL " +
				"SELECT d.id, d.parent, d.timer, ");
		        if (db.getDBType().compareTo("h2") == 0) {
					sb.append("concat (cp.name, ' => ', dt.name) FROM ");
		        } else {
					sb.append("cp.name || ' => ' || dt.name FROM ");
		        }
				sb.append(db.getSchemaPrefix() +
				"timer_callpath AS d JOIN cp ON (d.parent = cp.id) join " +
				db.getSchemaPrefix() +
				"timer dt on d.timer = dt.id where dt.trial = " + trial.getID() + " ) " +
				"SELECT distinct cp.id, cp.parent, cp.timer, cp.name FROM cp order by parent ");
		try {
			PreparedStatement statement = session.getDB().prepareStatement(sb.toString());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				Integer parentID = results.getInt(2);
				Integer timerID = results.getInt(3);
				String name = results.getString(4);
				TAUdbTimer timer = timers.get(timerID);
				TAUdbTimerCallpath parent = timerCallpaths.get(parentID);
				TAUdbTimerCallpath timerCallpath = new TAUdbTimerCallpath (session, id, timer, parent, name);
				if (parent != null) 
					parent.addChild(timerCallpath);
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
		TAUdbSession session = new TAUdbSession("callpath", false);
		TAUdbTrial trial = TAUdbTrial.getTrial(session, 1, true);
		Map<Integer, TAUdbTimerCallpath> timerCallpaths = TAUdbTimerCallpath.getTimerCallpaths(session, trial);
		for (Integer cp : timerCallpaths.keySet()) {
			System.out.println(timerCallpaths.get(cp).toString());
		}
		session.close();
	}

}
