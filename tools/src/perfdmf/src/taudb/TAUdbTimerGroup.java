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

/**
 * @author khuck
 *
 */
public class TAUdbTimerGroup {
	String name = null;
	List<TAUdbTimer> timers = null;

	/**
	 * 
	 */
	public TAUdbTimerGroup(String name, TAUdbTimer timer) {
		this.name = name;
		this.timers = new ArrayList<TAUdbTimer>();
		timers.add(timer);
	}
	
	public void addTimer(TAUdbTimer timer) {
		this.timers.add(timer);
	}
	
	public String toString() {
		return this.name;
	}

	public static Map<String, TAUdbTimerGroup> getTimerGroups(TAUdbSession session, TAUdbTrial trial, Map<Integer, TAUdbTimer> timers) {
		Map<String, TAUdbTimerGroup> groups = new HashMap<String, TAUdbTimerGroup>();
		String query = "select tg.group_name, tg.timer, t.trial from timer_group tg join timer t on tg.timer = t.id where t.trial = ?;";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getID());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				String name = results.getString(1);
				Integer timerID = results.getInt(2);
				TAUdbTimer timer = timers.get(timerID);
				TAUdbTimerGroup group = groups.get(name);
				if (group == null) {
					group = new TAUdbTimerGroup(name, timer);
				} else {
					group.addTimer(timer);
				}
				timer.addGroup(group);
				trial.addTimerGroup(group);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return groups;
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
