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
public class TAUdbCounter {
	private TAUdbSession session = null;
	private int id = 0;
	private TAUdbTrial trial = null;
	private String name = null;

	/**
	 * 
	 */
	public TAUdbCounter() {
		super();
	}
	
	public TAUdbCounter(TAUdbSession session, int id, TAUdbTrial trial, String name) {
		this.session = session;
		this.id = id;
		this.trial = trial;
		this.name = name;
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
		StringBuilder b = new StringBuilder();
		b.append("name:" + name + ",");
		return b.toString();
	}

	public static TAUdbCounter getCounter(TAUdbSession session, TAUdbTrial trial, int id) {
		TAUdbCounter counter = null;
		String query = "select name from counter where id = ?;";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, id);
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				String name = results.getString(1);
				counter = new TAUdbCounter (session, id, trial, name);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return counter;
	}

	public static Map<Integer, TAUdbCounter> getCounters(TAUdbSession session, TAUdbTrial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getCounters() != null && trial.getCounters().size() > 0) {
			return trial.getCounters();
		}
		Map<Integer, TAUdbCounter> counters = new HashMap<Integer, TAUdbCounter>();
		String query = "select id, name from counter where trial = ?;";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getID());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				String name = results.getString(2);
				TAUdbCounter counter = new TAUdbCounter (session, id, trial, name);
				counters.put(id, counter);
			}
			results.close();
			statement.close();
			trial.setCounters(counters);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return counters;
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TAUdbSession session = new TAUdbSession("callpath", false);
		TAUdbTrial trial = TAUdbTrial.getTrial(session, 1, true);
		Map<Integer, TAUdbCounter> counters = TAUdbCounter.getCounters(session, trial);
		for (Integer id : counters.keySet()) {
			TAUdbCounter counter = counters.get(id);
			System.out.println(counter.toString());
		}
		session.close();
	}

}
