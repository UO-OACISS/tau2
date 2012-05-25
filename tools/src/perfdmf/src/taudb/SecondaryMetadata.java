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
public class SecondaryMetadata {
	private Session session = null;
	private Trial trial = null;
	private int id = 0;
	private TimerCallpath timerCallpath = null;
	private Thread thread = null;
	private SecondaryMetadata parent = null;
	private List<SecondaryMetadata> children = null;
	private double timestamp = 0.0;
	private String name = null;
	private String value = null;
	private boolean isArray = false;

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
	 * @return the parent
	 */
	public SecondaryMetadata getParent() {
		return parent;
	}

	/**
	 * @param parent the parent to set
	 */
	public void setParent(SecondaryMetadata parent) {
		this.parent = parent;
	}

	/**
	 * @return the children
	 */
	public List<SecondaryMetadata> getChildren() {
		return children;
	}

	/**
	 * @param children the children to set
	 */
	public void setChildren(List<SecondaryMetadata> children) {
		this.children = children;
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
	 * @return the value
	 */
	public String getValue() {
		return value;
	}

	/**
	 * @param value the value to set
	 */
	public void setValue(String value) {
		this.value = value;
	}

	/**
	 * @return the isArray
	 */
	public boolean isArray() {
		return isArray;
	}

	/**
	 * @param isArray the isArray to set
	 */
	public void setArray(boolean isArray) {
		this.isArray = isArray;
	}

	/**
	 * 
	 */
	public SecondaryMetadata(Session session, Trial trial, int id, TimerCallpath timerCallpath, Thread thread, SecondaryMetadata parent, String name, String value, boolean isArray) {
		this.session = session;
		this.trial = trial;
		this.id = id;
		this.timerCallpath = timerCallpath;
		this.thread = thread;
		this.parent = parent;
		this.children = new ArrayList<SecondaryMetadata>();
		this.name = name;
		this.value = value;
		this.isArray = isArray;
	}

	public static Map<Integer, SecondaryMetadata> getSecondaryMetadata(Session session, Trial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getSecondaryMetadata() != null && trial.getSecondaryMetadata().size() > 0) {
			return trial.getSecondaryMetadata();
		}
		Map<Integer, SecondaryMetadata> secondaryMetadata = new HashMap<Integer, SecondaryMetadata>();
		String query = "select id, thread, timer_callpath, parent, name, value, is_array from secondary_metadata where trial = ?";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getId());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				Integer threadID = results.getInt(2);
				Integer timerCallpathID = results.getInt(3);
				Integer parentID = results.getInt(4);
				String name = results.getString(5);
				String value = results.getString(6);
				boolean isArray = results.getBoolean(7);
				TimerCallpath timerCallpath = trial.getTimerCallpaths().get(timerCallpathID);
				Thread thread = trial.getThreads().get(threadID);
				SecondaryMetadata parent = null;
				if (parentID != null) {
					parent = secondaryMetadata.get(parentID);
				}
				SecondaryMetadata secondaryMetadatum = new SecondaryMetadata (session, trial, id, timerCallpath, thread, parent, name, value, isArray);
				secondaryMetadata.put(id, secondaryMetadatum);
			}
			results.close();
			statement.close();
			trial.setSecondaryMetadata(secondaryMetadata);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return secondaryMetadata;
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Session session = new Session("callpath", false);
		Trial trial = Trial.getTrial(session, 1, true);
		Map<Integer, SecondaryMetadata> secondaryMetadata = SecondaryMetadata.getSecondaryMetadata(session, trial);
		for (Integer cp : secondaryMetadata.keySet()) {
			System.out.println(secondaryMetadata.get(cp).toString());
		}
		session.close();
	}

}
