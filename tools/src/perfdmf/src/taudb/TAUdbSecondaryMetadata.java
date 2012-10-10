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
public class TAUdbSecondaryMetadata {
	private TAUdbSession session = null;
	private TAUdbTrial trial = null;
	private int id = 0;
	private TAUdbTimerCallpath timerCallpath = null;
	private TAUdbThread thread = null;
	private TAUdbSecondaryMetadata parent = null;
	private List<TAUdbSecondaryMetadata> children = null;
	private double timestamp = 0.0;
	private String name = null;
	private String value = null;
	private boolean isArray = false;

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
	 * @return the parent
	 */
	public TAUdbSecondaryMetadata getParent() {
		return parent;
	}

	/**
	 * @param parent the parent to set
	 */
	public void setParent(TAUdbSecondaryMetadata parent) {
		this.parent = parent;
	}

	/**
	 * @return the children
	 */
	public List<TAUdbSecondaryMetadata> getChildren() {
		return children;
	}

	/**
	 * @param children the children to set
	 */
	public void setChildren(List<TAUdbSecondaryMetadata> children) {
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
	public TAUdbSecondaryMetadata(TAUdbSession session, TAUdbTrial trial, int id, TAUdbTimerCallpath timerCallpath, TAUdbThread thread, TAUdbSecondaryMetadata parent, String name, String value, boolean isArray) {
		this.session = session;
		this.trial = trial;
		this.id = id;
		this.timerCallpath = timerCallpath;
		this.thread = thread;
		this.parent = parent;
		this.children = new ArrayList<TAUdbSecondaryMetadata>();
		this.name = name;
		this.value = value;
		this.isArray = isArray;
	}

	public static Map<Integer, TAUdbSecondaryMetadata> getSecondaryMetadata(TAUdbSession session, TAUdbTrial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getSecondaryMetadata() != null && trial.getSecondaryMetadata().size() > 0) {
			return trial.getSecondaryMetadata();
		}
		Map<Integer, TAUdbSecondaryMetadata> secondaryMetadata = new HashMap<Integer, TAUdbSecondaryMetadata>();
		String query = "select id, thread, timer_callpath, parent, name, value, is_array from secondary_metadata where trial = ?";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getID());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				Integer threadID = results.getInt(2);
				Integer timerCallpathID = results.getInt(3);
				Integer parentID = results.getInt(4);
				String name = results.getString(5);
				String value = results.getString(6);
				boolean isArray = results.getBoolean(7);
				TAUdbTimerCallpath timerCallpath = trial.getTimerCallpaths().get(timerCallpathID);
				TAUdbThread thread = trial.getThreads().get(threadID);
				TAUdbSecondaryMetadata parent = null;
				if (parentID != null) {
					parent = secondaryMetadata.get(parentID);
				}
				TAUdbSecondaryMetadata secondaryMetadatum = new TAUdbSecondaryMetadata (session, trial, id, timerCallpath, thread, parent, name, value, isArray);
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
		TAUdbSession session = new TAUdbSession("callpath", false);
		TAUdbTrial trial = TAUdbTrial.getTrial(session, 1, true);
		Map<Integer, TAUdbSecondaryMetadata> secondaryMetadata = TAUdbSecondaryMetadata.getSecondaryMetadata(session, trial);
		for (Integer cp : secondaryMetadata.keySet()) {
			System.out.println(secondaryMetadata.get(cp).toString());
		}
		session.close();
	}

}
