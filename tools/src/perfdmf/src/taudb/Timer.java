/**
 * 
 */
package edu.uoregon.tau.perfdmf.taudb;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * @author khuck
 *
 */
public class Timer {
	private Session session = null;
	private int id = 0;
	private Trial trial = null;
	private String name = null;
	private String shortName = null;
	private String sourceFile = null;
	private int lineNumber = 0;
	private int lineNumberEnd = 0;
	private int columnNumber = 0;
	private int columnNumberEnd = 0;
	private Set<TimerGroup> groups;
	private Set<TimerParameter> parameters;

	/**
	 * 
	 */
	public Timer() {
		super();
	}
	
	public Timer(Session session, int id, Trial trial, String name, String shortName, String sourceFile, int lineNumber, int lineNumberEnd, int columnNumber, int columnNumberEnd) {
		this.session = session;
		this.id = id;
		this.trial = trial;
		this.name = name;
		this.shortName = shortName;
		this.sourceFile = sourceFile;
		this.lineNumber = lineNumber;
		this.lineNumberEnd = lineNumberEnd;
		this.columnNumber = columnNumber;
		this.columnNumberEnd = columnNumberEnd;
		this.groups = new HashSet<TimerGroup>();
		this.parameters = new HashSet<TimerParameter>();
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
	 * @return the shortName
	 */
	public String getShortName() {
		return shortName;
	}

	/**
	 * @param shortName the shortName to set
	 */
	public void setShortName(String shortName) {
		this.shortName = shortName;
	}

	/**
	 * @return the sourceFile
	 */
	public String getSourceFile() {
		return sourceFile;
	}

	/**
	 * @param sourceFile the sourceFile to set
	 */
	public void setSourceFile(String sourceFile) {
		this.sourceFile = sourceFile;
	}

	/**
	 * @return the lineNumber
	 */
	public int getLineNumber() {
		return lineNumber;
	}

	/**
	 * @param lineNumber the lineNumber to set
	 */
	public void setLineNumber(int lineNumber) {
		this.lineNumber = lineNumber;
	}

	/**
	 * @return the lineNumberEnd
	 */
	public int getLineNumberEnd() {
		return lineNumberEnd;
	}

	/**
	 * @param lineNumberEnd the lineNumberEnd to set
	 */
	public void setLineNumberEnd(int lineNumberEnd) {
		this.lineNumberEnd = lineNumberEnd;
	}

	/**
	 * @return the columnNumber
	 */
	public int getColumnNumber() {
		return columnNumber;
	}

	/**
	 * @param columnNumber the columnNumber to set
	 */
	public void setColumnNumber(int columnNumber) {
		this.columnNumber = columnNumber;
	}

	/**
	 * @return the columnNumberEnd
	 */
	public int getColumnNumberEnd() {
		return columnNumberEnd;
	}

	/**
	 * @param columnNumberEnd the columnNumberEnd to set
	 */
	public void setColumnNumberEnd(int columnNumberEnd) {
		this.columnNumberEnd = columnNumberEnd;
	}
	
	public void addGroup(TimerGroup group) {
		this.groups.add(group);
	}
	
	public void addParameter(TimerParameter parameter) {
		this.parameters.add(parameter);
	}

	public String toString() {
		StringBuilder b = new StringBuilder();
		b.append("name:" + name + ",");
		for (TimerGroup group : groups) {
			b.append(group.toString());
		}
		for (TimerParameter parameter : parameters) {
			b.append(parameter.toString());
		}
		return b.toString();
	}

	public static Timer getTimer(Session session, Trial trial, int id) {
		Timer timer = null;
		String query = "select name, short_name, source_file, line_number, line_number_end, column_number, column_number_end from timer where id = ?;";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, id);
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				String name = results.getString(1);
				String shortName = results.getString(2);
				String sourceFile = results.getString(3);
				int lineNumber = results.getInt(4);
				int lineNumberEnd = results.getInt(5);
				int columnNumber = results.getInt(6);
				int columnNumberEnd = results.getInt(7);
				timer = new Timer (session, id, trial, name, shortName, sourceFile, lineNumber, lineNumberEnd, columnNumber, columnNumberEnd);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return timer;
	}

	public static Map<Integer, Timer> getTimers(Session session, Trial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getTimers() != null && trial.getTimers().size() > 0) {
			return trial.getTimers();
		}
		Map<Integer, Timer> timers = new HashMap<Integer, Timer>();
		String query = "select id, name, short_name, source_file, line_number, line_number_end, column_number, column_number_end from timer where trial = ?;";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getId());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer id = results.getInt(1);
				String name = results.getString(2);
				String shortName = results.getString(3);
				String sourceFile = results.getString(4);
				int lineNumber = results.getInt(5);
				int lineNumberEnd = results.getInt(6);
				int columnNumber = results.getInt(7);
				int columnNumberEnd = results.getInt(8);
				Timer timer = new Timer (session, id, trial, name, shortName, sourceFile, lineNumber, lineNumberEnd, columnNumber, columnNumberEnd);
				timers.put(id, timer);
			}
			results.close();
			statement.close();
			trial.setTimers(timers);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		TimerGroup.getTimerGroups(session, trial, timers);
		return timers;
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		Session session = new Session("callpath", false);
		Trial trial = Trial.getTrial(session, 1, false);
		Map<Integer, Timer> timers = Timer.getTimers(session, trial);
		for (Integer id : timers.keySet()) {
			Timer timer = timers.get(id);
			System.out.println(timer.toString());
		}
		session.close();
	}

}
