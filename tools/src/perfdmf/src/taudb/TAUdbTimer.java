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
public class TAUdbTimer {
	private TAUdbSession session = null;
	private int id = 0;
	private TAUdbTrial trial = null;
	private String name = null;
	private String shortName = null;
	private String sourceFile = null;
	private int lineNumber = 0;
	private int lineNumberEnd = 0;
	private int columnNumber = 0;
	private int columnNumberEnd = 0;
	private Set<TAUdbTimerGroup> groups;
	private Set<TAUdbTimerParameter> parameters;

	/**
	 * 
	 */
	public TAUdbTimer() {
		super();
	}
	
	public TAUdbTimer(TAUdbSession session, int id, TAUdbTrial trial, String name, String shortName, String sourceFile, int lineNumber, int lineNumberEnd, int columnNumber, int columnNumberEnd) {
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
		this.groups = new HashSet<TAUdbTimerGroup>();
		this.parameters = new HashSet<TAUdbTimerParameter>();
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
	
	public void addGroup(TAUdbTimerGroup group) {
		this.groups.add(group);
	}
	
	public void addParameter(TAUdbTimerParameter parameter) {
		this.parameters.add(parameter);
	}

	public String toString() {
		StringBuilder b = new StringBuilder();
		b.append("name:" + name + ",");
		for (TAUdbTimerGroup group : groups) {
			b.append(group.toString());
		}
		for (TAUdbTimerParameter parameter : parameters) {
			b.append(parameter.toString());
		}
		return b.toString();
	}

	public static TAUdbTimer getTimer(TAUdbSession session, TAUdbTrial trial, int id) {
		TAUdbTimer timer = null;
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
				timer = new TAUdbTimer (session, id, trial, name, shortName, sourceFile, lineNumber, lineNumberEnd, columnNumber, columnNumberEnd);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return timer;
	}

	public static Map<Integer, TAUdbTimer> getTimers(TAUdbSession session, TAUdbTrial trial) {
		// if the trial has already loaded them, don't get them again.
		if (trial.getTimers() != null && trial.getTimers().size() > 0) {
			return trial.getTimers();
		}
		Map<Integer, TAUdbTimer> timers = new HashMap<Integer, TAUdbTimer>();
		String query = "select id, name, short_name, source_file, line_number, line_number_end, column_number, column_number_end from timer where trial = ?;";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getID());
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
				TAUdbTimer timer = new TAUdbTimer (session, id, trial, name, shortName, sourceFile, lineNumber, lineNumberEnd, columnNumber, columnNumberEnd);
				timers.put(id, timer);
			}
			results.close();
			statement.close();
			trial.setTimers(timers);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		TAUdbTimerGroup.getTimerGroups(session, trial, timers);
		return timers;
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TAUdbSession session = new TAUdbSession("callpath", false);
		TAUdbTrial trial = TAUdbTrial.getTrial(session, 1, false);
		Map<Integer, TAUdbTimer> timers = TAUdbTimer.getTimers(session, trial);
		for (Integer id : timers.keySet()) {
			TAUdbTimer timer = timers.get(id);
			System.out.println(timer.toString());
		}
		session.close();
	}

}
