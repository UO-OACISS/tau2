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
public class TAUdbTimerParameter {
	private TAUdbTimer timer = null;
	private String name = null;
	private String value = null;

	/**
	 * 
	 */
	public TAUdbTimerParameter(TAUdbTimer timer, String name, String value) {
		this.timer = timer;
		this.name = name;
		this.value = value;
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

	public String toString() {
		return (this.name + "=" + this.value);
	}

	public static Map<String, TAUdbTimerParameter> getTimerParameters(TAUdbSession session, TAUdbTrial trial, Map<Integer, TAUdbTimer> timers) {
		Map<String, TAUdbTimerParameter> parameters = new HashMap<String, TAUdbTimerParameter>();
		String query = "select tp.timer, tp.parameter_name, tp.parameter_value, t.trial from timer_parameter tp join timer t on tg.timer = t.id where t.trial = ?;";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(query);
			statement.setInt(1, trial.getID());
			ResultSet results = statement.executeQuery();
			while(results.next()) {
				Integer timerID = results.getInt(1);
				String name = results.getString(2);
				String value = results.getString(3);
				TAUdbTimer timer = timers.get(timerID);
				TAUdbTimerParameter parameter = new TAUdbTimerParameter(timer, name, value);
				timer.addParameter(parameter);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return parameters;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
