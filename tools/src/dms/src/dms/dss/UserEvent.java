package dms.dss;

import dms.perfdb.DB;
import java.sql.ResultSet;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.Vector;

/**
 * Holds all the data for a user event in the database.
 * This object is returned by the DataSession class and all of its subtypes.
 * The UserEvent object contains all the information associated with
 * an user event from which the TAU performance data has been generated.
 * A user event is associated with one trial, experiment and application, and has one or more
 * UserEventData objects (one for each metric in the trial) associated with it.  
 * <p>
 * A user event has particular information, including the name of the user event, 
 * the TAU group, and the application, experiment and trial IDs.
 *
 * <P>CVS $Id: UserEvent.java,v 1.3 2004/04/07 17:36:58 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getUserEvents
 * @see		DataSession#setUserEvent
 * @see		Application
 * @see		Experiment
 * @see		Trial
 * @see		UserEventDataObject
 */
public class UserEvent {
	private int userEventID;
	private String name;
	private String group;
	private int trialID;
	private int experimentID;
	private int applicationID;

/**
 * Gets the unique identifier of this user event object.
 *
 * @return	the unique identifier of the user event
 */
	public int getUserEventID () {
		return this.userEventID;
	}

/**
 * Gets the name of the user event object.
 *
 * @return	the name of the user event
 */
	public String getName () {
		return this.name;
	}

/**
 * Gets the group of the user event object.
 *
 * @return	the group of the user event
 */
	public String getGroup () {
		return this.group;
	}

/**
 * Gets the trial ID of the user event object.
 *
 * @return	the trial ID of the user event
 */
	public int getTrialID () {
		return this.trialID;
	}

/**
 * Gets the experiment ID of the user event object.
 *
 * @return	the experiment ID of the user event
 */
	public int getExperimentID () {
		return this.experimentID;
	}

/**
 * Gets the application ID of the user event object.
 *
 * @return	the application ID of the user event
 */
	public int getApplicationID () {
		return this.applicationID;
	}

/**
 * Sets the unique ID associated with this user event.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	id unique ID associated with this user event
 */
	public void setUserEventID (int id) {
		this.userEventID = id;
	}

/**
 * Sets the user event name.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	name the name of the user event
 */
	public void setName (String name) {
		this.name = name;
	}

/**
 * Sets the TAU group of this user event object.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	group the TAU group the user event is in.
 */
	public void setGroup (String group) {
		this.group = group;
	}

/**
 * Sets the trial ID of this user event object.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	id the trial ID for the user event.
 */
	public void setTrialID (int id) {
		this.trialID = id;
	}

/**
 * Sets the experiment ID of this user event object.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	id the experiment ID for the user event.
 */
	public void setExperimentID (int id) {
		this.experimentID = id;
	}

/**
 * Sets the application ID of this user event object.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	id the application ID for the user event.
 */
	public void setApplicationID (int id) {
		this.applicationID = id;
	}

	// returns a Vector of UserEvents
	public static Vector getUserEvents(DB db, String whereClause) {
		Vector userEvents = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select u.id, u.trial, u.name, ");
		buf.append("u.group_name ");
		buf.append("from user_event u inner join trial t on u.trial = t.id ");
		buf.append("inner join experiment e on t.experiment = e.id ");
		buf.append(whereClause);
		// System.out.println(buf.toString());

		// get the results
		try {
	    	ResultSet resultSet = db.executeQuery(buf.toString());	
			UserEvent tmpUserEvent = null;
	    	while (resultSet.next() != false) {
				UserEvent ue = new UserEvent();
				ue.setUserEventID(resultSet.getInt(1));
				ue.setTrialID(resultSet.getInt(2));
				ue.setName(resultSet.getString(3));
				ue.setGroup(resultSet.getString(4));
				userEvents.addElement(ue);
	    	}
			resultSet.close(); 
		}catch (Exception ex) {
	    	ex.printStackTrace();
	    	return null;
		}
		
		return userEvents;
	}

	public int saveUserEvent(DB db, int newTrialID) {
		int newUserEventID = 0;
		try {
			PreparedStatement statement = null;
			statement = db.prepareStatement("INSERT INTO user_event (trial, name, group_name) VALUES (?, ?, ?)");
			statement.setInt(1, newTrialID);
			statement.setString(2, name);
			statement.setString(3, group);
			statement.executeUpdate();
			String tmpStr = new String();
			if (db.getDBType().compareTo("mysql") == 0)
				tmpStr = "select LAST_INSERT_ID();";
			if (db.getDBType().compareTo("db2") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM user_event";
			else
				tmpStr = "select currval('user_event_id_seq');";
			newUserEventID = Integer.parseInt(db.getDataItem(tmpStr));
		} catch (SQLException e) {
			System.out.println("An error occurred while saving the trial.");
			e.printStackTrace();
			System.exit(0);
		}
		return newUserEventID;
	}

}

