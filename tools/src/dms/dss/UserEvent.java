package dms.dss;

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
 * <P>CVS $Id: UserEvent.java,v 1.2 2003/08/27 17:07:39 khuck Exp $</P>
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

}

