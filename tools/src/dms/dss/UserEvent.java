package dms.dss;

/**
 * Holds all the data for a user event in the database.
 *
 * <P>CVS $Id: UserEvent.java,v 1.1 2003/08/01 21:42:00 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	%I%, %G%
 */
public class UserEvent {
	private int userEventID;
	private String name;
	private String group;
	private int trialID;
	private int experimentID;
	private int applicationID;

	public void setUserEventID (int id) {
		this.userEventID = id;
	}

	public void setName (String name) {
		this.name = name;
	}

	public void setGroup (String group) {
		this.group = group;
	}

	public void setTrialID (int id) {
		this.trialID = id;
	}

	public void setExperimentID (int id) {
		this.experimentID = id;
	}

	public void setApplicationID (int id) {
		this.applicationID = id;
	}

	public int getUserEventID () {
		return this.userEventID;
	}

	public String getName () {
		return this.name;
	}

	public String getGroup () {
		return this.group;
	}

	public int getTrialID () {
		return this.trialID;
	}

	public int getExperimentID () {
		return this.experimentID;
	}

	public int getApplicationID () {
		return this.applicationID;
	}

}

