package dms.dss;

/**
 * Holds all the data for a user event data object in the database.
 *
 * <P>CVS $Id: UserEventDataObject.java,v 1.5 2003/08/25 17:32:16 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	%I%, %G%
 */
public class UserEventDataObject {
	private int userEventIndexID;
	private int userEventID;
	private int profileID;
	private int node;
	private int context;
	private int thread;
	private int sampleCount;
	private double maximumValue;
	private double minimumValue;
	private double meanValue;
	private double standardDeviation;

	public void setUserEventIndexID (int userEventID) {
		this.userEventID = userEventID;
	}

	public void setUserEventID (int userEventID) {
		this.userEventID = userEventID;
	}

	public void setProfileID (int profileID) {
		this.profileID = profileID;
	}

	public void setNode (int node) {
		this.node = node;
	}

	public void setContext (int context) {
		this.context = context;
	}

	public void setThread (int thread) {
		this.thread = thread;
	}

	public void setSampleCount (int sampleCount) {
		this.sampleCount = sampleCount;
	}

	public void setMaximumValue (double maximumValue) {
		this.maximumValue = maximumValue;
	}

	public void setMinimumValue (double minimumValue) {
		this.minimumValue = minimumValue;
	}

	public void setMeanValue (double meanValue) {
		this.meanValue = meanValue;
	}

	public void setStandardDeviation (double standardDeviation) {
		this.standardDeviation = standardDeviation;
	}

	public int getUserEventID () {
		return this.userEventID;
	}

	public int getProfileID () {
		return this.profileID;
	}

	public int getNode () {
		return this.node;
	}

	public int getContext () {
		return this.context;
	}

	public int getThread () {
		return this.thread;
	}

	public int getSampleCount () {
		return this.sampleCount;
	}

	public double getMaximumValue () {
		return this.maximumValue;
	}

	public double getMinimumValue () {
		return this.minimumValue;
	}

	public double getMeanValue () {
		return this.meanValue;
	}

	public double getStandardDeviation () {
		return this.standardDeviation;
	}

}

