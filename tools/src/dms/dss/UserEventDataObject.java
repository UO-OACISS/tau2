package dms.dss;

/**
 * Holds all the data for a user event data object in the database.
 * This object is returned by the DataSession class and all of its subtypes.
 * The UserEventData object contains all the information associated with
 * an user event location instance from which the TAU performance data has been generated.
 * A user event location is associated with one node, context, thread, user event, trial, 
 * experiment and application.
 * <p>
 * A UserEventData object has information
 * related to one particular user event location in the trial, including the ID of the user event,
 * the node, context and thread that identify the location, and the data collected for this
 * location, such as sample count, maximum value, minimum value, mean value and standard deviation.  
 *
 * <P>CVS $Id: UserEventDataObject.java,v 1.6 2003/08/27 17:07:39 khuck Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getUserEventData
 * @see		DataSession#setUserEvent
 * @see		DataSession#setNode
 * @see		DataSession#setContext
 * @see		DataSession#setThread
 * @see		DataSession#setMetric
 * @see		UserEvent
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

/**
 * Returns the unique ID for the user event that owns this data
 *
 * @return	the user event ID.
 * @see		UserEvent
 */
	public int getUserEventID () {
		return this.userEventID;
	}

/**
 * Returns the unique ID for this data object. 
 *
 * @return	the user event data ID.
 */
	public int getProfileID () {
		return this.profileID;
	}

/**
 * Returns the node for this data location.
 *
 * @return the node index.
 */
	public int getNode () {
		return this.node;
	}

/**
 * Returns the context for this data location.
 *
 * @return the context index.
 */
	public int getContext () {
		return this.context;
	}

/**
 * Returns the thread for this data location.
 *
 * @return the thread index.
 */
	public int getThread () {
		return this.thread;
	}

/**
 * Returns the number of calls to this function at this location.
 *
 * @return	the number of calls.
 */
	public int getSampleCount () {
		return this.sampleCount;
	}

/**
 * Returns the maximum value recorded for this user event.
 *
 * @return	the maximum value.
 */
	public double getMaximumValue () {
		return this.maximumValue;
	}

/**
 * Returns the minimum value recorded for this user event.
 *
 * @return	the minimum value.
 */
	public double getMinimumValue () {
		return this.minimumValue;
	}

/**
 * Returns the mean value recorded for this user event.
 *
 * @return	the mean value.
 */
	public double getMeanValue () {
		return this.meanValue;
	}

/**
 * Returns the standard deviation calculated for this user event.
 *
 * @return	the standard deviation value.
 */
	public double getStandardDeviation () {
		return this.standardDeviation;
	}

/**
 * Sets the unique user event ID for the user event at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	userEventID a unique user event ID.
 */
	public void setUserEventID (int userEventID) {
		this.userEventID = userEventID;
	}

/**
 * Sets the unique ID for this data object. 
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	profileID	the unique user event data ID.
 */
	public void setProfileID (int profileID) {
		this.profileID = profileID;
	}

/**
 * Sets the node of the current location that this data object represents.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	node the node for this location.
 */
	public void setNode (int node) {
		this.node = node;
	}

/**
 * Sets the context of the current location that this data object represents.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	context the context for this location.
 */
	public void setContext (int context) {
		this.context = context;
	}

/**
 * Sets the thread of the current location that this data object represents.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	thread the thread for this location.
 */
	public void setThread (int thread) {
		this.thread = thread;
	}

/**
 * Sets the number of times the user event occurred at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	sampleCount the sample count at this location
 */
	public void setSampleCount (int sampleCount) {
		this.sampleCount = sampleCount;
	}

/**
 * Sets the maximum value recorded for this user event at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	maximumValue the maximum value at this location
 */
	public void setMaximumValue (double maximumValue) {
		this.maximumValue = maximumValue;
	}

/**
 * Sets the minimum value recorded for this user event at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	minimumValue the minimum value at this location
 */
	public void setMinimumValue (double minimumValue) {
		this.minimumValue = minimumValue;
	}

/**
 * Sets the mean value calculated for this user event at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	meanValue the mean value at this location
 */
	public void setMeanValue (double meanValue) {
		this.meanValue = meanValue;
	}

/**
 * Sets the standard deviation value calculated for this user event at this location.
 * <i> NOTE: This method is used by the DataSession object to initialize
 * the object.  Not currently intended for use by any other code.</i>
 *
 * @param	standardDeviation the standard deviation value at this location
 */
	public void setStandardDeviation (double standardDeviation) {
		this.standardDeviation = standardDeviation;
	}

}

