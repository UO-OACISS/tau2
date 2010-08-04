package edu.uoregon.tau.perfdmf;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Vector;

import edu.uoregon.tau.perfdmf.database.DB;
/**
 * Holds all the data for a atomic event data object in the database.
 * This object is returned by the DataSession class and all of its subtypes.
 * The AtomicEventData object contains all the information associated with
 * an atomic event location instance from which the TAU performance data has been generated.
 * A atomic event location is associated with one node, context, thread, atomic event, trial, 
 * experiment and application.
 * <p>
 * A AtomicEventData object has information
 * related to one particular atomic event location in the trial, including the ID of the atomic event,
 * the node, context and thread that identify the location, and the data collected for this
 * location, such as sample count, maximum value, minimum value, mean value and sum squared.  
 *
 * <P>CVS $Id: AtomicLocationProfile.java,v 1.2 2007/05/02 19:43:28 amorris Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 * @see		DataSession#getAtomicEventData
 * @see		DataSession#setAtomicEvent
 * @see		DataSession#setNode
 * @see		DataSession#setContext
 * @see		DataSession#setThread
 * @see		DataSession#setMetric
 * @see		AtomicEvent
 */
public class AtomicLocationProfile {
    private int atomicEventID;
    private int node;
    private int context;
    private int thread;
    private int sampleCount;
    private double maximumValue;
    private double minimumValue;
    private double meanValue;
    private double sumSquared;

    /**
     * Returns the unique ID for the atomic event that owns this data
     *
     * @return	the atomic event ID.
     * @see		AtomicEvent
     */
    public int getAtomicEventID () {
	return this.atomicEventID;
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
     * Returns the maximum value recorded for this atomic event.
     *
     * @return	the maximum value.
     */
    public double getMaximumValue () {
	return this.maximumValue;
    }

    /**
     * Returns the minimum value recorded for this atomic event.
     *
     * @return	the minimum value.
     */
    public double getMinimumValue () {
	return this.minimumValue;
    }

    /**
     * Returns the mean value recorded for this atomic event.
     *
     * @return	the mean value.
     */
    public double getMeanValue () {
	return this.meanValue;
    }

    /**
     * Returns the sum squared calculated for this atomic event.
     *
     * @return	the sum squared value.
     */
    public double getSumSquared () {
	return this.sumSquared;
    }

    /**
     * Sets the unique atomic event ID for the atomic event at this location.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	atomicEventID a unique atomic event ID.
     */
    public void setAtomicEventID (int atomicEventID) {
	this.atomicEventID = atomicEventID;
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
     * Sets the number of times the atomic event occurred at this location.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	sampleCount the sample count at this location
     */
    public void setSampleCount (int sampleCount) {
	this.sampleCount = sampleCount;
    }

    /**
     * Sets the maximum value recorded for this atomic event at this location.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	maximumValue the maximum value at this location
     */
    public void setMaximumValue (double maximumValue) {
	this.maximumValue = maximumValue;
    }

    /**
     * Sets the minimum value recorded for this atomic event at this location.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	minimumValue the minimum value at this location
     */
    public void setMinimumValue (double minimumValue) {
	this.minimumValue = minimumValue;
    }

    /**
     * Sets the mean value calculated for this atomic event at this location.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	meanValue the mean value at this location
     */
    public void setMeanValue (double meanValue) {
	this.meanValue = meanValue;
    }

    /**
     * Sets the sum squared value calculated for this atomic event at this location.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	sumSquared the sum squared value at this location
     */
    public void setSumSquared (double sumSquared) {
	this.sumSquared = sumSquared;
    }

    /**
     * Documentation?
     */

    public static Vector<AtomicLocationProfile> getAtomicEventData(DB db, String whereClause) {
	Vector<AtomicLocationProfile> atomicEventData = new Vector<AtomicLocationProfile>();
	// create a string to hit the database
	StringBuffer buf = new StringBuffer();
	buf.append("select p.atomic_event, p.node, ");
	buf.append("p.context, p.thread, p.sample_count, ");
	buf.append("p.maximum_value, p.minimum_value, p.mean_value, ");
	buf.append("p.standard_deviation, e.trial ");
	buf.append("from " + db.getSchemaPrefix() + "atomic_location_profile p ");
	buf.append("inner join " + db.getSchemaPrefix() + "atomic_event e on e.id = p.atomic_event ");
	buf.append(whereClause);
	buf.append(" order by p.node, p.context, p.thread, p.atomic_event");
	// System.out.println(buf.toString());

	// get the results
	try {
	    ResultSet resultSet = db.executeQuery(buf.toString());	
	    while (resultSet.next() != false) {
		AtomicLocationProfile ueDO = new AtomicLocationProfile();
		ueDO.setAtomicEventID(resultSet.getInt(1));
		ueDO.setNode(resultSet.getInt(2));
		ueDO.setContext(resultSet.getInt(3));
		ueDO.setThread(resultSet.getInt(4));
		ueDO.setSampleCount(resultSet.getInt(5));
		ueDO.setMaximumValue(resultSet.getDouble(6));
		ueDO.setMinimumValue(resultSet.getDouble(7));
		ueDO.setMeanValue(resultSet.getDouble(8));
		ueDO.setSumSquared(resultSet.getDouble(9));
		atomicEventData.addElement(ueDO);
	    }
	    resultSet.close(); 
	} catch (Exception ex) {
	    ex.printStackTrace();
	    return null;
	}
	return atomicEventData;
    }

    /**
     * Documentation?
     */

    public static void getAtomicEventDetail(DB db, AtomicEvent atomicEvent, String whereClause) {
	// create a string to hit the database
	StringBuffer buf = new StringBuffer();
	buf.append("select p.atomic_event, avg(p.sample_count), ");
	buf.append("avg(p.maximum_value), avg(p.minimum_value), avg(p.mean_value), ");
	buf.append("avg(p.standard_deviation), ");
	buf.append("sum(p.sample_count), ");
	buf.append("sum(p.maximum_value), sum(p.minimum_value), sum(p.mean_value), ");
	buf.append("sum(p.standard_deviation) ");
	buf.append("from " + db.getSchemaPrefix() + "atomic_location_profile p ");
	buf.append("inner join " + db.getSchemaPrefix() + "atomic_event e on e.id = p.atomic_event ");
	buf.append(whereClause);
	buf.append(" group by p.atomic_event");
	buf.append(" order by p.atomic_event");
	// System.out.println(buf.toString());

	// get the results
	try {
	    ResultSet resultSet = db.executeQuery(buf.toString());	
	    AtomicLocationProfile eMS = new AtomicLocationProfile();
	    AtomicLocationProfile eTS = new AtomicLocationProfile();
	    while (resultSet.next() != false) {
		eMS.setAtomicEventID(resultSet.getInt(1));
		eTS.setAtomicEventID(resultSet.getInt(1));
		eMS.setSampleCount((int)(resultSet.getDouble(2)));
		eMS.setMaximumValue(resultSet.getDouble(3));
		eMS.setMinimumValue(resultSet.getDouble(4));
		eMS.setMeanValue(resultSet.getDouble(5));
		eMS.setSumSquared(resultSet.getDouble(6));
		eTS.setSampleCount((int)(resultSet.getDouble(7)));
		eTS.setMaximumValue(resultSet.getDouble(8));
		eTS.setMinimumValue(resultSet.getDouble(9));
		eTS.setMeanValue(resultSet.getDouble(10));
		eTS.setSumSquared(resultSet.getDouble(11));
	    }
	    resultSet.close(); 
	    atomicEvent.setMeanSummary(eMS);
	    atomicEvent.setTotalSummary(eTS);
	}catch (Exception ex) {
	    ex.printStackTrace();
	}
    }

    public void saveAtomicEventData(DB db, int atomicEventID) {
	try {
	    PreparedStatement statement = null;
	    statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix() + "atomic_location_profile (atomic_event, node, context, thread, sample_count, maximum_value, minimum_value, mean_value, standard_deviation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
	    statement.setInt(1, atomicEventID);
	    statement.setInt(2, node);
	    statement.setInt(3, context);
	    statement.setInt(4, thread);
	    statement.setInt(5, sampleCount);
	    statement.setDouble(6, maximumValue);
	    statement.setDouble(7, minimumValue);
	    statement.setDouble(8, meanValue);
	    statement.setDouble(9, sumSquared);
	    statement.executeUpdate();
	} catch (SQLException e) {
	    System.out.println("An error occurred while saving the trial.");
	    e.printStackTrace();
	}
    }
}

