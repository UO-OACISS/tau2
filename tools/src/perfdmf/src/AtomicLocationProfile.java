package edu.uoregon.tau.perfdmf;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
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

	public static Vector<UserEventProfile> getAtomicEventData(DB db, String whereClause, DataSource dataSource, Map<Integer, UserEvent> eventMap) {
		if(db.getSchemaVersion()>0) return TAUdbGetAtomicEventData(db,whereClause, dataSource, eventMap);
		Vector<UserEventProfile> userEventData = new Vector<UserEventProfile>();
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
				int eventid = resultSet.getInt(1);
				int nodeid = resultSet.getInt(2);
				int contextid = resultSet.getInt(3);
				int threadid = resultSet.getInt(4);
				Thread thread = dataSource.addThread(nodeid, contextid, threadid);
				UserEvent userEvent = eventMap.get(new Integer(eventid));
				UserEventProfile userEventProfile = thread.getUserEventProfile(userEvent);

				if (userEventProfile == null) {
					userEventProfile = new UserEventProfile(userEvent);
					thread.addUserEventProfile(userEventProfile);
				}

				userEventProfile.setNumSamples(resultSet.getInt(5));
				userEventProfile.setMaxValue(resultSet.getDouble(6));
				userEventProfile.setMinValue(resultSet.getDouble(7));
				userEventProfile.setMeanValue(resultSet.getDouble(8));
				userEventProfile.setSumSquared(resultSet.getDouble(9));
				userEventProfile.updateMax();

				userEventData.addElement(userEventProfile);
			}
			resultSet.close(); 
		} catch (Exception ex) {
			ex.printStackTrace();
			return null;
		}
		return userEventData;
	}

	private static Vector<UserEventProfile> TAUdbGetAtomicEventData(DB db,
			String whereClause, DataSource dataSource, Map<Integer, UserEvent> eventMap) {
		Vector<UserEventProfile> userEventData = new Vector<UserEventProfile>();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select v.counter, h.node_rank as node, h.context_rank as context, h.thread_rank as thread,v.sample_count,");
		buf.append("v.maximum_value, v.minimum_value, v.mean_value,v.standard_deviation");
		buf.append(" from " +
				db.getSchemaPrefix() +"counter_value v  left outer join counter e on v.counter = e.id ");
		buf.append("left outer join " +
				db.getSchemaPrefix() +"thread h on v.thread = h.id ");

		buf.append(whereClause);
		buf.append(" order by node, context, thread, v.counter");
		//System.out.println(buf.toString());

		// get the results
		try {
			ResultSet resultSet = db.executeQuery(buf.toString());	
			while (resultSet.next() != false) {
				int eventid = resultSet.getInt(1);
				int nodeid = resultSet.getInt(2);
				int contextid = resultSet.getInt(3);
				int threadid = resultSet.getInt(4);
				Thread thread = dataSource.addThread(nodeid, contextid, threadid);
				UserEvent userEvent = eventMap.get(new Integer(eventid));
				UserEventProfile userEventProfile = thread.getUserEventProfile(userEvent);
				if (userEventProfile == null) {
					userEventProfile = new UserEventProfile(userEvent);
					thread.addUserEventProfile(userEventProfile);
				}
				userEventProfile.setNumSamples(resultSet.getInt(5));
				userEventProfile.setMaxValue(resultSet.getDouble(6));
				userEventProfile.setMinValue(resultSet.getDouble(7));
				userEventProfile.setMeanValue(resultSet.getDouble(8));
				userEventProfile.setSumSquared(resultSet.getDouble(9));
				userEventProfile.updateMax();
				userEventData.addElement(userEventProfile);
			}
			resultSet.close(); 
		} catch (Exception ex) {
			ex.printStackTrace();
			return null;
		}
		return userEventData;
	}

	public static void saveAtomicEventData(DB db, Hashtable<Integer, Integer> newUEHash, List<Thread> threads) {
		try {
			PreparedStatement statement = null;
			if (db.getSchemaVersion() > 0) {
				// need to look up the thread index!
				statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix() + 
						"counter_value (atomic_event, sample_count, maximum_value, minimum_value, mean_value, standard_deviation, thread) VALUES (?, ?, ?, ?, ?, ?, ?)");
			} else {
				statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix() + 
						"atomic_location_profile (atomic_event, sample_count, maximum_value, minimum_value, mean_value, standard_deviation, node, context, thread) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
			}
	        for (Iterator<Thread> it = threads.iterator(); it.hasNext();) {
	            edu.uoregon.tau.perfdmf.Thread thread = it.next();
	            for (Iterator<UserEventProfile> e4 = thread.getUserEventProfiles(); e4.hasNext();) {
	                UserEventProfile uep = e4.next();
	                if (uep != null) {
	                    statement.setInt(1, uep.getUserEvent().getID());
	                    statement.setInt(2, (int) uep.getNumSamples());
	                    statement.setDouble(3, uep.getMaxValue());
	                    statement.setDouble(4, uep.getMinValue());
	                    statement.setDouble(5, uep.getMeanValue());
	                    statement.setDouble(6, uep.getSumSquared());
	                    if (db.getSchemaVersion() > 0) {
	                    	statement.setInt(7, thread.getThreadID());
	                    } else {
		                    statement.setInt(7, thread.getNodeID());
		                    statement.setInt(8, thread.getContextID());
		                    statement.setInt(9, thread.getThreadID());
	                    }
	        			statement.addBatch();
	                }
	            }
	        }
			statement.executeBatch();
			statement.close();
		} catch (SQLException e) {
			System.out.println("An error occurred while saving the trial.");
			e.printStackTrace();
		}
	}
}

