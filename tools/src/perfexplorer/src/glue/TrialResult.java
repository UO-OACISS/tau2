/**
 * 
 */
package glue;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;

import server.PerfExplorerServer;
import edu.uoregon.tau.perfdmf.IntervalEvent;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.database.DB;

/**
 * This class is an implementation of the AbstractResult class, and loads a trial
 * from the database into a result object.
 * 
 * <P>CVS $Id: TrialResult.java,v 1.7 2008/08/26 23:21:53 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0 
 */

public class TrialResult extends AbstractResult {

	private boolean callPath = true;

	public TrialResult() {
		super();
	}

	/**
	 * @param input
	 */
	public TrialResult(TrialResult input) {
		super(input);
	}

	public TrialResult(Trial trial) {
		super();
		this.trialID = trial.getID();
		this.trial = trial;
		this.name = this.trial.getName();
		buildTrialResult(trial, null, null, null);
	}
	
	public TrialResult(Trial trial, String metric, String event, String thread, boolean callPath) {
		super();
		this.trialID = trial.getID();
		this.trial = trial;
		this.callPath = callPath;
		this.name = this.trial.getName();
		buildTrialResult(trial, metric, event, thread);
	}
	
	private void buildTrialResult(Trial trial, String metric, String event, String thread) {
		// hit the databsae, and get the data for this trial
		DB db = PerfExplorerServer.getServer().getDB();
		
		try {
			int threadsPerContext = Integer.parseInt(trial.getField("threads_per_context"));
			int threadsPerNode = Integer.parseInt(trial.getField("contexts_per_node")) * threadsPerContext;
			StringBuffer sql = new StringBuffer();
			sql.append("select e.name, ");
			sql.append("m.name, ");
			sql.append("(p.node * " + threadsPerNode + ") + ");
			sql.append("(p.context * " + threadsPerContext + ") + ");
			sql.append("p.thread as thread, ");
            
            if (db.getDBType().compareTo("oracle") == 0) {
                sql.append("p.excl, ");
            } else {
                sql.append("p.exclusive, ");
            }

			sql.append("p.inclusive, ");

			if (db.getDBType().compareTo("derby") == 0) {
    			sql.append("p.num_calls, ");
            } else {
    			sql.append("p.call, ");
            }

			sql.append("p.subroutines, e.id ");
			sql.append("from interval_event e ");
			sql.append("left outer join interval_location_profile p ");
			sql.append("on e.id = p.interval_event ");
			sql.append("left outer join metric m on m.trial = e.trial ");
			sql.append("and m.id = p.metric ");
			sql.append("where e.trial = ? ");
			if (metric != null) {
				sql.append(" and m.name = ? ");
			}
			if (event != null) {
				sql.append(" and e.name = ? ");
			}
			if (thread != null) {
				sql.append(" and thread = ? ");				
			}
			if (!callPath) {
            	sql.append(" and (e.group_name is null or e.group_name not like '%TAU_CALLPATH%') ");
			}
			sql.append(" order by 3,2,1 ");
			
			PreparedStatement statement = db.prepareStatement(sql.toString());
			
			statement.setInt(1, trial.getID());
			int index = 1;
			if (metric != null) {
				statement.setString(index++, metric);
			}
			if (event != null) {
				statement.setString(index++, event);
			}
			if (thread != null) {
				statement.setString(index++, thread);
			}
			//System.out.println(statement.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				String eventName = results.getString(1);
				Integer threadID = results.getInt(3);
				this.putExclusive(threadID, eventName, results.getString(2), results.getDouble(4));
				this.putInclusive(threadID, eventName, results.getString(2), results.getDouble(5));
				this.putCalls(threadID, eventName, results.getDouble(6));
				this.putSubroutines(threadID, eventName, results.getDouble(7));
				Integer eventID = results.getInt(8);
				this.eventMap.put(eventID, eventName);
			}
			results.close();
			statement.close();

			// now, get the user events
			sql = new StringBuffer();
			sql.append("select a.name, ");
			sql.append("(p.node * " + threadsPerNode + ") + ");
			sql.append("(p.context * " + threadsPerContext + ") + ");
			sql.append("p.thread as thread, ");
			sql.append("p.sample_count, ");
			sql.append("p.maximum_value, ");
			sql.append("p.minimum_value, ");
			sql.append("p.mean_value, ");
			sql.append("p.standard_deviation ");
			sql.append("from atomic_event a ");
			sql.append("left outer join atomic_location_profile p ");
			sql.append("on a.id = p.atomic_event ");
			sql.append("where a.trial = ? ");
			sql.append(" order by 2,1 ");
			
			statement = db.prepareStatement(sql.toString());
			
			statement.setInt(1, trial.getID());
			//System.out.println(statement.toString());
			results = statement.executeQuery();
			while (results.next() != false) {
				this.putUsereventNumevents(results.getInt(2), results.getString(1), results.getDouble(3));
				this.putUsereventMax(results.getInt(2), results.getString(1), results.getDouble(4));
				this.putUsereventMin(results.getInt(2), results.getString(1), results.getDouble(5));
				this.putUsereventMean(results.getInt(2), results.getString(1), results.getDouble(6));
				this.putUsereventSumsqr(results.getInt(2), results.getString(1), results.getDouble(7));
			}
			results.close();
			statement.close();
			
		} catch (SQLException exception) {
			System.err.println(exception.getMessage());
			exception.printStackTrace();
		}
	}

	public String toString() {
		return this.trial.getName();
	}
	
	public String getEventGroupName(String eventName) {
		String group = null;
		// find the event in the trial
		List<IntervalEvent> events = Utilities.getEventsForTrial(trial, 0);
		for (IntervalEvent event : events) {
			if (event.getName().equals(eventName)) {
				group = event.getGroup();
			}
		}
		// find the group name for the event
		return group;
	}
}

