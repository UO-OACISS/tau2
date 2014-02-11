/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;

import edu.uoregon.tau.perfdmf.IntervalEvent;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfexplorer.common.RMISortableIntervalEvent;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;

/**
 * This class is an implementation of the AbstractResult class, and loads a trial
 * from the database into a result object.
 * 
 * <P>CVS $Id: TrialResult.java,v 1.14 2009/04/03 23:53:37 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0 
 */

public class TrialResult extends AbstractResult {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7164598771800856462L;
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
	
	public TrialResult(Trial trial, List<String> metrics, List<String> events, List<String> threads, boolean callPath) {
		super();
		this.trialID = trial.getID();
		this.trial = trial;
		this.callPath = callPath;
		this.name = this.trial.getName();
		buildTrialResult(trial, metrics, events, threads);
	}
	
	private void buildTrialResult(Trial trial, List<String> metrics, List<String> events, List<String> threads) {
		// hit the databsae, and get the data for this trial
		DB db = PerfExplorerServer.getServer().getDB();
		if (db.getSchemaVersion() > 0) {
			buildTrialResultFromTAUdb(trial, metrics, events, threads);
			return;
		}

		StringBuilder sql = null;
		PreparedStatement statement = null;
		
		try {
			int threadsPerContext = Integer.parseInt(trial.getField("threads_per_context"));
			int threadsPerNode = Integer.parseInt(trial.getField("contexts_per_node")) * threadsPerContext;
			sql = new StringBuilder();
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
			if (metrics != null) {
                sql.append(" and m.name in (");
                int count = 0;
                for (String m : metrics) {
                    if (count > 0) {
                        sql.append(",");
                    }
                    sql.append("'" + m + "'");
					count++;
                }
                sql.append(") ");
			}
			if (events != null) {
                sql.append(" and e.name in (");
                int count = 0;
                for (String e : events) {
                    if (count > 0) {
                        sql.append(",");
                    }
                    sql.append("'" + e + "'");
					count++;
                }
                sql.append(") ");

			}
			if (threads != null) {
                sql.append(" and thread in (");
                int count = 0;
                for (String h : threads) {
                    if (count > 0) {
                        sql.append(",");
                    }
                    sql.append(h);
					count++;
                }
                sql.append(") ");

			}
			if (!callPath) {
            	sql.append(" and (e.group_name is null or e.group_name not like '%TAU_CALLPATH%') ");
			}
			sql.append(" order by 3,2,1 ");
			
			statement = db.prepareStatement(sql.toString());
			
			statement.setInt(1, trial.getID());
			//System.out.println(statement.toString());
			long start = System.currentTimeMillis();
			ResultSet results = statement.executeQuery();
			long elapsedTimeMillis = System.currentTimeMillis()-start;
			float elapsedTimeSec = elapsedTimeMillis/1000F;
			System.out.println("Time to query interval data: " + elapsedTimeSec + " seconds");
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
			sql = new StringBuilder();
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
			start = System.currentTimeMillis();
			results = statement.executeQuery();
			elapsedTimeMillis = System.currentTimeMillis()-start;
			elapsedTimeSec = elapsedTimeMillis/1000F;
			System.out.println("Time to query counter data: " + elapsedTimeSec + " seconds");
			while (results.next() != false) {
//				Integer threadID = results.getInt(2);
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
			if (statement != null)
				System.err.println(statement);
			else
				System.err.println(sql);
		}
	}

	private void buildTrialResultFromTAUdb(Trial trial, List<String> metrics, List<String> events, List<String> threads) {
		// hit the database, and get the data for this trial
		DB db = PerfExplorerServer.getServer().getDB();
		StringBuilder sql = null;
		PreparedStatement statement = null;
		
		try {
			sql = new StringBuilder();
			if (!callPath) {
				// easy query.
        		sql.append(" select t.name, m.name, h.thread_index, tv.exclusive_value, ");
				sql.append(" tv.inclusive_value, tcd.calls, tcd.subroutines, cp.id from timer t ");
				sql.append(" left outer join timer_callpath cp on t.trial = " + trial.getID());
				sql.append(" and cp.timer = t.id ");
        		sql.append(" left outer join timer_call_data tcd on tcd.timer_callpath = cp.id ");
        		sql.append(" left outer join timer_value tv on tv.timer_call_data = tcd.id ");
        		sql.append(" left outer join metric m on m.trial = " + trial.getID() + " and tv.metric = m.id ");
        		sql.append(" left outer join thread h on h.trial = " + trial.getID() + " and tcd.thread = h.id ");
			} else {
         		sql.append(" with recursive cp (id, parent, timer, name) as (  ");
         		sql.append(" SELECT tc.id, tc.parent, tc.timer, timer.name  ");
         		sql.append(" FROM  timer_callpath tc inner join timer on tc.timer = timer.id ");
				sql.append(" where timer.trial = " + trial.getID() + " and tc.parent is null ");
         		sql.append(" UNION ALL ");
         		sql.append(" SELECT d.id, d.parent, d.timer, ");
				if (db.getDBType().compareTo("h2") == 0) {
					sql.append("concat (cp.name, ' => ', dt.name) ");
				} else {
					sql.append("cp.name || ' => ' || dt.name ");
				}
         		sql.append(" FROM timer_callpath AS d JOIN cp ON (d.parent = cp.id) ");
				sql.append(" join timer dt on d.timer = dt.id where dt.trial = " + trial.getID() +" ) ");
        		sql.append(" select cp.name, m.name, h.thread_index, tv.exclusive_value, ");
				sql.append(" tv.inclusive_value, tcd.calls, tcd.subroutines, cp.id from cp ");
        		sql.append(" left outer join timer_call_data tcd on tcd.timer_callpath = cp.id ");
        		sql.append(" left outer join timer_value tv on tv.timer_call_data = tcd.id ");
        		sql.append(" left outer join metric m on m.trial = " + trial.getID() + " and tv.metric = m.id ");
        		sql.append(" left outer join thread h on h.trial = " + trial.getID() + " and tcd.thread = h.id ");
            }

			sql.append("where m.trial = " + trial.getID());
			if (metrics != null) {
                sql.append(" and m.name in (");
                int count = 0;
                for (String m : metrics) {
                    if (count > 0) {
                        sql.append(",");
                    }
                    sql.append("'" + m + "'");
					count++;
                }
                sql.append(") ");
			}
			if (events != null) {
                sql.append(" and t.name in (");
                int count = 0;
                for (String e : events) {
                    if (count > 0) {
                        sql.append(",");
                    }
                    sql.append("'" + e + "'");
					count++;
                }
                sql.append(") ");

			}
			if (threads != null) {
                sql.append(" and h.thread_index in (");
                int count = 0;
                for (String t : threads) {
                    if (count > 0) {
                        sql.append(",");
                    }
                    sql.append(t);
					count++;
                }
                sql.append(") ");

			} else {
				sql.append(" and h.thread_index > -1 ");
			}
			sql.append(" order by 3,2,1 ");
			
			statement = db.prepareStatement(sql.toString());
			
			//System.out.println(statement.toString());
			long start = System.currentTimeMillis();
			ResultSet results = statement.executeQuery();
			long elapsedTimeMillis = System.currentTimeMillis()-start;
			float elapsedTimeSec = elapsedTimeMillis/1000F;
			System.out.println("Time to query interval data: " + elapsedTimeSec + " seconds");
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
			sql = new StringBuilder();
			int threadsPerContext = Integer.parseInt(trial.getField("threads_per_context"));
			int threadsPerNode = Integer.parseInt(trial.getField("contexts_per_node")) * threadsPerContext;
// select a.name, (p.node * 1) + (p.context * 1) + p.thread as thread, p.sample_count, p.maximum_value, p.minimum_value, p.mean_value, p.standard_deviation from atomic_event a left outer join atomic_location_profile p on a.id = p.atomic_event where a.trial = '48014'  order by 2,1
			sql.append("select c.name, ");
			sql.append("h.thread_index as thread, ");
			sql.append("cv.sample_count, ");
			sql.append("cv.maximum_value, ");
			sql.append("cv.minimum_value, ");
			sql.append("cv.mean_value, ");
			sql.append("cv.standard_deviation ");
			sql.append("from counter c ");
			sql.append("left outer join counter_value cv ");
			sql.append("on c.id = cv.counter and c.trial = " + trial.getID());
			sql.append(" left outer join thread h ");
			sql.append("on h.id = cv.thread and h.trial = " + trial.getID());
			sql.append(" where c.trial = " + trial.getID());
			if (threads != null) {
                sql.append(" and h.thread_index in (");
                int count = 0;
                for (String h : threads) {
                    if (count > 0) {
                        sql.append(",");
                    }
                    sql.append(h);
                }
                sql.append(") ");

			} else {
				sql.append(" and h.thread_index > -1 ");
			}
			sql.append(" order by 2,1 ");
			
			statement = db.prepareStatement(sql.toString());
			
			//System.out.println(statement.toString());
			start = System.currentTimeMillis();
			results = statement.executeQuery();
			elapsedTimeMillis = System.currentTimeMillis()-start;
			elapsedTimeSec = elapsedTimeMillis/1000F;
			System.out.println("Time to query counter data: " + elapsedTimeSec + " seconds");
			while (results.next() != false) {
//				Integer threadID = results.getInt(2);
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
			if (statement != null)
				System.err.println(statement);
			else
				System.err.println(sql);
		}
	}

	public String toString() {
		return this.trial.getName();
	}
	
	public String getEventGroupName(String eventName) {
		String group = null;
		// find the event in the trial
		List<RMISortableIntervalEvent> events = Utilities.getEventsForTrial(trial, 0);
		for (RMISortableIntervalEvent event : events) {
			if (event.getFunction().getName().equals(eventName)) {
				group = event.getFunction().getGroupString();
			}
		}
		// find the group name for the event
		return group;
	}
}

