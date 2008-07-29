/**
 * 
 */
package glue;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import server.PerfExplorerServer;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.database.DB;

/**
 * @author khuck
 *
 */
public class TrialMeanResult extends AbstractResult {

	private Integer originalThreads = 0;
	private boolean callPath = true;

	/**
	 * 
	 */
	public TrialMeanResult() {
		super();
	}

	/**
	 * @param input
	 */
	public TrialMeanResult(TrialMeanResult input) {
		super(input);
		this.trialID = input.getTrialID();
		this.trial = input.getTrial();
	}

	public TrialMeanResult(Trial trial) {
		super();
		this.trialID = trial.getID();
		this.trial = trial;
		buildTrialMeanResult(trial, null, null);
	}
	
	public TrialMeanResult(Trial trial, String metric, String event, boolean callPath) {
		super();
		this.trialID = trial.getID();
		this.callPath = callPath;
		buildTrialMeanResult(trial, null, null);
	}
	
	private void buildTrialMeanResult(Trial trial, String metric, String event) {
		// hit the databsae, and get the data for this trial
		DB db = PerfExplorerServer.getServer().getDB();
		
		try {
			int threadsPerContext = Integer.parseInt(trial.getField("threads_per_context"));
			int threadsPerNode = Integer.parseInt(trial.getField("contexts_per_node")) * threadsPerContext;
			this.originalThreads = threadsPerNode * Integer.parseInt(trial.getField("node_count"));
			StringBuffer sql = new StringBuffer();
			sql.append("select e.name, ");
			sql.append("m.name, ");
            
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

			sql.append("p.subroutines ");
			sql.append("from interval_event e ");
			sql.append("left outer join interval_mean_summary p ");
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
			if (!callPath) {
            	sql.append(" and (e.group_name is null or e.group_name not like '%TAU_CALLPATH%') ");
			}
			sql.append(" order by 2,1 ");
			
			PreparedStatement statement = db.prepareStatement(sql.toString());
			//System.out.println(sql.toString() + " " + trial.getID() + " " + metric + " " + event);
			
			statement.setInt(1, trial.getID());
			int index = 1;
			if (metric != null) {
				statement.setString(index++, metric);
			}
			if (event != null) {
				statement.setString(index++, event);
			}
			//System.out.println(statement.toString());
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				String eventName = results.getString(1);
				String metricName = results.getString(2);
				this.putExclusive(0, eventName, metricName, results.getDouble(3));
				this.putInclusive(0, eventName, metricName, results.getDouble(4));
				this.putCalls(0, eventName, results.getDouble(5));
				this.putSubroutines(0, eventName, results.getDouble(6));
			}
			results.close();
			statement.close();

		} catch (SQLException exception) {
			System.err.println(exception.getMessage());
			exception.printStackTrace();
		}
	}

	public Integer getTrialID() {
		return trialID;
	}

	public void setTrialID(Integer trialID) {
		this.trialID = trialID;
	}

	/**
	 * @return the originalThreads
	 */
	public Integer getOriginalThreads() {
		return originalThreads;
	}

	/**
	 * @param originalThreads the originalThreads to set
	 */
	public void setOriginalThreads(Integer originalThreads) {
		this.originalThreads = originalThreads;
	}

}
