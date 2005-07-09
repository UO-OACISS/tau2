package server;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import common.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.database.*;
import java.util.List;

/**
 * The ChartData class is used to select data from the database which 
 * represents the performance profile of the selected trials, and return them
 * in a format for JFreeChart to display them.
 *
 * <P>CVS $Id: ChartData.java,v 1.2 2005/07/09 00:09:38 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class ChartData extends RMIChartData {

	private RMIPerfExplorerModel model;
	private String metricName = null;
	private String groupName = null;
	private String eventName = null;
	private String groupByColumn = null;
	
	
	/**
	 * Constructor
	 * 
	 * @param model
	 * @param dataType
	 */
	public ChartData (RMIPerfExplorerModel model, int dataType) {
		super (dataType);
		this.model = model;
		this.metricName = model.getMetricName();
		this.groupName = model.getGroupName();
		this.eventName = model.getEventName();
	}

	/**
	 * Main method.  The dataType represents the type of chart data being
	 * requested.  The model represents the selected trials of interest.
	 * 
	 * @param model
	 * @param dataType
	 * @return
	 */
	public static ChartData getChartData(RMIPerfExplorerModel model, int dataType) {
		System.out.println("getChartData(" + model.toString() + ")...");
		ChartData chartData = new ChartData(model, dataType);
		chartData.doQuery();
		return chartData;
	}
	
	/**
	 * One method to interface with the database and get the performance
	 * data for the selected trials.  The query is constructed, and the
	 * data is returned as the experiment_name, number_of_threads, and value.
	 * As the experiment_name changes, the data represents a new line on the
	 * chart.  Each line shows the scalability of a particular configuration,
	 * as the number of threads of execution increases.
	 * 
	 * TODO - this code assumes a scalability study!
	 *
	 */
	private void doQuery () {
		// get the database lock
		PerfExplorerServer.getServer().getControl().WAIT("doQuery");
		PreparedStatement statement = null;
		try {
			// all query results are organized the same, only the selection
			// parameters are different.
			statement = buildStatement();
			ResultSet results = statement.executeQuery();
			// TODO - this query assumes a scalability study...!
			String groupingName = null;
			String threadName = null;
			double value = 0.0;
			double numThreads = 0;
			String currentExperiment = "";
			int experimentIndex = -1;
			while (results.next() != false) {
				groupingName = results.getString(1);
				threadName = results.getString(2);
				numThreads = results.getDouble(2);
				value = results.getDouble(3);

				if (!currentExperiment.equals(groupingName)) {
					experimentIndex++;
					currentExperiment = groupingName;
					addRow(groupingName);
				}
				addColumn(experimentIndex, numThreads, value);
			} 
			results.close();
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the analysis settings from the database!";
			System.out.println(error);
			e.printStackTrace();
			PerfExplorerServer.getServer().getControl().SIGNAL("doQuery");		
		}
		PerfExplorerServer.getServer().getControl().SIGNAL("doQuery");		
	}

	/**
	 * The buildStatment method will construct the query to get the chart
	 * data, customizing it to the type of data desired.
	 * 
	 * @return
	 * @throws SQLException
	 */
	private PreparedStatement buildStatement () throws SQLException {
	        DB db = PerfExplorerServer.getServer().getDB();

		PreparedStatement statement = null;
		StringBuffer buf = new StringBuffer();
		Object object = model.getCurrentSelection();
		if (dataType == FRACTION_OF_TOTAL) {
			// The user wants to know the runtime breakdown by events of one 
			// experiment as the number of threads of execution increases.
			buf.append("select ie.name, ");
			buf.append("(t.node_count * t.contexts_per_node * t.threads_per_context), ");
			buf.append("ims.exclusive_percentage from interval_mean_summary ims ");
			buf.append("inner join interval_event ie ");
			buf.append("on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}
			buf.append(" and m.name = ? ");
			buf.append("and ims.inclusive_percentage < 100.0 ");
			buf.append("and (ie.group_name is null or (");
			buf.append("ie.group_name != 'TAU_CALLPATH' ");
			buf.append("and ie.group_name != 'TAU_PHASENAME')) order by 1, 2");
			
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
		} else if (dataType == RELATIVE_EFFICIENCY) {
			// The user wants to know the relative efficiency or speedup
			// of one or more experiments, as the number of threads of 
			// execution increases.
			buf.append("select ");
			if (object instanceof RMIView) {
				if (isLeafView()) {
					buf.append(" " + model.getViewSelectionString() + ", ");
				} else {
					buf.append(" " + groupByColumn + ", ");
				}
			} else {
				buf.append(" e.name, ");
			}
			buf.append("(t.node_count * t.contexts_per_node * t.threads_per_context), ");
			buf.append("ims.inclusive from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true));
			} else {
				buf.append("inner join experiment e on t.experiment = e.id ");
				buf.append("where t.experiment in (");
				List selections = model.getMultiSelection();
				if (selections == null) {
					// just one selection
					buf.append (model.getExperiment().getID());
				} else {
					for (int i = 0 ; i < selections.size() ; i++) {
						Experiment exp = (Experiment)selections.get(i);
						if (i > 0)
							buf.append(",");
						buf.append(exp.getID());
					}
				}
				buf.append(")");
			}
			buf.append(" and m.name = ? and ims.inclusive_percentage = 100.0 ");
			buf.append("order by 1, 2");
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
		} else if (dataType == TOTAL_FOR_GROUP) {
			// The user wants to know the percentage of total runtime that
			// comes from one group of events, such as communication or 
			// computation.  This query is done for 
			// one or more experiments, as the number of threads of 
			// execution increases.
			buf.append("select ");
			if (object instanceof RMIView) {
				if (isLeafView()) {
					buf.append(" " + model.getViewSelectionString() + ", ");
				} else {
					buf.append(" " + groupByColumn + ", ");
				}
			} else {
				buf.append(" e.name, ");
			}
			buf.append("(t.node_count * t.contexts_per_node * t.threads_per_context), ");
			buf.append("sum(ims.exclusive) from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.trial = t.id ");
			buf.append("inner join experiment e on t.experiment = e.id ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true));
			} else {
				buf.append("where t.experiment in (");
				List selections = model.getMultiSelection();
				if (selections == null) {
					// just one selection
					buf.append (model.getExperiment().getID());
				} else {
					for (int i = 0 ; i < selections.size() ; i++) {
						Experiment exp = (Experiment)selections.get(i);
						if (i > 0)
							buf.append(",");
						buf.append(exp.getID());
					}
				}
				buf.append(") ");
			}
			buf.append(" and m.name = ? ");
//			buf.append(" and ims.inclusive_percentage < 100.0 ");
			buf.append("and ie.group_name = ? group by 1, 2, ie.group_name order by 1, 2");
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
			statement.setString(2, groupName);
		} else if (dataType == RELATIVE_EFFICIENCY_EVENTS) {
			// The user wants to know the relative efficiency or speedup
			// of all the events for one experiment, as the number of threads of 
			// execution increases.
			buf.append("select ");
			buf.append(" ie.name, ");
			buf.append("(t.node_count * t.contexts_per_node * t.threads_per_context), ");
			buf.append("ims.exclusive from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}
			buf.append(" and m.name = ? ");
//			buf.append("and ims.inclusive_percentage < 100.0 ");
			buf.append("and (ie.group_name is null or (");
			buf.append("ie.group_name != 'TAU_CALLPATH' ");
			buf.append("and ie.group_name != 'TAU_PHASENAME')) order by 1, 2");
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
		} else if (dataType == RELATIVE_EFFICIENCY_ONE_EVENT) {
			// The user wants to know the relative efficiency or speedup
			// of one event for one or more experiments, as the number of
			// threads of execution increases.
			buf.append("select ");
			if (object instanceof RMIView) {
				if (isLeafView()) {
					buf.append(" " + model.getViewSelectionString() + ", ");
				} else {
					buf.append(" " + groupByColumn + ", ");
				}
			} else {
				buf.append(" e.name, ");
			}
			buf.append("(t.node_count * t.contexts_per_node * t.threads_per_context), ");
			buf.append("inclusive from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true));
			} else {
				buf.append("inner join experiment e on t.experiment = e.id ");
				buf.append("where t.experiment in (");
				List selections = model.getMultiSelection();
				if (selections == null) {
					// just one selection
					buf.append (model.getExperiment().getID());
				} else {
					for (int i = 0 ; i < selections.size() ; i++) {
						Experiment exp = (Experiment)selections.get(i);
						if (i > 0)
							buf.append(",");
						buf.append(exp.getID());
					}
				}
				buf.append(") ");
			}
			buf.append(" and m.name = ? ");
			buf.append("and ie.name = ? order by 1, 2");
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
			statement.setString(2, eventName);
		} else if (dataType == RELATIVE_EFFICIENCY_PHASES) {
			// The user wants to know the relative efficiency or speedup
			// of all the phases for one experiment, as the number of threads of 
			// execution increases.
			buf.append("select ");
			if (object instanceof RMIView) {
				if (isLeafView()) {
					buf.append(" " + model.getViewSelectionString() + ", ");
				} else {
					buf.append(" " + groupByColumn + ", ");
				}
			} else {
				buf.append(" ie.name, ");
			}
			buf.append("(t.node_count * t.contexts_per_node * t.threads_per_context), ");
			buf.append("ims.inclusive from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.trial = t.id ");
			buf.append("where t.experiment = ? and m.name = ? ");
//			buf.append("and ims.inclusive_percentage < 100.0 ");
			buf.append("and ie.group_name = 'TAU_PHASENAME' order by 1, 2");
			statement = db.prepareStatement(buf.toString());
			statement.setInt (1, model.getExperiment().getID());
			statement.setString(2, metricName);
		} else if (dataType == FRACTION_OF_TOTAL_PHASES) {
			// The user wants to know the runtime breakdown by phases 
			// of one experiment as the number of threads of execution
			// increases.
			buf.append("select ");
			if (object instanceof RMIView) {
				if (isLeafView()) {
					buf.append(" " + model.getViewSelectionString() + ", ");
				} else {
					buf.append(" " + groupByColumn + ", ");
				}
			} else {
				buf.append(" ie.name, ");
			}
			buf.append("(t.node_count * t.contexts_per_node * t.threads_per_context), ");
			buf.append("ims.inclusive_percentage from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.trial = t.id ");
			statement = null;
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}
			buf.append(" and m.name = ? ");
//			buf.append("and ims.inclusive_percentage < 100.0 ");
			buf.append("and ie.group_name = 'TAU_PHASENAME' order by 1, 2");
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
		}
		System.out.println(statement.toString());
		return statement;
	}
	
	/**
	 * This method checks to see if the object selected in the view/subview
	 * tree is a leaf node.  A leaf view will have no subviews, so by selecting
	 * all subviews of this view, 0 rows indicates a leaf view.  If the view
	 * is not a leaf, then the "group by" clause is set for the particular
	 * query being constructed.
	 * 
	 * @return
	 */
	private boolean isLeafView () {
	    DB db = PerfExplorerServer.getServer().getDB();

		// check to see if the selected view is a leaf view.
		PreparedStatement statement = null;
		boolean returnValue = true;
		try {
			statement = db.prepareStatement ("select table_name, column_name from trial_view where parent = ?");
			statement.setString(1, model.getViewID());
			ResultSet results = statement.executeQuery();
			if (results.next() != false) {
				String tableName = results.getString(1);
				if (tableName.equalsIgnoreCase("Application")) {
					groupByColumn = new String ("a.");
				} else if (tableName.equalsIgnoreCase("Experiment")) {
					groupByColumn = new String ("e.");
				} else /*if (tableName.equalsIgnoreCase("Trial")) */ {
					groupByColumn = new String ("t.");
				}
				groupByColumn += results.getString(2);
				returnValue = false;
			} 
			results.close();
			statement.close();
		} catch (Exception e) {
			String error = "ERROR: Couldn't select the analysis settings from the database!";
			System.out.println(error);
			e.printStackTrace();
		}
		return returnValue;
	}
}

