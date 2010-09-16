package edu.uoregon.tau.perfexplorer.server;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.IntervalEvent;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfexplorer.common.ChartDataType;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;
import edu.uoregon.tau.perfexplorer.common.RMIChartData;
import edu.uoregon.tau.perfexplorer.common.RMIPerfExplorerModel;
import edu.uoregon.tau.perfexplorer.common.RMIView;

/**
 * The ChartData class is used to select data from the database which 
 * represents the performance profile of the selected trials, and return them
 * in a format for JFreeChart to display them.
 *
 * <P>CVS $Id: ChartData.java,v 1.56 2009/08/17 14:43:26 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class ChartData extends RMIChartData {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4066428857538615268L;
	private RMIPerfExplorerModel model;
	private String metricName = null;
	private String groupName = null;
	private String eventName = null;
	private String groupByColumn = null;
	private List<Double> columnValues = null;
	private StringBuilder buf = null;
	
	/**
	 * Constructor
	 * 
	 * @param model
	 * @param dataType
	 */
	public ChartData (RMIPerfExplorerModel model, ChartDataType dataType) {
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
	public static ChartData getChartData(RMIPerfExplorerModel model, ChartDataType dataType) {
		//PerfExplorerOutput.println("getChartData(" + model.toString() + ")...");
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
		PreparedStatement statement = null;
		try {
			String groupingName = null;
			//String threadName = null;
			double value = 0.0;
			double numThreads = 0;
			String currentExperiment = "";
			int experimentIndex = -1;
			if (dataType == ChartDataType.CORRELATION_DATA) {
				columnValues = new ArrayList<Double>();
				// do a pre-query to get the event with inclusive value
				// of 100.0.
				statement = buildPreQueryStatement();
				//PerfExplorerOutput.println(statement.toString());
				ResultSet results = statement.executeQuery();
				// TODO - this query assumes a scalability study...!
				while (results.next() != false) {
					groupingName = results.getString(1);
					numThreads = results.getDouble(2) * results.getDouble(3) *
						results.getDouble(4);
					value = results.getDouble(5);
					if (metricName.toLowerCase().indexOf("time") != -1)
						value = value/1000000;
	
					if (!currentExperiment.equals(groupingName)) {
						experimentIndex++;
						currentExperiment = groupingName;
						addRow(groupingName);
					}
					addColumn(experimentIndex, numThreads, value);
					columnValues.add(new Double(numThreads));
				} 
				results.close();
				statement.close();
			}
			// all query results are organized the same, only the selection
			// parameters are different.
			statement = buildStatement();
			//PerfExplorerOutput.println(buf.toString());
			ResultSet results = statement.executeQuery();
			// TODO - this query assumes a scalability study...!
			int columnCounter = 0;
			while (results.next() != false) {
				groupingName = results.getString(1);
				if (dataType == ChartDataType.IQR_DATA || dataType == ChartDataType.DISTRIBUTION_DATA) {
					numThreads = results.getDouble(2);
					//threadName = Double.toString(numThreads);
					value = results.getDouble(3);
				} else {
					numThreads = results.getDouble(2) * results.getDouble(3) *
						results.getDouble(4);
					//threadName = Double.toString(numThreads);
					value = results.getDouble(5);
				}
				if ((metricName.toLowerCase().indexOf("time") != -1) 
					&& (dataType != ChartDataType.FRACTION_OF_TOTAL) &&
					!metricName.contains("+") &&
					!metricName.contains("-") &&
					!metricName.contains("*") &&
					!metricName.contains("/"))
					value = value/1000000;

				if (!currentExperiment.equals(groupingName)) {
					experimentIndex++;
					currentExperiment = groupingName;
					addRow(groupingName);
					columnCounter = 0;
				}
				// some methods may not have been called in the
				// base case - MPI methods, for example
				// add 0 values for those processor counts.
				if (dataType == ChartDataType.CORRELATION_DATA) {
					Double mainThreads = columnValues.get(columnCounter);
					while (mainThreads.doubleValue() < numThreads) {
						addColumn(experimentIndex, mainThreads.doubleValue(), 0.0);
						columnCounter++;
						mainThreads = columnValues.get(columnCounter);
					}
					columnCounter++;
				}
				addColumn(experimentIndex, numThreads, value);
			} 
			results.close();
			statement.close();

			// now that we got the main data, get the "other" data.
			statement = buildOtherStatement();
			if (statement != null) {
				results = statement.executeQuery();
				// TODO - this query assumes a scalability study...!
				while (results.next() != false) {
					groupingName = "other";
					numThreads = results.getDouble(1) * results.getDouble(2) *
						results.getDouble(3);
					//threadName = Double.toString(numThreads);
					value = results.getDouble(4);
					if ((metricName.toLowerCase().indexOf("time") != -1) 
						&& (dataType != ChartDataType.FRACTION_OF_TOTAL))
						value = value/1000000;
	
					if (!currentExperiment.equals(groupingName)) {
						experimentIndex++;
						currentExperiment = groupingName;
						addRow(groupingName);
					}
					addColumn(experimentIndex, numThreads, value);
				} 
				results.close();
				statement.close();
			}

			try {
				if ((dataType == ChartDataType.RELATIVE_EFFICIENCY_EVENTS) || 
					(dataType == ChartDataType.CORRELATION_DATA)) {
					DB db = PerfExplorerServer.getServer().getDB();
					if (db.getDBType().compareTo("oracle") == 0) {
						statement = db.prepareStatement("truncate table working_table");
						statement.execute();
						statement.close();
					}
					if ((db.getDBType().compareTo("derby") == 0) ||
					    (db.getDBType().compareTo("db2") == 0))
						statement = db.prepareStatement("drop table SESSION.working_table");
					else
						statement = db.prepareStatement("drop table working_table");
					statement.execute();
					statement.close();
				}
			} catch (Exception e) {
				// do nothing, as all we did was truncate & drop the table
			}
		} catch (Exception e) {
			if (statement != null)
				PerfExplorerOutput.println(statement.toString());
			PerfExplorerOutput.println(buf.toString());
			String error = "ERROR: Couldn't select the analysis settings from the database!";
			System.err.println(error);
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
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
		buf = new StringBuilder();
		Object object = model.getCurrentSelection();
		if (dataType == ChartDataType.FRACTION_OF_TOTAL) {
			// The user wants to know the runtime breakdown by events of one 
			// experiment as the number of threads of execution increases.
			StringBuilder tmpBuf = new StringBuilder();
			buf.append("select ");
			if (db.getDBType().compareTo("db2") == 0) {
				tmpBuf.append(" cast (ie.name as varchar(256)), ");
			} else {
				tmpBuf.append(" ie.name, ");
			}
			buf.append(tmpBuf.toString());
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");
			buf.append(" avg(ims.exclusive_percentage), max(ims.inclusive) ");  // this ensures we get the main routine?
			buf.append("from interval_mean_summary ims ");
			buf.append("inner join interval_event ie ");
			buf.append("on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ? ");
			} else {
				buf.append(" and m.name = ? ");
			}
			buf.append("and ims.exclusive_percentage > ");
			buf.append(model.getXPercent());
			buf.append(" and (ie.group_name is null or (");
			buf.append("ie.group_name not like '%TAU_CALLPATH%' ");
			buf.append("and ie.group_name not like '%TAU_PARAM%' ");
			buf.append("and ie.group_name not like '%TAU_PHASE%')");
			buf.append("or ims.exclusive_percentage = 100.0) ");
			buf.append("group by ");
			buf.append(tmpBuf.toString());
			buf.append(" t.node_count, t.contexts_per_node, t.threads_per_context ");
			buf.append("order by 1, 2, 3, 4");
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
		} else if (dataType == ChartDataType.RELATIVE_EFFICIENCY) {
			// The user wants to know the relative efficiency or speedup
			// of one or more experiments, as the number of threads of 
			// execution increases.
			List<Object> selections = model.getMultiSelection();
			buf.append("select ");
			StringBuilder tmpBuf = new StringBuilder();
			if (object instanceof RMIView) {
				if (isLeafView()) {
					tmpBuf.append(model.getViewSelectionString(db.getDBType()));
				} else {
					tmpBuf.append(groupByColumn);
				}
			} else {
				if (db.getDBType().compareTo("db2") == 0) {
					tmpBuf.append("cast (e.name as varchar(256))");
				} else {
					tmpBuf.append("e.name");
				}
			}
			buf.append(" " + tmpBuf.toString() + ", ");
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");
			buf.append("max(ims.inclusive) from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("inner join experiment e on t.experiment = e.id ");
				buf.append("where t.experiment in (");
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

			// this sucks - can't use 100.0, because of rounding errors. Bah.
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ?");
			} else {
				buf.append(" and m.name = ?");
			}
			buf.append(" group by ");
			buf.append(tmpBuf.toString());
			buf.append(", ");
			buf.append(" t.node_count, t.contexts_per_node, t.threads_per_context ");
			buf.append("order by 1, 2, 3, 4");
			//PerfExplorerOutput.println(buf.toString());
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
		} else if (dataType == ChartDataType.TOTAL_FOR_GROUP) {
			// The user wants to know the percentage of total runtime that
			// comes from one group of events, such as communication or 
			// computation.  This query is done for 
			// one or more experiments, as the number of threads of 
			// execution increases.
			buf.append("select ");
			StringBuilder tmpBuf = new StringBuilder();
			if (object instanceof RMIView) {
				if (isLeafView()) {
					tmpBuf.append(" " + model.getViewSelectionString(db.getDBType()) + ", ");
				} else {
					tmpBuf.append(" " + groupByColumn + ", ");
				}
			} else {
				if (db.getDBType().compareTo("db2") == 0) {
					tmpBuf.append(" cast (e.name as varchar(256)), ");
				} else {
					tmpBuf.append(" e.name, ");
				}
			}
			buf.append(tmpBuf.toString());
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");

			if (db.getDBType().compareTo("oracle") == 0) {
				buf.append("sum(ims.excl) from interval_mean_summary ims ");
			} else {
				buf.append("sum(ims.exclusive) from interval_mean_summary ims ");
			}

			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			buf.append("inner join experiment e on t.experiment = e.id ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, false, db.getDBType()));
			} else {
				buf.append("where t.experiment in (");
				List<Object> selections = model.getMultiSelection();
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

			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ? ");
				buf.append(" and ie.group_name like ? group by ");
			} else {
				buf.append(" and m.name = ? ");
				buf.append(" and ie.group_name = ? group by ");
			}

			buf.append(tmpBuf.toString());
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("cast (ie.group_name as varchar(256)) order by 1, 2, 3, 4");
			} else {
				buf.append("ie.group_name order by 1, 2, 3, 4");
			}

			//PerfExplorerOutput.println(buf.toString());
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
			statement.setString(2, groupName);
		} else if ((dataType == ChartDataType.RELATIVE_EFFICIENCY_EVENTS) ||
			(dataType == ChartDataType.CORRELATION_DATA)) {
			// The user wants to know the relative efficiency or speedup
			// of all the events for one experiment, as the number of threads of 
			// execution increases.
			if (db.getDBType().compareTo("oracle") == 0) {
				buf.append("create global temporary table working_table ");
				buf.append("(name varchar2(4000)) ");
			} else if (db.getDBType().compareTo("derby") == 0) {
				buf.append("declare global temporary table working_table ");
				buf.append("(name varchar(4000)) on commit preserve rows not logged ");
			} else if (db.getDBType().compareTo("db2") == 0) {
				buf.append("declare global temporary table working_table ");
				buf.append("(name varchar(256)) on commit preserve rows not logged ");
			} else {
				buf.append("create temporary table working_table (name text) ");
			}
			try {
				//PerfExplorerOutput.println(buf.toString());
				statement = db.prepareStatement(buf.toString());
				statement.execute();
				statement.close();
			} catch (Exception e) {
				if (statement != null)
					PerfExplorerOutput.println(statement.toString());
				PerfExplorerOutput.println(buf.toString());
				String error = "ERROR: Couldn't select the analysis settings from the database!";
				System.err.println(error);
				System.err.println(e.getMessage());
				e.printStackTrace();
			}

			buf = new StringBuilder();
			buf.append("insert into ");
			if ((db.getDBType().compareTo("derby") == 0) ||
			    (db.getDBType().compareTo("db2") == 0))
				buf.append("SESSION.");
			buf.append("working_table (select distinct ");
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("cast (ie.name as varchar(256)) ");
			} else {
				buf.append("ie.name ");
			}
			buf.append("from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ? ");
			} else {
				buf.append(" and m.name = ? ");
			}
			buf.append("and ims.exclusive_percentage > ");
			buf.append(model.getXPercent());
			buf.append(" and (ie.group_name is null or (");
			buf.append("ie.group_name not like '%TAU_CALLPATH%' ");
			buf.append("and ie.group_name not like '%TAU_PARAM%' ");
			buf.append("and ie.group_name not like '%TAU_PHASE%')");
			buf.append("or ims.exclusive_percentage = 100.0) ");
			buf.append("and ims.inclusive_percentage < 100.0) ");

			//PerfExplorerOutput.println(buf.toString());
			try {
				statement = db.prepareStatement(buf.toString());
				statement.setString(1, metricName);
				//PerfExplorerOutput.println(statement.toString());
				statement.execute();
				statement.close();
			} catch (Exception e) {
				PerfExplorerOutput.println(statement.toString());
				PerfExplorerOutput.println(buf.toString());
				String error = "ERROR: Couldn't select the analysis settings from the database!";
				System.err.println(error);
				System.err.println(e.getMessage());
				e.printStackTrace();
			}

			buf = new StringBuilder();
			buf.append("select distinct ");
			StringBuilder tmpBuf = new StringBuilder();
			if (db.getDBType().compareTo("db2") == 0) {
				tmpBuf.append(" cast (ie.name as varchar(256)), ");
			} else {
				tmpBuf.append(" ie.name, ");
			}
			buf.append(tmpBuf.toString());
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");

			if (dataType == ChartDataType.CORRELATION_DATA) {
				if (db.getDBType().compareTo("oracle") == 0) {
					buf.append("avg(ims.excl) from interval_mean_summary ims ");
				} else {
					buf.append("avg(ims.exclusive) from interval_mean_summary ims ");
				}
			} else {
				if (db.getDBType().compareTo("oracle") == 0) {
					buf.append("avg(ims.excl) from interval_mean_summary ims ");
				} else {
					buf.append("avg(ims.exclusive) from interval_mean_summary ims ");
				}
			}
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			buf.append("inner join ");
			if (db.getDBType().compareTo("derby") == 0)
				buf.append("SESSION.");
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("SESSION.working_table w on w.name = cast (ie.name as varchar(256)) ");
			} else {
				buf.append("working_table w on w.name = ie.name ");
			}
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}

			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ? ");
			} else {
				buf.append(" and m.name = ? ");
			}

			buf.append(" group by ");
			buf.append(tmpBuf.toString());
			buf.append(" t.node_count, t.contexts_per_node, t.threads_per_context ");

			buf.append(" order by 1, 2, 3, 4 ");
			//PerfExplorerOutput.println(buf.toString());
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
		} else if (dataType == ChartDataType.RELATIVE_EFFICIENCY_ONE_EVENT) {
			// The user wants to know the relative efficiency or speedup
			// of one event for one or more experiments, as the number of
			// threads of execution increases.
			buf.append("select ");
			StringBuilder tmpBuf = new StringBuilder();
			if (object instanceof RMIView) {
				if (isLeafView()) {
					tmpBuf.append(" " + model.getViewSelectionString(db.getDBType()) + ", ");
				} else {
					tmpBuf.append(" " + groupByColumn + ", ");
				}
			} else {
				if (db.getDBType().compareTo("db2") == 0) {
					tmpBuf.append(" cast (e.name as varchar(256)), ");
				} else {
					tmpBuf.append(" e.name, ");
				}
			}
			buf.append(tmpBuf.toString());
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");
			if (db.getDBType().compareTo("oracle") == 0) {
				buf.append("avg(ims.excl) from interval_mean_summary ims ");
			} else {
				buf.append("avg(ims.exclusive) from interval_mean_summary ims ");
			}
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("inner join experiment e on t.experiment = e.id ");
				buf.append("where t.experiment in (");
				List<Object> selections = model.getMultiSelection();
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

			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ? ");
				buf.append("and ie.name like ? ");
			} else {
				buf.append(" and m.name = ? ");
				buf.append("and ie.name = ? ");
			}
			
			buf.append(" group by ");
			buf.append(tmpBuf.toString());
			buf.append(" t.node_count, t.contexts_per_node, t.threads_per_context ");

			buf.append(" order by 1, 2, 3, 4");
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
			statement.setString(2, eventName);
		} else if (dataType == ChartDataType.RELATIVE_EFFICIENCY_PHASES) {
			// The user wants to know the relative efficiency or speedup
			// of all the phases for one experiment, as the number of threads of 
			// execution increases.
			buf.append("select ");
			StringBuilder tmpBuf = new StringBuilder();
			if (object instanceof RMIView) {
				if (isLeafView()) {
					//tmpBuf.append(" " + model.getViewSelectionString(db.getDBType()) + ", ");
					if (db.getDBType().compareTo("db2") == 0) {
						tmpBuf.append(" cast (ie.name as varchar(256)), ");
					} else {
						tmpBuf.append(" ie.name, ");
					}
				} else {
					tmpBuf.append(" " + groupByColumn + ", ");
				}
			} else {
				if (db.getDBType().compareTo("db2") == 0) {
					tmpBuf.append(" cast (ie.name as varchar(256)), ");
				} else {
					tmpBuf.append(" ie.name, ");
				}
			}
			buf.append(tmpBuf.toString());
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");
			buf.append("avg(ims.inclusive) from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("where t.experiment = ? ");
			}
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ? ");
			} else {
				buf.append(" and m.name = ? ");
			}

//			buf.append("and ims.inclusive_percentage < 100.0 ");
			buf.append(" and ie.group_name like '%TAU_PHASE%' ");
			buf.append(" and ie.group_name not like '%TAU_CALLPATH%' ");
			buf.append(" and ie.group_name not like '%TAU_PARAM%' ");

			buf.append(" group by ");
			buf.append(tmpBuf.toString());
			buf.append(" t.node_count, t.contexts_per_node, t.threads_per_context ");

			buf.append(" order by 1, 2, 3, 4");
			
			statement = db.prepareStatement(buf.toString());
			if (object instanceof RMIView) {
				statement.setString(1, metricName);
			} else {
				statement.setInt (1, model.getExperiment().getID());
				statement.setString(2, metricName);
			}
		} else if ((dataType == ChartDataType.FRACTION_OF_TOTAL_PHASES) || 
			(dataType == ChartDataType.CORRELATION_DATA)) {
			// The user wants to know the runtime breakdown by phases 
			// of one experiment as the number of threads of execution
			// increases.
			buf.append("select ");
			StringBuilder tmpBuf = new StringBuilder();
			if (object instanceof RMIView) {
				if (isLeafView()) {
					//tmpBuf.append(" " + model.getViewSelectionString(db.getDBType()) + ", ");
					if (db.getDBType().compareTo("db2") == 0) {
						tmpBuf.append(" cast (ie.name as varchar(256)), ");
					} else {
						tmpBuf.append(" ie.name, ");
					}
				} else {
					tmpBuf.append(" " + groupByColumn + ", ");
				}
			} else {
				if (db.getDBType().compareTo("db2") == 0) {
					tmpBuf.append(" cast (ie.name as varchar(256)), ");
				} else {
					tmpBuf.append(" ie.name, ");
				}
			}
			buf.append(tmpBuf.toString());
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");
			buf.append("avg(ims.inclusive_percentage) from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			statement = null;
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}

			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ? ");
			} else {
				buf.append(" and m.name = ? ");
			}

			buf.append("and ims.inclusive_percentage < 100.0 ");
			buf.append("and ie.group_name like '%TAU_PHASE%' ");
			buf.append("and ie.group_name not like '%TAU_CALLPATH%' ");
			buf.append("and ie.group_name not like '%TAU_PARAM%' ");
			buf.append(" group by ");

			buf.append(tmpBuf.toString());
			buf.append(" t.node_count, t.contexts_per_node, t.threads_per_context ");

			buf.append("order by 1, 2, 3, 4");

			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
		} else if (dataType == ChartDataType.IQR_DATA) {
			// The user wants to know the runtime breakdown by phases 
			// of one experiment as the number of threads of execution
			// increases.
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("select cast (ie.name as varchar(256)), ");
			} else {
				buf.append("select ie.name, ");
			}
			buf.append("(p.node * t.contexts_per_node * ");
			buf.append("t.threads_per_context) + (p.context * ");
			buf.append("t.threads_per_context) + p.thread as thread, ");
            
            if (db.getDBType().compareTo("oracle") == 0) {
                buf.append("p.excl ");
            } else {
                buf.append("p.exclusive ");
            }

			buf.append("from interval_event ie ");
			buf.append("inner join interval_mean_summary s ");
			buf.append("on ie.id = s.interval_event and s.exclusive_percentage > ");
			buf.append(model.getXPercent());
			buf.append(" left outer join interval_location_profile p ");
			buf.append("on ie.id = p.interval_event ");
			buf.append("and p.metric = s.metric ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("where ie.trial = ? ");
			buf.append("and p.metric = ? ");
			buf.append("and (ie.group_name is null or (");
			buf.append("ie.group_name not like '%TAU_CALLPATH%' ");
			buf.append("and ie.group_name not like '%TAU_PARAM%' ");
			buf.append("and ie.group_name not like '%TAU_PHASE%')");
			buf.append("or s.exclusive_percentage = 100.0) ");
			buf.append(" order by 1,2 ");
			statement = db.prepareStatement(buf.toString());
			statement.setInt(1, model.getTrial().getID());
			statement.setInt(2, ((Metric)(model.getCurrentSelection())).getID());
		} else if (dataType == ChartDataType.DISTRIBUTION_DATA) {
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("select cast (ie.name as varchar(256)), ");
			} else {
				buf.append("select ie.name, ");
			}
			buf.append("(p.node * t.contexts_per_node * ");
			buf.append("t.threads_per_context) + (p.context * ");
			buf.append("t.threads_per_context) + p.thread as thread, ");
            
            if (db.getDBType().compareTo("oracle") == 0) {
                buf.append("p.excl ");
            } else {
                buf.append("p.exclusive ");
            }

			buf.append("from interval_event ie ");
			buf.append(" left outer join interval_location_profile p ");
			buf.append("on ie.id = p.interval_event ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("where ie.trial = ? ");
			buf.append("and p.metric = ? ");
			buf.append("and ie.id in (");
			List<Object> selections = model.getMultiSelection();
			if (selections == null) {
				// just one selection
				buf.append (model.getEvent().getID());
			} else {
				for (int i = 0 ; i < selections.size() ; i++) {
					IntervalEvent event = (IntervalEvent)selections.get(i);
					if (i > 0)
						buf.append(",");
					buf.append(event.getID());
				}
			}
			buf.append(") order by 1,2 ");
			statement = db.prepareStatement(buf.toString());
			statement.setInt(1, model.getTrial().getID());
			statement.setInt(2, model.getMetric().getID());
			System.out.println(statement.toString());
		}
		return statement;
	}

	private PreparedStatement buildOtherStatement () throws SQLException {
			DB db = PerfExplorerServer.getServer().getDB();

		PreparedStatement statement = null;
		buf = new StringBuilder();
		Object object = model.getCurrentSelection();
		if (dataType == ChartDataType.FRACTION_OF_TOTAL) {
			// The user wants to know the runtime breakdown by events of one 
			// experiment as the number of threads of execution increases.
			buf.append("select ");
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");
			buf.append("sum(ims.exclusive_percentage) from interval_mean_summary ims ");
			buf.append("inner join interval_event ie ");
			buf.append("on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ? ");
			} else {
				buf.append(" and m.name = ? ");
			}
			buf.append("and ims.inclusive_percentage < 100.0 ");
			buf.append("and ims.exclusive_percentage < ");
			buf.append(model.getXPercent());
			buf.append(" and (ie.group_name is null or (");
			buf.append("ie.group_name not like '%TAU_CALLPATH%' ");
			buf.append("and ie.group_name not like '%TAU_PARAM%' ");
			buf.append("and ie.group_name not like '%TAU_PHASE%')");
			buf.append("or ims.exclusive_percentage = 100.0) group by t.node_count, t.contexts_per_node, t.threads_per_context order by 1, 2, 3");
			
			//PerfExplorerOutput.println(buf.toString());
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);

		} else if (dataType == ChartDataType.RELATIVE_EFFICIENCY_EVENTS) {
			// The user wants to know the relative efficiency or speedup
			// of all the events for one experiment, as the number of threads of 
			// execution increases.
			/*
			buf.append("insert into (select distinct ");
			buf.append("ie.name from interval_mean_summary ims ");
			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}
			buf.append(" and m.name = ? ");
			buf.append("and ims.exclusive_percentage > ");
			buf.append(model.getXPercent());
			buf.append(" and (ie.group_name is null or (");
			buf.append("ie.group_name not like '%TAU_CALLPATH%' ");
			buf.append("and ie.group_name not like '%TAU_PARAM%' ");
			buf.append("and ie.group_name not like '%TAU_PHASE%'));");
			*/

			buf.append("select ");
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");

			if (db.getDBType().compareTo("oracle") == 0) {
				buf.append("sum(ims.excl) from interval_mean_summary ims ");
			} else {
				buf.append("sum(ims.exclusive) from interval_mean_summary ims ");
			}

			buf.append("inner join interval_event ie on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			buf.append("left outer join ");
			if (db.getDBType().compareTo("derby") == 0)
				buf.append("SESSION.");
			if (db.getDBType().compareTo("db2") == 0)
				buf.append("SESSION.working_table w on w.name = cast(ie.name as varchar(256)) ");
			else
				buf.append("working_table w on w.name = ie.name ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}
			
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ? ");
			} else {
				buf.append(" and m.name = ? ");
			}
			buf.append("and w.name is null ");
			buf.append("group by t.node_count, t.contexts_per_node, t.threads_per_context order by 1, 2, 3 ");

			//PerfExplorerOutput.println(buf.toString());
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);

		}
		return statement;
	}


	private PreparedStatement buildPreQueryStatement () throws SQLException {
		DB db = PerfExplorerServer.getServer().getDB();
		PreparedStatement statement = null;
		buf = new StringBuilder();
		Object object = model.getCurrentSelection();
		if (dataType == ChartDataType.CORRELATION_DATA) {
			// The user wants to know the runtime breakdown by events of one 
			// experiment as the number of threads of execution increases.
			buf.append("select 'TOTAL', ");
			buf.append("t.node_count, t.contexts_per_node, t.threads_per_context, ");
			buf.append(" max(ims.inclusive) ");
			buf.append("from interval_mean_summary ims ");
			buf.append("inner join interval_event ie ");
			buf.append("on ims.interval_event = ie.id ");
			buf.append("inner join trial t on ie.trial = t.id ");
			buf.append("inner join metric m on m.id = ims.metric ");
			if (object instanceof RMIView) {
				buf.append(model.getViewSelectionPath(true, true, db.getDBType()));
			} else {
				buf.append("where t.experiment = ");
				buf.append(model.getExperiment().getID() + " ");
			}
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append(" and m.name like ? ");
			} else {
				buf.append(" and m.name = ? ");
			}
			//buf.append("and ims.exclusive_percentage > ");
			//buf.append(model.getXPercent());
			buf.append(" and (ie.group_name is null or (");
			buf.append("ie.group_name not like '%TAU_CALLPATH%' ");
			buf.append("and ie.group_name not like '%TAU_PARAM%' ");
			buf.append("and ie.group_name not like '%TAU_PHASE%')");
			buf.append("or ims.exclusive_percentage = 100.0) ");
			buf.append("group by t.node_count, t.contexts_per_node, t.threads_per_context order by 2, 3, 4 ");
			statement = db.prepareStatement(buf.toString());
			statement.setString(1, metricName);
		}
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
			System.err.println(error);
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
		return returnValue;
	}
}

