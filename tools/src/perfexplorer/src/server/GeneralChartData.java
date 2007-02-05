package server;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import common.*;

import common.ChartDataType;
import common.PerfExplorerOutput;
import common.RMIChartData;
import common.RMIPerfExplorerModel;
import common.RMIView;

import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.IntervalEvent;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import java.util.ArrayList;
import java.util.List;

/**
 * The GeneralChartData class is used to select data from the database which 
 * represents the performance profile of the selected trials, and return them
 * in a format for JFreeChart to display them.
 *
 * <P>CVS $Id: GeneralChartData.java,v 1.1 2007/02/05 22:59:04 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.2
 * @since   0.2
 */
public class GeneralChartData extends RMIGeneralChartData {

	private RMIPerfExplorerModel model;
	private String metricName = null;
	private String groupName = null;
	private String eventName = null;
	private String groupByColumn = null;
	private List columnValues = null;
	private StringBuffer buf = null;
	
	/**
	 * Constructor
	 * 
	 * @param model
	 * @param dataType
	 */
	public GeneralChartData (RMIPerfExplorerModel model, ChartDataType dataType) {
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
	public static GeneralChartData getChartData(RMIPerfExplorerModel model, 
			ChartDataType dataType) {
		PerfExplorerOutput.println("getChartData(" + model.toString() + ")...");
		GeneralChartData chartData = new GeneralChartData(model, dataType);
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
		// declare the statement here, so we can reference it in the catch
		// region, if necessary
		PreparedStatement statement = null;
		try {
			DB db = PerfExplorerServer.getServer().getDB();

			Object object = model.getCurrentSelection();

			// create and populate the temporary trial table
			buf = buildCreateTableStatement("temp_trial", db);
   			buf.append("(select trial.* from trial ");
	        buf.append("inner join experiment ");
			buf.append("on trial.experiment = experiment.id ");
			buf.append("inner join application ");
			buf.append("on experiment.application = application.id ");
			buf.append("where ");
			// add the where clause
			buf.append("experiment = 138 ");
			buf.append("or experiment = 121 ");
			buf.append("or experiment = 125) ");
			statement = db.prepareStatement(buf.toString());
			System.out.println(statement.toString());
			statement.execute();
			statement.close();

			// create and populate the temporary metric table
			buf = buildCreateTableStatement("temp_metric", db);
    		buf.append("(select metric.* from metric ");
			buf.append("inner join temp_trial ");
			buf.append("on metric.trial = temp_trial.id ");
			// add the where clause
			if (db.getDBType().compareTo("db2") == 0) {
				buf.append("where metric.name like ?) ");
			} else {
				buf.append("where metric.name = ?) ");
			}
			statement = db.prepareStatement(buf.toString());
			//statement.setString(1, metricName);
			statement.setString(1, "WALL_CLOCK_TIME");
			System.out.println(statement.toString());
			statement.execute();
			statement.close();

			// create and populate the temporary event table
			buf = buildCreateTableStatement("temp_event", db);
			buf.append("(select interval_event.* from interval_event ");
			buf.append("inner join temp_trial ");
			buf.append("on interval_event.trial = temp_trial.id ");
			buf.append("inner join interval_mean_summary ");
   			buf.append("on interval_mean_summary.interval_event = interval_event.id ");
			buf.append("inner join temp_metric ");
			buf.append("on interval_mean_summary.metric = temp_metric.id ");
			buf.append("where interval_mean_summary.inclusive_percentage = 100 ");
			buf.append("and (group_name is null ");
        	buf.append("or (group_name not like '%TAU_CALLPATH%' ");
            buf.append("and group_name not like '%TAU_PHASE%'))) ");
			statement = db.prepareStatement(buf.toString());
			System.out.println(statement.toString());
			statement.execute();
			statement.close();

			// The user wants parametric study data, with the data
			// organized with two axes, the x and the y.
			// unlike scalability: 
			// NO ASSUMPTION ABOUT WHICH COLUMN IS THE Y AXIS!
			String seriesName = new String("experiment.name");
			String xAxisName = new String("node_count * contexts_per_node * threads_per_context");
			String yAxisName = new String("interval_mean_summary.inclusive");
			buf = new StringBuffer();
			buf.append("select ");
			// first item - the series name
			buf.append(seriesName + ", ");
			// second item - the x axis
			buf.append(xAxisName + ", ");
			// second item - the y axis
			buf.append(yAxisName);
			buf.append(" from interval_mean_summary ");
			buf.append("inner join temp_metric ");
			buf.append("on interval_mean_summary.metric = temp_metric.id ");
			buf.append("inner join temp_event ");
			buf.append("on interval_mean_summary.interval_event = temp_event.id ");
			buf.append("inner join temp_trial ");
			buf.append("on temp_event.trial = temp_trial.id ");
			buf.append("inner join experiment ");
			buf.append("on temp_trial.experiment = experiment.id ");
			buf.append("inner join application ");
			buf.append("on experiment.application = application.id ");
			buf.append("order by 1, 2 ");
			statement = db.prepareStatement(buf.toString());
			System.out.println(statement.toString());
			ResultSet results = statement.executeQuery();

			int columnCounter = 0;
			while (results.next() != false) {
				System.out.println(results.getString(1) + ": " + results.getString(2) + ", " + results.getDouble(3));
				addRow(results.getString(1), results.getString(2), results.getDouble(3));
			} 
			results.close();
			statement.close();

		} catch (Exception e) {
			if (statement != null)
				PerfExplorerOutput.println(statement.toString());
			PerfExplorerOutput.println(buf.toString());
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
	}

	private StringBuffer buildCreateTableStatement (String tableName, DB db) {
		StringBuffer buf = new StringBuffer();
		if (db.getDBType().equalsIgnoreCase("oracle")) {
			buf.append("create global ");
		} else if ((db.getDBType().equalsIgnoreCase("derby")) ||
					(db.getDBType().equalsIgnoreCase("db2"))) {
			buf.append("declare global ");
		} else {
			buf.append("create ");
		}
		buf.append("temporary table ");
		buf.append(tableName);
		buf.append(" as ");
		return buf;
	}
}

