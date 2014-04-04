package edu.uoregon.tau.perfexplorer.server;

import java.io.InputStream;
import java.io.Reader;
import java.io.StringReader;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import edu.uoregon.tau.common.Gzip;
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.AtomicEvent;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.IntervalEvent;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfexplorer.common.ChartDataType;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;
import edu.uoregon.tau.perfexplorer.common.RMIPerfExplorerModel;
import edu.uoregon.tau.perfexplorer.common.TransformationType;

/**
 * The GeneralChartData class is used to select data from the database which 
 * represents the performance profile of the selected trials, and return them
 * in a format for JFreeChart to display them.
 *
 * <P>CVS $Id: GeneralChartData.java,v 1.38 2009/08/04 22:19:01 wspear Exp $</P>
 * @author  Kevin Huck
 * @version 0.2
 * @since   0.2
 */
public class TAUdbGeneralChartData extends GeneralChartData {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6775990055590146350L;
	private RMIPerfExplorerModel model;
	private StringBuilder buf = null;

	/**
	 * Constructor
	 * 
	 * @param model
	 * @param dataType
	 */
	public TAUdbGeneralChartData (RMIPerfExplorerModel model, ChartDataType dataType) {
		super (dataType);
		this.model = model;
		doQuery();
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
		//System.out.println("doQuery");
		PreparedStatement statement = null;
		DB db = null;
		try {
			db = PerfExplorerServer.getServer().getDB();

			String listOfTrials = getListOfTrials();
			//System.out.println("Trial list: "+listOfTrials);
			
			String listOfMetrics = getListOfMetrics();
			//System.out.println("Metric list: "+listOfMetrics);
			
			String seriesName = model.getChartSeriesName();
			if(seriesName.equals("interval_event.name")) seriesName = "timer.name";
			boolean seriesXML=model.isChartSeriesXML();
			String xAxisName = model.getChartXAxisName();
			String yAxisNameWithStat = model.getChartYAxisName();
			//For now, decode the Y Axis Name 
			int openIndex = yAxisNameWithStat.indexOf("(");
			int dotIndex = yAxisNameWithStat.indexOf(".");
			String yAxisStat = yAxisNameWithStat.substring(0,openIndex);
			String yAxisValue = yAxisNameWithStat.substring(dotIndex+1, yAxisNameWithStat.length()-1);
			if(yAxisNameWithStat.contains("total")){
				yAxisStat = "sum";
			}
			
			if(yAxisValue.equals("inclusive")) yAxisValue="inclusive_value";
			if(yAxisValue.equals("exclusive")) yAxisValue="exclusive_value";
			if(yAxisValue.equalsIgnoreCase("INCLUSIVE_PERCENTAGE")) yAxisValue = "inclusive_percent";
			if (yAxisValue.equalsIgnoreCase("EXCLUSIVE_PERCENTAGE"))
				yAxisValue = "exclusive_percent";

			String yAxisName = "timer_value";
			
			buf = new StringBuilder();
			String metaDataFieldName = null;
			if (xAxisName.equals("primary_metadata.value")) {
				metaDataFieldName = model.getChartMetadataFieldName();
			}
		    
/*
 *So, this is a big if/else block. I might think that it would be easier to build the query in pieces.
 *All that does is confuse the code, because it is impossible to piece together the full SQL statement
 *without running the code.  You would have to do this once for each possible query.  This way you can
 *see all the possible queries on after the other in the separate functions.  
 *
 *Also, if the series name is primary metadata only then is that tabled joined in.  The join with the 
 *metadata may waste time.  (I haven't tested to see how much time on average.
 */
			boolean dimensionReduction = (model.getDimensionReduction() != null && model.getDimensionReduction()== TransformationType.OVER_X_PERCENT);
//			if(model.getMainEventOnly() && metaDataFieldName != null){
//				buf.append(getMainOnlyQueryWitMetaData(db, listOfTrials, listOfMetrics,getDervivedThreadRank(yAxisStat), metaDataFieldName,
//						 seriesName,  xAxisName,  yAxisName,  yAxisStat, yAxisValue));
//			}else 
			if(model.getMainEventOnly() ){
				buf.append(getMainOnlyQuery(db, listOfTrials, listOfMetrics,getDervivedThreadRank(yAxisStat),metaDataFieldName,
						seriesName, seriesXML,  xAxisName,  yAxisName,  yAxisStat, yAxisValue));
			}else if(model.getEventNames()==null  && model.getEventNoCallpath()) {
				buf.append(getAllEventsNoCallpathMetadata(db, listOfTrials, listOfMetrics, getDervivedThreadRank(yAxisStat),  metaDataFieldName,
						seriesName, seriesXML,  xAxisName,  yAxisName,  yAxisStat, yAxisValue));
		
			}else if(model.getEventNames()==null && !model.getEventNoCallpath()) {
				buf.append(getAllEventsCallpathMetadata(db, listOfTrials, listOfMetrics, getDervivedThreadRank(yAxisStat),metaDataFieldName,
						seriesName, seriesXML,  xAxisName,  yAxisName,  yAxisStat, yAxisValue));
			}else if (model.getEventNoCallpath()) {
				buf.append(getSomeEventsNoCallpathMetadata(db, listOfTrials, listOfMetrics, getDervivedThreadRank(yAxisStat),metaDataFieldName,
						seriesName, seriesXML,  xAxisName,  yAxisName,  yAxisStat, yAxisValue));
			} else  if (!model.getEventNoCallpath()) {
				buf.append(getSomeEventsCallpathMetadata(db, listOfTrials, listOfMetrics, getDervivedThreadRank(yAxisStat),metaDataFieldName,
						seriesName, seriesXML,  xAxisName,  yAxisName,  yAxisStat, yAxisValue));
			}
			
//			else if(model.getEventNames()==null && model.getEventNoCallpath()) {
//					buf.append(getAllEventsNoCallpath(db, listOfTrials, listOfMetrics, getDervivedThreadRank(yAxisStat),
//							seriesName, seriesXML,  xAxisName,  yAxisName,  yAxisStat, yAxisValue));
//			}else if(model.getEventNames()==null && !model.getEventNoCallpath()) {
//					buf.append(getAllEventsCallpath(db, listOfTrials, listOfMetrics, getDervivedThreadRank(yAxisStat),
//							seriesName, seriesXML,  xAxisName,  yAxisName,  yAxisStat, yAxisValue));
//			}else if (model.getEventNoCallpath()) {
//					buf.append(getSomeEventsNoCallpath(db, listOfTrials, listOfMetrics, getDervivedThreadRank(yAxisStat),
//							seriesName, seriesXML,  xAxisName,  yAxisName,  yAxisStat, yAxisValue));
//			} else  if (!model.getEventNoCallpath()) {
//					buf.append(getSomeEventsCallpath(db, listOfTrials, listOfMetrics, getDervivedThreadRank(yAxisStat),
//							seriesName, seriesXML,  xAxisName,  yAxisName,  yAxisStat, yAxisValue));
//			}

		//So I know I said don't do this, but this one is simple enough to do once.
		//Also, I don't think this is used too much
			if (!model.getMainEventOnly()  && dimensionReduction && yAxisValue.contains("inclusive")) {
				buf.append(" where timer_value.inclusive_percent > " + model.getXPercent());
			}else if(!model.getMainEventOnly()  && dimensionReduction) {
				buf.append(" where timer_value.exclusive_percent > " + model.getXPercent());
			}
		
			
//			//End which events?
//			
//			buf.append(")");
			buf.append(" group by series_name, xaxis_value ");
			// add the order by clause
			buf.append("order by 1, 2 ");


			statement = db.prepareStatement(buf.toString());

			//System.out.println(statement.toString());
			ResultSet results = statement.executeQuery();

			while (results.next()) {
				// System.out.print(results.getString(2) + ", " );
				// System.out.println(results.getDouble(3));
				if (db.getDBType().compareTo("derby") == 0) {
					if (seriesName.startsWith("trial.node_count")) {
						addRow(Integer.toString(results.getInt(1) * results.getInt(2) * results.getInt(3)), results.getString(4), results.getDouble(5));
					} else if (xAxisName.startsWith("trial.node_count")) {
						addRow(results.getString(1), Integer.toString(results.getInt(2) * results.getInt(3) * results.getInt(4)), results.getDouble(5));
					} else {
						addRow(results.getString(1), results.getString(2), results.getDouble(3));
					}
				} else {
					addRow(results.getString(1), results.getString(2), results.getDouble(3));
				}
			} 
			results.close();
			statement.close();

		} catch (Exception e) {
			if (statement != null)
				PerfExplorerOutput.println(statement.toString());
			else 
				PerfExplorerOutput.println(buf.toString());
			System.err.println(e.getMessage());
			e.printStackTrace();
		} finally {
			try {
				db.setAutoCommit(true);
				dropTable(db, "temp_event");
				if (!model.getChartSeriesName().equals("atomic_event.name")) {
					dropTable(db, "temp_metric");
				}
				dropTable(db, "temp_xml_metadata");
				dropTable(db, "temp_trial");
			} catch (Exception e) {}
		}
	}

	private int getDervivedThreadRank(String stat) {
		/*
		 * We could query this information, but we don't want to waste the time.
		 * Just use the standard schema values
		 * VALUES (-1, 'MEAN', 'MEAN (nulls ignored)');
		 * VALUES (-2, 'TOTAL', 'TOTAL');
		 * VALUES (-4, 'MIN', 'MIN');
		 * VALUES (-5, 'MAX', 'MAX');
		 * VALUES (-6, 'MEAN', 'MEAN (nulls are 0 value)');
		 */
		if(stat.equals("mean")){
			return -1;
		}else if(stat.equals("total")|| stat.equals("sum")){
			return -2;
		}else if(stat.equals("min")){
			return -4;
		}else if(stat.equals("max")){
			return -5;
		}else if(stat.equals("avg")){
			return -6;
		}
		return 0;
	}

	private String getListOfMetrics() {
		String listOfMetrics="( ";
		List<String> metricNames = model.getMetricNames();
		for(String metric : metricNames){
			listOfMetrics += "\'"+metric +"\',";
		}
		listOfMetrics = listOfMetrics.substring(0, listOfMetrics.length()-1);
		listOfMetrics += " )";
		return listOfMetrics;
	}

	private String getListOfTrials() {
		String listOfTrials = "";
		List<Object> selections = model.getMultiSelection();
		if (selections == null) {
			Object obj = model.getCurrentSelection();
			if (obj instanceof Trial) {
				listOfTrials = "( "+ model.getTrial().getID()+" )";
			} else if (obj instanceof View) {
				View view = (View) obj;
				listOfTrials = view.getTrialID();
			}
		} else {
			listOfTrials = "( ";
			for(Object obj: selections){
				if (obj instanceof Trial) {
					listOfTrials += ((Trial)obj).getID() + " , ";
				} else {
					System.err.println("We don't support mutli select of Views (yet?)");
				}
			}
			listOfTrials = listOfTrials.substring(0, listOfTrials.length()-3);
			listOfTrials += " )";
		}
		return listOfTrials;
	}

	private String getMainOnlyQuery(DB db, String listOfTrials, String
	listOfMetrics, int derivedThread, String metadata, String seriesName,
	boolean seriesXML, String xAxisName, String yAxisName, String yAxisStat,
	String yAxisValue) throws SQLException {
			if(yAxisStat.equals("mean")){ yAxisStat = "avg"; };
			StringBuffer buffer = new StringBuffer();
			if(seriesXML)
			{
				buffer.append("select "+"series_metadata.value" +" as series_name, ");
			}
			else{
				buffer.append("select "+seriesName +" as series_name, ");
			}
		    buffer.append(xAxisName + " as xaxis_value, ");
		    buffer.append(yAxisStat+"("+yAxisName+"."+yAxisValue+") as yaxis_value from  ");
			buffer.append(" (select trial.id,  max(timer_value.inclusive_value) as maxinc ");
			buffer.append(" from  timer_value");
			buffer.append(" inner join metric on metric.name in "+listOfMetrics+" and timer_value.metric=metric.id ");
			buffer.append(" inner join thread on thread.thread_rank = "+derivedThread);
			buffer.append(" inner join trial on thread.trial=trial.id and thread.trial in "+listOfTrials);
			buffer.append(" inner join timer_call_data on timer_call_data.thread = thread.id   and  timer_call_data.id = timer_value.timer_call_data ");
			buffer.append(" inner join timer_callpath on timer_callpath.id=timer_call_data.timer_callpath ");
			buffer.append(" inner join timer on timer_callpath.timer = timer.id       ");
			buffer.append(" group by trial.id) as getmax");
			buffer.append(" ,timer_value");
			buffer.append(" inner join metric on metric.name in "+listOfMetrics+" and timer_value.metric=metric.id ");
			buffer.append(" inner join thread on thread.thread_rank = " +derivedThread);
			buffer.append(" inner join trial on thread.trial=trial.id and thread.trial in "+listOfTrials);
			buffer.append(" inner join timer_call_data on timer_call_data.thread = thread.id   and  timer_call_data.id = timer_value.timer_call_data ");
			buffer.append(" inner join timer_callpath on timer_callpath.id=timer_call_data.timer_callpath ");
			buffer.append(" inner join timer on timer_callpath.timer = timer.id   ");
			if(metadata!=null&&metadata.length()>0){
				buffer.append("  inner join primary_metadata on primary_metadata.trial=trial.id and primary_metadata.name = \'"+metadata+'\'');
			}
			if(seriesXML){
				buffer.append("  inner join primary_metadata as series_metadata on series_metadata.trial=trial.id and series_metadata.name = \'"+seriesName+'\'');
			}
			buffer.append(" where timer_value.inclusive_value = maxinc ");
			//System.out.println(buffer.toString());
			return buffer.toString();		
	}

	private String getAllEventsCallpathMetadata(DB db, String listOfTrials,
	String listOfMetrics, int derivedThread, String metadata, String
	seriesName, boolean seriesXML, String xAxisName, String yAxisName, String
	yAxisStat, String yAxisValue) throws SQLException {
		if(yAxisStat.equals("mean")){ yAxisStat = "avg"; };
		StringBuffer buffer = new StringBuffer();
		if(seriesXML)
		{
			buffer.append("select "+"series_metadata.value" +" as series_name, ");
		}
		else{
			buffer.append("select "+seriesName +" as series_name, ");
		}
	    buffer.append(xAxisName + " as xaxis_value, ");
		buffer.append(yAxisStat+"("+yAxisName+"."+yAxisValue+") as yaxis_value from  ");

		buffer.append(" timer_value");
		buffer.append(" inner join metric on metric.name in "+listOfMetrics+" and timer_value.metric=metric.id ");
		buffer.append(" inner join thread on thread.thread_rank = " +derivedThread);
		buffer.append(" inner join trial on thread.trial=trial.id and thread.trial in "+listOfTrials);
		buffer.append(" inner join timer_call_data on timer_call_data.thread = thread.id   and  timer_call_data.id = timer_value.timer_call_data ");
		buffer.append(" inner join timer_callpath on timer_callpath.id=timer_call_data.timer_callpath ");
		buffer.append(" inner join timer on timer_callpath.timer = timer.id   ");
		if(metadata!=null&&metadata.length()>0){
			buffer.append("  inner join primary_metadata on primary_metadata.trial=trial.id and primary_metadata.name = \'"+metadata+'\'');
		}
		if(seriesXML){
			buffer.append("  inner join primary_metadata as series_metadata on series_metadata.trial=trial.id and series_metadata.name = \'"+seriesName+'\'');
		}


		return buffer.toString();		
    }

	private String getAllEventsNoCallpathMetadata(DB db, String listOfTrials,
	String listOfMetrics, int derivedThread, String metadata, String
	seriesName, boolean seriesXML, String xAxisName, String yAxisName, String
	yAxisStat, String yAxisValue) throws SQLException {
		if(yAxisStat.equals("mean")){ yAxisStat = "avg"; };
		StringBuffer buffer = new StringBuffer();
		if(seriesXML)
		{
			buffer.append("select "+"series_metadata.value" +" as series_name, ");
		}
		else{
			buffer.append("select "+seriesName +" as series_name, ");
		}
	    buffer.append(xAxisName + " as xaxis_value, ");
		buffer.append(yAxisStat+"("+yAxisName+"."+yAxisValue+") as yaxis_value from  ");

		buffer.append(" timer_value");
		buffer.append(" inner join metric on metric.name in "+listOfMetrics+" and timer_value.metric=metric.id ");
		buffer.append(" inner join thread on thread.thread_rank = " +derivedThread);
		buffer.append(" inner join trial on thread.trial=trial.id and thread.trial in "+listOfTrials);
		buffer.append(" inner join timer_call_data on timer_call_data.thread = thread.id   and  timer_call_data.id = timer_value.timer_call_data ");
		buffer.append(" inner join timer_callpath on timer_callpath.id=timer_call_data.timer_callpath and timer_callpath.parent is null");
		buffer.append(" inner join timer on timer_callpath.timer = timer.id   ");
		if(metadata!=null&&metadata.length()>0)
		{
			buffer.append("  inner join primary_metadata on primary_metadata.trial=trial.id and primary_metadata.name = \'"+metadata+'\'');
		}
		if(seriesXML){
			buffer.append("  inner join primary_metadata as series_metadata on series_metadata.trial=trial.id and series_metadata.name = \'"+seriesName+'\'');
		}


		return buffer.toString();		
	}

	private String getSomeEventsCallpathMetadata(DB db, String listOfTrials,
	String listOfMetrics, int derivedThread, String metadata, String
	seriesName, boolean seriesXML, String xAxisName, String yAxisName, String
	yAxisStat, String yAxisValue) throws SQLException {
	if(yAxisStat.equals("mean")){ yAxisStat = "avg"; };
		if(seriesName.equals("timer.name")) seriesName = "callpath.name";

		StringBuffer buffer = new StringBuffer();
		StringBuilder listOfEvents = new StringBuilder();
		listOfEvents.append("( ");
		for(String event : model.getEventNames()){
			listOfEvents.append("\'"+event+"\',");
		}
		String events = listOfEvents.substring(0, listOfEvents.length()-1) + ") ";
		

		buffer.append(" with recursive callpath (id, parent, timer, name) as (  "); 
		buffer.append(" SELECT tc.id, tc.parent, tc.timer, timer.name  "); 
		buffer.append(" FROM  timer_callpath tc inner join timer on tc.timer = timer.id where timer.trial in "+listOfTrials+" and tc.parent is null "); 
		buffer.append(" UNION ALL "); 
		buffer.append(" SELECT d.id, d.parent, d.timer,  concat(callpath.name, ' => ', dt.name)  "); //TODO: concat does not exist in postgresql maybe try something like "callpath.name||' => '||dt.name"
		buffer.append(" FROM timer_callpath AS d JOIN callpath ON (d.parent = callpath.id) join timer dt on d.timer = dt.id where dt.trial in "+listOfTrials+" ) "); 

		if(seriesXML)
		{
			buffer.append("select "+"series_metadata.value" +" as series_name, ");
		}
		else{
			buffer.append("select "+seriesName +" as series_name, ");
		}
	    buffer.append(xAxisName + " as xaxis_value, ");
		buffer.append(yAxisStat+"("+yAxisName+"."+yAxisValue+") as yaxis_value from  ");
	    buffer.append(" timer_value ");
		buffer.append(" inner join metric on metric.name in "+listOfMetrics+" and timer_value.metric=metric.id   "); 
		buffer.append(" inner join thread on thread.thread_rank = "+derivedThread); 
		buffer.append(" inner join trial on thread.trial=trial.id and thread.trial in "+listOfTrials); 
		buffer.append(" inner join timer_call_data on timer_call_data.thread = thread.id   and  timer_call_data.id = timer_value.timer_call_data   "); 
		buffer.append(" inner join timer_callpath on timer_callpath.id=timer_call_data.timer_callpath  "); 
		buffer.append(" inner join callpath on  callpath.id = timer_callpath.id and callpath.name in "+events+"   "); 
		buffer.append(" inner join timer on timer_callpath.timer = timer.id  " );
		if(metadata!=null&&metadata.length()>0){
			buffer.append("  inner join primary_metadata on primary_metadata.trial=trial.id and primary_metadata.name = \'"+metadata+'\'');
		}
		if(seriesXML){
			buffer.append("  inner join primary_metadata as series_metadata on series_metadata.trial=trial.id and series_metadata.name = \'"+seriesName+'\'');
		}

//				buffer.append(	"  group by series_name, xaxis_value order by 1, 2  "); 
		return buffer.toString();		
	}
	private String getSomeEventsNoCallpathMetadata(DB db, String listOfTrials, String listOfMetrics, int derivedThread, String metadata,
			String seriesName, boolean seriesXML, String xAxisName, String yAxisName, String yAxisStat, String yAxisValue) throws SQLException {
	    if(yAxisStat.equals("mean")){ yAxisStat = "avg"; };
		StringBuffer buffer = new StringBuffer();
		StringBuilder listOfEvents = new StringBuilder();
		listOfEvents.append("( ");
		for(String event : model.getEventNames()){
			listOfEvents.append("\'"+event+"\',");
		}
		String events = listOfEvents.substring(0, listOfEvents.length()-1) + ") ";

		if(seriesXML)
		{
			buffer.append("select "+"series_metadata.value" +" as series_name, ");
		}
		else{
			buffer.append("select "+seriesName +" as series_name, ");
		}
	    buffer.append(xAxisName + " as xaxis_value, ");
		buffer.append(yAxisStat+"("+yAxisName+"."+yAxisValue+") as yaxis_value from  ");
		
		buffer.append(" timer_value");
		buffer.append(" inner join metric on metric.name in "+listOfMetrics+" and timer_value.metric=metric.id ");
		buffer.append(" inner join thread on thread.thread_rank = " +derivedThread);
		buffer.append(" inner join trial on thread.trial=trial.id and thread.trial in "+listOfTrials);
		buffer.append(" inner join timer_call_data on timer_call_data.thread = thread.id   and  timer_call_data.id = timer_value.timer_call_data ");
		buffer.append(" inner join timer_callpath on timer_callpath.id=timer_call_data.timer_callpath and timer_callpath.parent is null");
		buffer.append(" inner join timer on timer.name in "+events+" and timer_callpath.timer = timer.id   ");
		if(metadata!=null&&metadata.length()>0){
			buffer.append("  inner join primary_metadata on primary_metadata.trial=trial.id and primary_metadata.name = \'"+metadata+'\'');
		}
		if(seriesXML){
			buffer.append("  inner join primary_metadata as series_metadata on series_metadata.trial=trial.id and series_metadata.name = \'"+seriesName+'\'');
		}

		return buffer.toString();		
    }
	
	private static StringBuilder buildCreateTableStatement (String oldTableName, String tableName, DB db, boolean appendAs, boolean doAtomic) {
		// just in case, drop the table in case it is still hanging around.
		// This sometimes happens with Derby.
		// Have I ever mentioned that Derby sucks?
		dropTable(db, tableName);

		StringBuilder buf = new StringBuilder();
		if (db.getDBType().equalsIgnoreCase("oracle")) {
			buf.append("create global temporary table ");
		} else if ((db.getDBType().equalsIgnoreCase("derby")) ||
				(db.getDBType().equalsIgnoreCase("db2"))) {
			buf.append("create table ");
		} else {
			buf.append("create temporary table ");
		}
		buf.append(tableName + " ");
		if (appendAs) {
			if (db.getDBType().equalsIgnoreCase("derby")) {
				String[] names = null;
				String[] types = null;
				// get the table definition
				if (oldTableName.equalsIgnoreCase("trial")) {
					Trial.getMetaData(db, true);
					names = db.getDatabase().getTrialFieldNames();
					types = db.getDatabase().getTrialFieldTypeNames();
					Trial.getMetaData(db, false);
				} else if (oldTableName.equalsIgnoreCase("metric")) {
					Metric.getMetaData(db);
					names = db.getDatabase().getMetricFieldNames();
					types = db.getDatabase().getMetricFieldTypeNames();
				} else if (oldTableName.equalsIgnoreCase("event")) {
					if (doAtomic) {
						AtomicEvent.getMetaData(db);
						names = db.getDatabase().getAtomicEventFieldNames();
						types = db.getDatabase().getAtomicEventFieldTypeNames();
					} else {
						IntervalEvent.getMetaData(db);
						names = db.getDatabase().getIntervalEventFieldNames();
						types = db.getDatabase().getIntervalEventFieldTypeNames();
					}
				}
				buf.append(" (");
				for (int i = 0 ; i < java.lang.reflect.Array.getLength(names) ; i++) {
					if (i > 0) {
						buf.append(", ");
					}
					buf.append(names[i] + " " + types[i]);
				}
				buf.append(") ");

				try {
					PreparedStatement statement = db.prepareStatement(buf.toString());
					statement.execute();
					statement.close();
				} catch (SQLException e) {
					System.err.println(buf.toString());
					System.err.println(e.getMessage());
					e.printStackTrace(System.err);
				}
				buf = new StringBuilder();
				buf.append(" insert into " + tableName + " ");
			} else {
				buf.append("as ");
			}
		}
		return buf;
	}


	public static List<String> getXMLFields (RMIPerfExplorerModel model) {
		// declare the statement here, so we can reference it in the catch
		// region, if necessary
		StringBuilder buf = null;
		PreparedStatement statement = null;
		HashSet<String> set = new HashSet<String>();
		DB db = null;
		try {
			db = PerfExplorerServer.getServer().getDB();

			//Object object = model.getCurrentSelection();

			// check to make sure the database has XML data
			Trial.getMetaData(db, true);
			String[] fieldNames = db.getDatabase().getTrialFieldNames();
			boolean foundXML = false;
			boolean foundXMLGZ = false;
			for (int i = 0 ; i < java.lang.reflect.Array.getLength(fieldNames) ; i++) {
				if (fieldNames[i].equalsIgnoreCase("XML_METADATA")) {
					foundXML = true;
				} else if (fieldNames[i].equalsIgnoreCase("XML_METADATA_GZ")) {
					foundXMLGZ = true;
				}
			}
			if (!foundXML) {
				// return an empty list
				return new ArrayList<String>();
			}
			////////////////////////////////

			// create and populate the temporary trial table
			buf = buildCreateTableStatement("trial", "temp_trial", db, true, false);
			buf.append("(select trial.* from trial ");
			if (db.getSchemaVersion() == 0) {
				buf.append("inner join experiment ");
				buf.append("on trial.experiment = experiment.id ");
				buf.append("inner join application ");
				buf.append("on experiment.application = application.id ");
			}
			buf.append("where ");
			// add the where clause
			List<Object> selections = model.getMultiSelection();
			if (selections == null) {
				// just one selection
				Object obj = model.getCurrentSelection();
				if (obj instanceof Application) {
					buf.append("application.id = " + model.getApplication().getID());
				} else if (obj instanceof Experiment) {
					buf.append("experiment.id = " + model.getExperiment().getID());
				} else if (obj instanceof Trial) {
					buf.append("trial.id = " + model.getTrial().getID());
				}
			} else {

				// get the selected applications
				boolean foundapp = false;
				for (int i = 0 ; i < selections.size() ; i++) {
					if (selections.get(i) instanceof Application) {
						Application a = (Application)selections.get(i);
						if (!foundapp) {
							buf.append("application.id in (");
							foundapp = true;
						} else {
							buf.append(",");
						}
						buf.append(a.getID());
					}
				}
				if (foundapp) {
					buf.append(") ");
				}

				// get the selected experiments
				boolean foundexp = false;
				for (int i = 0 ; i < selections.size() ; i++) {
					if (selections.get(i) instanceof Experiment) {
						Experiment e = (Experiment)selections.get(i);
						if (!foundexp) {
							if (foundapp) {
								buf.append(" and ");
							}
							buf.append("experiment.id in (");
							foundexp = true;
						} else {
							buf.append(",");
						}
						buf.append(e.getID());
					}
				}
				if (foundexp) {
					buf.append(") ");
				}

				// get the selected trials
				boolean foundtrial = false;
				for (int i = 0 ; i < selections.size() ; i++) {
					if (selections.get(i) instanceof Trial) {
						Trial t = (Trial)selections.get(i);
						if (!foundtrial) {
							if (foundapp || foundexp) {
								buf.append(" and ");
							}
							buf.append("trial.id in (");
							foundtrial = true;
						} else {
							buf.append(",");
						}
						buf.append(t.getID());
					}
				}
				if (foundtrial) {
					buf.append(") ");
				}
			}
			buf.append(") ");
			statement = db.prepareStatement(buf.toString());
			// System.out.println(buf.toString());
			//System.out.println(statement.toString());
			statement.execute();
			statement.close();
			statement = null;

			/////////////////////////

			if (foundXMLGZ) {
				statement = db.prepareStatement("select id, XML_METADATA, XML_METADATA_GZ from temp_trial ");
			} else {
				statement = db.prepareStatement("select id, XML_METADATA from temp_trial ");
			}
			//System.out.println(statement.toString());
			ResultSet xmlResults = statement.executeQuery();

			// build a factory
			DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
			// ask the factory for the document builder
			DocumentBuilder builder = factory.newDocumentBuilder();

			while (xmlResults.next() != false) {
				// get the uncompressed string first...
				String tmp = xmlResults.getString(2);
				if (foundXMLGZ && (tmp == null || tmp.length() == 0)) {
					InputStream compressedStream = xmlResults.getBinaryStream(3);
					tmp = Gzip.decompress(compressedStream);
				}
				// by adding these, we ensure only the main event
				// will be selected in the next temporary table creation!
				Reader reader = new StringReader(tmp);
				InputSource source = new InputSource(reader);
				Document metadata = builder.parse(source);

				/* this is the 1.5 way
				// build the xpath object to jump around in that document
				XPath xpath = XPathFactory.newInstance().newXPath();
				xpath.setNamespaceContext(new TauNamespaceContext());

				// get the common profile attributes from the metadata
				NodeList names = (NodeList) 
					xpath.evaluate("/metadata/CommonProfileAttributes/attribute/name", 
					metadata, XPathConstants.NODESET);

				NodeList values = (NodeList) 
					xpath.evaluate("/metadata/CommonProfileAttributes/attribute/value", 
					metadata, XPathConstants.NODESET);
				 */

				NodeList names = null;

				//				try {
				//					/* this is the 1.3 through 1.4 way */
				//					names = org.apache.xpath.XPathAPI.selectNodeList(metadata, 
				//						"/metadata/CommonProfileAttributes/attribute/name");
				//					for (int i = 0 ; i < names.getLength() ; i++) {
				//						Node name = (Node)names.item(i).getFirstChild();
				//						set.add(name.getNodeValue());
				//					}
				//					names = org.apache.xpath.XPathAPI.selectNodeList(metadata, 
				//						"/metadata/ProfileAttributes/attribute/name");
				//				
				//					for (int i = 0 ; i < names.getLength() ; i++) {
				//						Node name = (Node)names.item(i).getFirstChild();
				//						set.add(name.getNodeValue());
				//					}
				//				} catch (NoClassDefFoundError e) {
				/* this is the 1.5 way */
				// build the xpath object to jump around in that document
				javax.xml.xpath.XPath xpath = javax.xml.xpath.XPathFactory.newInstance().newXPath();
				xpath.setNamespaceContext(new TauNamespaceContext());

				// get the common profile attributes from the metadata
				names = (NodeList) 
				xpath.evaluate("/metadata/CommonProfileAttributes/attribute/name", 
						metadata, javax.xml.xpath.XPathConstants.NODESET);

				for (int i = 0 ; i < names.getLength() ; i++) {
					Node name = (Node)names.item(i).getFirstChild();
					set.add(name.getNodeValue());
				}

				names = (NodeList) 
				xpath.evaluate("/metadata/CommonProfileAttributes/attribute/name", 
						metadata, javax.xml.xpath.XPathConstants.NODESET);

				for (int i = 0 ; i < names.getLength() ; i++) {
					Node name = (Node)names.item(i).getFirstChild();
					set.add(name.getNodeValue());
				}
				//				}

			} 
			xmlResults.close();
			statement.close();
			statement = null;

		} catch (Exception e) {
			// the user may have gotten here because they don't have XML data.
			/*
			if (statement != null)
				PerfExplorerOutput.println(statement.toString());
			PerfExplorerOutput.println(buf.toString());
			System.err.println(e.getMessage());
			e.printStackTrace();
			 */
			// System.err.println("***************************************");
			// System.err.println(e.getMessage());
			// System.err.println("***************************************");
		} finally {
			if (statement != null) {
				try {
					statement.close();
				} catch (Exception e) {
					// System.err.println("***************************************");
					// System.err.println(e.getMessage());
					// System.err.println("***************************************");
				}
			}
			dropTable(db, "temp_trial");
		}

		List<String> list = new ArrayList<String>(set);
		Collections.sort(list, String.CASE_INSENSITIVE_ORDER);
		return list;
	}

	private static void dropTable(DB db, String name) {
		PreparedStatement statement = null;
		try {
			if (db.getDBType().compareTo("oracle") == 0) {
				statement = db.prepareStatement("truncate table " + name);
				//System.out.println(statement.toString());
				statement.execute();
				statement.close();
			}
			statement = db.prepareStatement("drop table " + name);
			//System.out.println(statement.toString());
			statement.execute();
			statement.close();
		} catch (Exception e) {
			// do nothing, it's ok
			// System.err.println("***************************************");
			// System.err.println(e.getMessage());
			// System.err.println("***************************************");
		} finally {
			// if we tried to truncate a table that doesn't exist,
			// then we got an error.  We still need to close the statement.
			try {
				statement.close();
			} catch (Exception e2) {
				//System.err.println("***************************************");
				//System.err.println(e2.getMessage());
				//System.err.println("***************************************");
			}
		}
	}

}

