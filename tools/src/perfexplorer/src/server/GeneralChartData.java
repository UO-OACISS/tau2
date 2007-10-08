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
import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.IntervalEvent;
import edu.uoregon.tau.common.Gzip;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import java.util.ArrayList;
import java.util.List;
import java.util.HashSet;
import java.util.Collections;

import javax.xml.parsers.ParserConfigurationException;
import java.util.regex.*;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.*;
import javax.xml.transform.dom.*;
import javax.xml.transform.stream.*;
import javax.xml.parsers.DocumentBuilder;
import org.w3c.dom.*;
import org.xml.sax.*;
import java.io.StringReader;
import java.io.Reader;
import java.io.InputStream;

/**
 * The GeneralChartData class is used to select data from the database which 
 * represents the performance profile of the selected trials, and return them
 * in a format for JFreeChart to display them.
 *
 * <P>CVS $Id: GeneralChartData.java,v 1.24 2007/10/08 16:35:52 khuck Exp $</P>
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
		//PerfExplorerOutput.println("getChartData(" + model.toString() + ")...");
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
		DB db = null;
		try {
			db = PerfExplorerServer.getServer().getDB();

			Object object = model.getCurrentSelection();

////////////////////////////////

			// create and populate the temporary trial table
			buf = buildCreateTableStatement("trial", "temp_trial", db, true);
   			buf.append("(select trial.* from trial ");
	        buf.append("inner join experiment ");
			buf.append("on trial.experiment = experiment.id ");
			buf.append("inner join application ");
			buf.append("on experiment.application = application.id ");
			buf.append("where ");
			// add the where clause
			List selections = model.getMultiSelection();
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
			//System.out.println(statement.toString());
			statement.execute();
			statement.close();

/////////////////////////

			boolean gotXMLData = false;
			if (model.getChartMetadataFieldName() != null ||
				model.getChartMetadataFieldValue() != null ||
				model.getChartSeriesName().toUpperCase().indexOf("XML") > 0 ||
				model.getChartXAxisName().toUpperCase().indexOf("XML") > 0) {
				gotXMLData = true;

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
			
			if (foundXML) {
				// create and populate the temporary XML_METADATA table
				buf = buildCreateTableStatement("xml_metadata",
						"temp_xml_metadata", db, false);
				if (db.getDBType().compareTo("derby") == 0) {
					buf.append(" (trial int, metadata_name varchar(4000), metadata_value varchar(4000))");
				} else {
					buf.append(" (trial int, metadata_name text, metadata_value text)");
				}
				statement = db.prepareStatement(buf.toString());
				//System.out.println(statement.toString());
				statement.execute();
				statement.close();

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
	
				db.setAutoCommit(false);
	
				while (xmlResults.next() != false) {
					// get the uncompressed string first...
					String tmp = xmlResults.getString(2);
					if (foundXMLGZ && (tmp == null || tmp.length() == 0)) {
						InputStream compressedStream = xmlResults.getBinaryStream(3);
						tmp = Gzip.decompress(compressedStream);
					}
					// by adding these, we ensure only the main event
					// will be selected in the next temporary table creation!
					if (tmp == null || tmp.length() == 0) {
						continue;
					}
					Reader reader = new StringReader(tmp);
					InputSource source = new InputSource(reader);
					Document metadata = builder.parse(source);
	
					NodeList names = null;
					NodeList values = null;
	
					try {
						/* this is the 1.3 through 1.4 way */
						names = org.apache.xpath.XPathAPI.selectNodeList(metadata, 
							"/metadata/CommonProfileAttributes/attribute/name");
						values = org.apache.xpath.XPathAPI.selectNodeList(metadata, 
							"/metadata/CommonProfileAttributes/attribute/value");
					} catch (NoClassDefFoundError e) {
	
						/* this is the 1.5 way */
						// build the xpath object to jump around in that document
						javax.xml.xpath.XPath xpath = javax.xml.xpath.XPathFactory.newInstance().newXPath();
						xpath.setNamespaceContext(new TauNamespaceContext());
		
						// get the common profile attributes from the metadata
						names = (NodeList) 
							xpath.evaluate("/metadata/CommonProfileAttributes/attribute/name", 
							metadata, javax.xml.xpath.XPathConstants.NODESET);
	
						values = (NodeList) 
							xpath.evaluate("/metadata/CommonProfileAttributes/attribute/value", 
							metadata, javax.xml.xpath.XPathConstants.NODESET);
					}
			
					for (int i = 0 ; i < names.getLength() ; i++) {
						Node name = (Node)names.item(i).getFirstChild();
						Node value = (Node)values.item(i).getFirstChild();
						//System.out.println(name.getNodeValue()+" =? "+model.getChartMetadataFieldName());
						if ((model.getChartMetadataFieldName() == null ||
					     	model.getChartMetadataFieldName().equals(name.getNodeValue())) &&
							(model.getChartMetadataFieldValue() == null ||
					     	model.getChartMetadataFieldValue().equals(value.getNodeValue()))) {
							buf = new StringBuffer();
							buf.append("insert into temp_xml_metadata VALUES (?,?,?)");
							PreparedStatement statement2 = db.prepareStatement(buf.toString());
							statement2.setInt(1, xmlResults.getInt(1));
							statement2.setString(2, name.getNodeValue());
							if (value == null) {
								statement2.setString(3, "");
							} else {
								statement2.setString(3, value.getNodeValue());
							}
							//System.out.println(statement2.toString());
							statement2.executeUpdate();
							statement2.close();
						}
					}
				} 
				db.commit();
				xmlResults.close();
				statement.close();
				db.setAutoCommit(true);
				}
			}

/////////////////////////

			// create and populate the temporary metric table
			buf = buildCreateTableStatement("metric", "temp_metric", db, true);
    		buf.append("(select metric.* from metric ");
			buf.append("inner join temp_trial ");
			buf.append("on metric.trial = temp_trial.id ");
			// add the where clause
			List metricNames = model.getMetricNames();
			if (metricNames != null) {
				//if (db.getDBType().compareTo("db2") == 0) {
					buf.append("where upper(metric.name) like ? ");
				//} else {
					//buf.append("where metric.name = ? ");
				//}
				for (int i = 1 ; i < metricNames.size() ; i++) {
					//if (db.getDBType().compareTo("db2") == 0) {
						buf.append("or upper(metric.name) like ? ");
					//} else {
						//buf.append("or metric.name = ? ");
					//}
				}
			}
			buf.append(") ");
//			System.out.println(buf.toString());
			statement = db.prepareStatement(buf.toString());
			if (metricNames != null) {
				for (int i = 1 ; i <= metricNames.size() ; i++) {
					String tmp = (String)metricNames.get(i-1);
					statement.setString(i, tmp.toUpperCase());
				}
			}
			//System.out.println(statement.toString());
			statement.execute();
			statement.close();

////////////////////////////////

			// if we only want the main event, handle that
			// we need a sub query.  Bah.
			if (model.getMainEventOnly()) {
				buf = new StringBuffer();
				buf.append("select ie.name from interval_event ie ");
   				buf.append("inner join interval_mean_summary ims ");
				buf.append("on ie.id = ims.interval_event, ");
				buf.append("(select temp_trial.id as trialid, ");
				buf.append("temp_metric.id as metricid, ");
				buf.append("max(interval_mean_summary.inclusive) as maxinclusive ");
				buf.append("from interval_event inner join temp_trial ");
				buf.append("on interval_event.trial = temp_trial.id ");
				buf.append("inner join interval_mean_summary on ");
				buf.append("interval_mean_summary.interval_event = interval_event.id ");
				buf.append("inner join temp_metric ");
				buf.append("on interval_mean_summary.metric = temp_metric.id ");
				if (db.getDBType().compareTo("derby") == 0) {
					buf.append("group by temp_trial.id, temp_metric.id) mr ");
				} else {
					buf.append("group by 1, 2) mr ");
				}
				buf.append("where ie.trial = trialid ");
   				buf.append("and ims.metric = metricid ");
   				buf.append("and ims.inclusive = maxinclusive");
				statement = db.prepareStatement(buf.toString());
				//System.out.println(statement.toString());
				ResultSet results = statement.executeQuery();

				while (results.next() != false) {
					// by adding these, we ensure only the main event
					// will be selected in the next temporary table creation!
					model.addEventName(results.getString(1));
				} 
				results.close();
				statement.close();
			} 

////////////////////////////////

			// create and populate the temporary event table
			buf = buildCreateTableStatement("event", "temp_event", db, true);
			buf.append("(select interval_event.* from interval_event ");
			buf.append("inner join temp_trial ");
			buf.append("on interval_event.trial = temp_trial.id ");
			buf.append("inner join interval_mean_summary ");
   			buf.append("on interval_mean_summary.interval_event = interval_event.id ");
			buf.append("inner join temp_metric ");
			buf.append("on interval_mean_summary.metric = temp_metric.id ");

			// add the where clause
			boolean didWhere = false;

			// if we don't want to include callpath or phase data, handle that
			if (model.getEventNoCallpath()) {
				if (didWhere) {
					buf.append("and ");
				} else {
					buf.append("where ");
					didWhere = true;
				}
				buf.append("(interval_event.group_name is null ");
        		buf.append("or (interval_event.group_name not like '%TAU_CALLPATH%' ");
            	buf.append("and interval_event.group_name not like '%TAU_PHASE%')) ");
			}

			// if we only want to see events with > X percent of total runtime
			if (model.getDimensionReduction() == TransformationType.OVER_X_PERCENT) {
				if (didWhere) {
					buf.append("and ");
				} else {
					buf.append("where ");
					didWhere = true;
				}
				buf.append("interval_mean_summary.exclusive_percentage > ");
				buf.append(model.getXPercent());
				buf.append(" ");
			}

			// if we want to see the event with 100% exclusive
			if (model.getEventExclusive100()) {
				if (didWhere) {
					buf.append("and ");
				} else {
					buf.append("where ");
					didWhere = true;
				}
				buf.append("interval_mean_summary.exclusive_percentage = 100 ");
			}

			// if we want to see events in particular groups
			List groupNames = model.getGroupNames();
			if (groupNames != null) {
				boolean gotOne = false;
				for (int i = 0 ; i < groupNames.size() ; i++) {
					if (didWhere) {
						if (gotOne) {
							buf.append("or ");
						} else {
							buf.append("and ");
							gotOne = true;
						}
					} else {
						buf.append("where ");
						didWhere = true;
					}
					if (db.getDBType().compareTo("db2") == 0) {
						buf.append("interval_event.group_name like ? ");
					} else {
						buf.append("interval_event.group_name = ? ");
					}
				}
			}

			// if we want to see particular events
			List eventNames = model.getEventNames();
			if (eventNames != null) {
				boolean gotOne = false;
				for (int i = 0 ; i < eventNames.size() ; i++) {
					if (didWhere) {
						if (gotOne) {
							buf.append("or ");
						} else {
							buf.append("and ");
							gotOne = true;
						}
					} else {
						buf.append("where ");
						didWhere = true;
					}
					if (db.getDBType().compareTo("db2") == 0) {
						buf.append("interval_event.name like ? ");
					} else {
						buf.append("interval_event.name = ? ");
					}
				}
			}
			buf.append(")");

			// build the statement
			statement = db.prepareStatement(buf.toString());

			// if we put parameters in the statement, fill them.
			// group names first...
			int currentParameter = 1;
			if (groupNames != null) {
				for (int i = 0 ; i < groupNames.size() ; i++) {
					String tmp = (String)groupNames.get(i);
					statement.setString(currentParameter, tmp);
					currentParameter++;
				}
			}
			// then event names...
			if (eventNames != null) {
				for (int i = 0 ; i < eventNames.size() ; i++) {
					String tmp = (String)eventNames.get(i);
					statement.setString(currentParameter, tmp);
					currentParameter++;
				}
			}

			//System.out.println(statement.toString());
			statement.execute();
			statement.close();

////////////////////////////////

			// The user wants parametric study data, with the data
			// organized with two axes, the x and the y.
			// unlike scalability: 
			// NO ASSUMPTION ABOUT WHICH COLUMN IS THE SERIES NAME,
			// Y AXIS OR X AXIS!
			String seriesName = model.getChartSeriesName();
			String xAxisName = model.getChartXAxisName();
			String yAxisName = model.getChartYAxisName();
			String tableName = "interval_mean_summary";
			if (yAxisName.indexOf("interval_mean_summary") >= 0) {
				tableName = "interval_mean_summary";
			} else if (yAxisName.indexOf("interval_total_summary") >= 0) {
				tableName = "interval_total_summary";
			}

			buf = new StringBuffer();
			buf.append("select ");
			if (db.getDBType().compareTo("derby") == 0) {
				// first item - the series name
				buf.append(fixClause(seriesName, db) + ", ");
				// second item - the x axis
				buf.append(fixClause(xAxisName, db) + ", ");
				// second item - the y axis
				buf.append(fixClause(yAxisName, db) + " ");
			} else {
				// first item - the series name
				buf.append(fixClause(seriesName, db) + " as series_name, ");
				// second item - the x axis
				buf.append(fixClause(xAxisName, db) + " as xaxis_value, ");
				// second item - the y axis
				buf.append(fixClause(yAxisName, db) + " as yaxis_value ");
			}
			// add the tables
			buf.append("from " + tableName + " ");
			buf.append("inner join temp_metric ");
			buf.append("on " + tableName + ".metric = temp_metric.id ");
			buf.append("inner join temp_event ");
			buf.append("on " + tableName + ".interval_event = temp_event.id ");
			buf.append("inner join temp_trial ");
			buf.append("on temp_event.trial = temp_trial.id ");
			if (gotXMLData) {
				buf.append("inner join temp_xml_metadata ");
				buf.append("on temp_event.trial = temp_xml_metadata.trial ");
			}
			buf.append("inner join experiment ");
			buf.append("on temp_trial.experiment = experiment.id ");
			buf.append("inner join application ");
			buf.append("on experiment.application = application.id ");
			// no where clause (thanks to the temporary tables)
			// group by clause, in case there are operations on the columns
			//buf.append("group by " + fixClause(seriesName, db));
			//buf.append(", " + fixClause(xAxisName, db) + " " );
			if (db.getDBType().compareTo("derby") == 0) {
				buf.append("group by " + fixClause(seriesName, db) + ", " + fixClause(xAxisName, db));
				// add the order by clause
				if (seriesName.startsWith("temp_trial.node_count") || xAxisName.startsWith("temp_trial.node_count")) {
					buf.append(" order by 1, 2, 3, 4 ");
				} else {
					buf.append(" order by 1, 2 ");					
				}
			} else {
				buf.append("group by series_name, xaxis_value ");
				// add the order by clause
				buf.append("order by 1, 2 ");
			}
			statement = db.prepareStatement(buf.toString());
			//System.out.println(statement.toString());
			ResultSet results = statement.executeQuery();

			while (results.next() != false) {
				// System.out.print(results.getString(1) + ": " );
				// System.out.print(results.getString(2) + ", " );
				// System.out.println(results.getDouble(3));
				if (db.getDBType().compareTo("derby") == 0) {
					if (seriesName.startsWith("temp_trial.node_count")) {
						addRow(Integer.toString(results.getInt(1) * results.getInt(2) * results.getInt(3)), results.getString(4), results.getDouble(5));
					} else if (xAxisName.startsWith("temp_trial.node_count")) {
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
				dropTable(db, "temp_metric");
				dropTable(db, "temp_xml_metadata");
				dropTable(db, "temp_trial");
			} catch (Exception e) {}
		}
	}

	private static StringBuffer buildCreateTableStatement (String oldTableName, String tableName, DB db, boolean appendAs) {
		// just in case, drop the table in case it is still hanging around.
		// This sometimes happens with Derby.
		// Have I ever mentioned that Derby sucks?
		dropTable(db, tableName);

		StringBuffer buf = new StringBuffer();
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
					IntervalEvent.getMetaData(db);
					names = db.getDatabase().getIntervalEventFieldNames();
					types = db.getDatabase().getIntervalEventFieldTypeNames();
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
				buf = new StringBuffer();
				buf.append(" insert into " + tableName + " ");
			} else {
				buf.append("as ");
			}
		}
		return buf;
	}

	private String fixClause (String inString, DB db) {
		// change the table names
		String outString = inString.replaceAll("trial.", "temp_trial.").replaceAll("metric.", "temp_metric.").replaceAll("interval_event.", "temp_event.");
		// fix the oracle specific stuff
		if (db.getDBType().equalsIgnoreCase("oracle")) {
			outString = outString.replaceAll("exclusive", "exec");
		}
		if (db.getDBType().equalsIgnoreCase("derby")) {
			outString = outString.replaceAll("call", "num_calls");
			outString = outString.replaceAll(" \\* ", ", ");
		}
		// and so forth
		return outString;
	}

	public static List getXMLFields (RMIPerfExplorerModel model) {
		// declare the statement here, so we can reference it in the catch
		// region, if necessary
		StringBuffer buf = null;
		PreparedStatement statement = null;
		HashSet set = new HashSet();
		DB db = null;
		try {
			db = PerfExplorerServer.getServer().getDB();

			Object object = model.getCurrentSelection();

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
				return new ArrayList();
			}
	////////////////////////////////

			// create and populate the temporary trial table
			buf = buildCreateTableStatement("trial", "temp_trial", db, true);
   			buf.append("(select trial.* from trial ");
	        buf.append("inner join experiment ");
			buf.append("on trial.experiment = experiment.id ");
			buf.append("inner join application ");
			buf.append("on experiment.application = application.id ");
			buf.append("where ");
			// add the where clause
			List selections = model.getMultiSelection();
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

				try {
					/* this is the 1.3 through 1.4 way */
					names = org.apache.xpath.XPathAPI.selectNodeList(metadata, 
						"/metadata/CommonProfileAttributes/attribute/name");
					for (int i = 0 ; i < names.getLength() ; i++) {
						Node name = (Node)names.item(i).getFirstChild();
						set.add(name.getNodeValue());
					}
					names = org.apache.xpath.XPathAPI.selectNodeList(metadata, 
						"/metadata/ProfileAttributes/attribute/name");
				
					for (int i = 0 ; i < names.getLength() ; i++) {
						Node name = (Node)names.item(i).getFirstChild();
						set.add(name.getNodeValue());
					}
				} catch (NoClassDefFoundError e) {
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
				}

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

		List list = new ArrayList(set);
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

