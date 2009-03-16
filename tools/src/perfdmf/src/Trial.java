package edu.uoregon.tau.perfdmf;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.sql.DatabaseMetaData;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Vector;

import org.xml.sax.Attributes;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;
import org.xml.sax.helpers.DefaultHandler;
import org.xml.sax.helpers.XMLReaderFactory;

import edu.uoregon.tau.common.AlphanumComparator;
import edu.uoregon.tau.common.Gzip;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.database.DBConnector;

/**
 * Holds all the data for a trial in the database. This object is returned by
 * the DataSession class and all of its subtypes. The Trial object contains all
 * the information associated with an trial from which the TAU performance data
 * has been generated. A trial is associated with one experiment and one
 * application, and has one or more interval_events and/or user events
 * associated with it. A Trial has information related to the particular run,
 * including the number of nodes used, the number of contexts per node, the
 * number of threads per context and the metrics collected during the run.
 * 
 * <P>
 * CVS $Id: Trial.java,v 1.32 2009/03/16 23:25:44 wspear Exp $
 * </P>
 * 
 * @author Kevin Huck, Robert Bell
 * @version 0.1
 * @since 0.1
 * @see DataSession#getTrialList
 * @see DataSession#setTrial
 * @see Application
 * @see Experiment
 * @see IntervalEvent
 * @see AtomicEvent
 */
public class Trial implements Serializable, Comparable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 3356242487725127605L;
	public static final String XML_METADATA = "XML_METADATA";
	public static final String XML_METADATA_GZ = "XML_METADATA_GZ";

	private int trialID;
	private int experimentID;
	private int applicationID;
	private String name;
	private Vector metrics;
	private String fields[];

	protected DataSource dataSource = null;

	private Database database;
	private Map metaData = new TreeMap();
	private Map uncommonMetaData = new TreeMap();

	private boolean xmlMetaDataLoaded=false;

	public boolean isXmlMetaDataLoaded() {
		return xmlMetaDataLoaded;
	}

	public void setXmlMetaDataLoaded(boolean xmlMetaDataLoaded) {
		this.xmlMetaDataLoaded = xmlMetaDataLoaded;
	}

	private static AlphanumComparator alphanum = new AlphanumComparator();

	private static class XMLParser extends DefaultHandler {
		private StringBuffer accumulator = new StringBuffer();
		private String currentName = "";

		private Map common, other, current;

		public XMLParser(Map common, Map other) {
			this.common = common;
			this.other = other;
			current = common;
		}

		public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
			accumulator = new StringBuffer();
			if (localName.equals("CommonProfileAttributes")) {
				current = common;
			} else if (localName.equals("ProfileAttributes")) {
				current = other;
			}
		}

		public void endElement(String uri, String localName, String qName) throws SAXException {
			if (localName.equals("name")) {
				currentName = accumulator.toString().trim();
			} else if (localName.equals("value")) {
				String currentValue = accumulator.toString().trim();
				current.put(currentName, currentValue);
			}
		}

		public void characters(char[] ch, int start, int length) throws SAXException {
			accumulator.append(ch, start, length);
		}
	}

	private void parseMetaData(String string) {
		try {
			metaData = new TreeMap();
			XMLReader xmlreader = XMLReaderFactory.createXMLReader("org.apache.xerces.parsers.SAXParser");
			XMLParser parser = new XMLParser(metaData, uncommonMetaData);
			xmlreader.setContentHandler(parser);
			xmlreader.setErrorHandler(parser);
			ByteArrayInputStream input = new ByteArrayInputStream(string.getBytes());
			xmlreader.parse(new InputSource(input));

		} catch (SAXException e) {
			// oh well, no metadata
		} catch (IOException e) {
			// oh well, no metadata
		}
	}

	public Trial() {
		this.fields = new String[0];
	}

	// copy constructor
	public Trial(Trial trial) {
		this.name = trial.getName();
		this.applicationID = trial.getApplicationID();
		this.experimentID = trial.getExperimentID();
		this.trialID = trial.getID();
		this.fields = (String[]) trial.fields.clone();
		this.metaData = trial.metaData;
		this.uncommonMetaData = trial.uncommonMetaData;
		this.database = trial.database;
		this.dataSource = trial.dataSource;
	}

	///////////////////////////////////////////////////////

	public int getNumFields() {
		return fields.length;
	}

	public String getFieldName(int idx) {
		return database.getTrialFieldNames()[idx];
	}

	public int getFieldType(int idx) {
		return database.getTrialFieldTypes()[idx];
	}

	public String getField(int idx) {

		return fields[idx];
	}

	public void loadXMLMetadata(DB db) throws SQLException{

		if(isXmlMetaDataLoaded()){
			return;
		}
		
		StringBuffer buf = new StringBuffer();
		buf.append("select ");
		boolean first=true;
		int xDex=-1;
		int zxDex=-1;
		for (int i = 0; i < database.getTrialFieldNames().length; i++) {
			if(database.getTrialFieldNames()[i].toUpperCase().equals(XML_METADATA_GZ)||database.getTrialFieldNames()[i].toUpperCase().equals(XML_METADATA)){
				if(database.getTrialFieldNames()[i].toUpperCase().equals(XML_METADATA_GZ)){
					zxDex=i;
				}
				if(database.getTrialFieldNames()[i].toUpperCase().equals(XML_METADATA)){
					xDex=i;
				}
				if(!first){
					buf.append(", ");
				}
				buf.append("t." + database.getTrialFieldNames()[i]);
				first=false;
			}	
		}

		buf.append(" from " + db.getSchemaPrefix() + "trial t inner join " + db.getSchemaPrefix() + "experiment e ");
		buf.append("on t.experiment = e.id ");
		buf.append("WHERE t.experiment = "+getExperimentID()+" AND t.id = "+getID());

		ResultSet resultSet=null;
		
		resultSet = db.executeQuery(buf.toString());
		
		
		
		if(resultSet.next()!=false)
		{
			setField(xDex, resultSet.getString(1));
		
			InputStream compressedStream = resultSet.getBinaryStream(2);
			String tmp = Gzip.decompress(compressedStream);
		
			if (tmp != null && tmp.length() > 0) {
				setField(XML_METADATA, tmp);
				parseMetaData(tmp);
			}
		
		}
		
		this.setXmlMetaDataLoaded(true);
	}

	public String getField(String name) {
		if (database.getTrialFieldNames() == null)
			return null;

		String field=null;
		int i = 0;
		for (; i < database.getTrialFieldNames().length; i++) {
			if (name.toUpperCase().equals(database.getTrialFieldNames()[i].toUpperCase())) {
				if (i < fields.length)
				{
					field = fields[i];
					break;
				}
			}
		}
		return field;
	}

	public void setField(String field, String value) {
		for (int i = 0; i < database.getTrialFieldNames().length; i++) {
			if (field.toUpperCase().equals(database.getTrialFieldNames()[i].toUpperCase())) {

				if (DBConnector.isIntegerType(database.getTrialFieldTypes()[i]) && value != null) {
					try {
						int test = Integer.parseInt(value);
					} catch (java.lang.NumberFormatException e) {
						return;
					}
				}

				if (DBConnector.isFloatingPointType(database.getTrialFieldTypes()[i]) && value != null) {
					try {
						double test = Double.parseDouble(value);
					} catch (java.lang.NumberFormatException e) {
						return;
					}
				}

				if (fields.length <= i) {
					fields = new String[database.getTrialFieldTypes().length];
				}
				fields[i] = value;
			}
		}
	}

	public void setField(int idx, String value) {
		if (DBConnector.isIntegerType(database.getTrialFieldTypes()[idx]) && value != null) {
			try {
				int test = Integer.parseInt(value);
			} catch (java.lang.NumberFormatException e) {
				return;
			}
		}

		if (DBConnector.isFloatingPointType(database.getTrialFieldTypes()[idx]) && value != null) {
			try {
				double test = Double.parseDouble(value);
			} catch (java.lang.NumberFormatException e) {
				return;
			}
		}

		fields[idx] = value;
	}

	/**
	 * Gets the unique identifier of the current trial object.
	 * 
	 * @return the unique identifier of the trial
	 */
	public int getID() {
		return trialID;
	}

	/**
	 * Gets the unique identifier for the experiment associated with this trial.
	 * 
	 * @return the unique identifier of the experiment
	 */
	public int getExperimentID() {
		return experimentID;
	}

	/**
	 * Gets the unique identifier for the application associated with this
	 * trial.
	 * 
	 * @return the unique identifier of the application
	 */
	public int getApplicationID() {
		return applicationID;
	}

	/**
	 * Gets the name of the current trial object.
	 * 
	 * @return the name of the trial
	 */
	public String getName() {
		return name;
	}

	public String toString() {
		return name;
	}

	/**
	 * Gets the data session for this trial.
	 * 
	 * @return data dession for this trial.
	 */
	public DataSource getDataSource() {
		return this.dataSource;
	}

	/**
	 * Gets the number of metrics collected in this trial.
	 * 
	 * @return metric count for this trial.
	 */
	public int getMetricCount() {
		if (this.metrics == null)
			return 0;
		else
			return this.metrics.size();
	}

	/**
	 * Gets the metrics collected in this trial.
	 * 
	 * @return metric vector
	 */
	public Vector getMetrics() {
		return this.metrics;
	}

	/**
	 * Get the metric name corresponding to the given id. The DataSession object
	 * will maintain a reference to the Vector of metric values. To clear this
	 * reference, call setMetric(String) with null.
	 * 
	 * @param metricID
	 *            metric id.
	 * 
	 * @return The metric name as a String.
	 */
	public String getMetricName(int metricID) {

		//Try getting the metric name.
		if ((this.metrics != null) && (metricID < this.metrics.size()))
			return ((Metric) this.metrics.elementAt(metricID)).getName();
		else
			return null;
	}

	/**
	 * Sets the unique ID associated with this trial. <i>NOTE: This method is
	 * used by the DataSession object to initialize the object. Not currently
	 * intended for use by any other code. </i>
	 * 
	 * @param id
	 *            unique ID associated with this trial
	 */
	public void setID(int id) {
		this.trialID = id;
	}

	/**
	 * Sets the experiment ID associated with this trial. <i>NOTE: This method
	 * is used by the DataSession object to initialize the object. Not currently
	 * intended for use by any other code. </i>
	 * 
	 * @param experimentID
	 *            experiment ID associated with this trial
	 */
	public void setExperimentID(int experimentID) {
		this.experimentID = experimentID;
	}

	/**
	 * Sets the application ID associated with this trial. <i>NOTE: This method
	 * is used by the DataSession object to initialize the object. Not currently
	 * intended for use by any other code. </i>
	 * 
	 * @param applicationID
	 *            application ID associated with this trial
	 */
	public void setApplicationID(int applicationID) {
		this.applicationID = applicationID;
	}

	/**
	 * Sets the name of the current trial object. <i>Note: This method is used
	 * by the DataSession object to initialize the object. Not currently
	 * intended for use by any other code. </i>
	 * 
	 * @param name
	 *            the trial name
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * Sets the data session for this trial.
	 * 
	 * @param dataSession
	 *            DataSession for this trial
	 */
	public void setDataSource(DataSource dataSource) {
		this.dataSource = dataSource;
	}

	/**
	 * Adds a metric to this trial. <i>NOTE: This method is used by the
	 * DataSession object to initialize the object. Not currently intended for
	 * use by any other code. </i>
	 * 
	 * @param metric
	 *            Adds a metric to this trial
	 */
	public void addMetric(Metric metric) {
		if (this.metrics == null)
			this.metrics = new Vector();
		this.metrics.addElement(metric);
	}

	// gets the metric data for the trial
	private void getTrialMetrics(DB db) {
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select id, name ");
		buf.append("from " + db.getSchemaPrefix() + "metric ");
		buf.append("where trial = ");
		buf.append(getID());
		buf.append(" order by id ");
		// System.out.println(buf.toString());

		// get the results
		try {
			ResultSet resultSet = db.executeQuery(buf.toString());
			while (resultSet.next() != false) {
				Metric tmp = new Metric();
				tmp.setID(resultSet.getInt(1));
				tmp.setName(resultSet.getString(2));
				tmp.setTrialID(getID());
				addMetric(tmp);
			}
			resultSet.close();
		} catch (Exception ex) {
			ex.printStackTrace();
			return;
		}
		return;
	}

	/**
	 * Returns the column names for the Trial table
	 *
	 * @param	db	the database connection
	 * @return	String[] an array of String objects
	 */
	public static String[] getFieldNames(DB db) {
		getMetaData(db);
		return db.getDatabase().getTrialFieldNames();
	}

	public static void getMetaData(DB db) {
		getMetaData(db, false);
	}

	public static void getMetaData(DB db, boolean allColumns) {
		// see if we've already have them
		// need to load each time in case we are working with a new database. 
		//        if (Trial.fieldNames != null)
		//            return;

		try {
			ResultSet resultSet = null;

			String trialFieldNames[] = null;
			int trialFieldTypes[] = null;

			DatabaseMetaData dbMeta = db.getMetaData();

			if ((db.getDBType().compareTo("oracle") == 0) || (db.getDBType().compareTo("derby") == 0)
					|| (db.getDBType().compareTo("db2") == 0)) {
				resultSet = dbMeta.getColumns(null, null, "TRIAL", "%");
			} else {
				resultSet = dbMeta.getColumns(null, null, "trial", "%");
			}

			Vector nameList = new Vector();
			Vector typeList = new Vector();
			List typeNames = new ArrayList();
			List columnSizes = new ArrayList();
			boolean seenID = false;

			ResultSetMetaData md = resultSet.getMetaData();
			for (int i = 0; i < md.getColumnCount(); i++) {
				//System.out.println(md.getColumnName(i));
			}

			while (resultSet.next() != false) {

				int ctype = resultSet.getInt("DATA_TYPE");
				String cname = resultSet.getString("COLUMN_NAME");
				String typename = resultSet.getString("TYPE_NAME");
				Integer size = new Integer(resultSet.getInt("COLUMN_SIZE"));

				// this code is because of a bug in derby...
				if (cname.equals("ID")) {
					if (!seenID)
						seenID = true;
					else
						break;
				}

				// only integer and string types (for now)
				// don't do name and id, we already know about them

				if (allColumns
						|| (DBConnector.isReadAbleType(ctype) && cname.toUpperCase().compareTo("ID") != 0
								&& cname.toUpperCase().compareTo("NAME") != 0
								&& cname.toUpperCase().compareTo("APPLICATION") != 0 && cname.toUpperCase().compareTo(
								"EXPERIMENT") != 0)) {

					nameList.add(resultSet.getString("COLUMN_NAME"));
					typeList.add(new Integer(ctype));
					typeNames.add(typename);
					columnSizes.add(size);
				}
			}
			resultSet.close();

			String[] fieldNames = new String[nameList.size()];
			int[] fieldTypes = new int[typeList.size()];
			String[] fieldTypeNames = new String[typeList.size()];
			for (int i = 0; i < typeList.size(); i++) {
				fieldNames[i] = (String) nameList.get(i);
				fieldTypes[i] = ((Integer) typeList.get(i)).intValue();
				if (((Integer) columnSizes.get(i)).intValue() > 255) {
					fieldTypeNames[i] = (String) typeNames.get(i) + "(" + columnSizes.get(i).toString() + ")";
				} else {
					fieldTypeNames[i] = (String) typeNames.get(i);
				}
			}

			db.getDatabase().setTrialFieldNames(fieldNames);
			db.getDatabase().setTrialFieldTypes(fieldTypes);
			db.getDatabase().setTrialFieldTypeNames(fieldTypeNames);
		} catch (SQLException e) {
			e.printStackTrace();
		}
	}


	private static Vector getTrialList(DB db, String whereClause){
		return getTrialList(db,whereClause,true);
	}

	public static Vector getTrialListWithoutMetadata(DB db, String whereClause){
		return getTrialList(db,whereClause,false);
	}

	public static Vector getTrialList(DB db, String whereClause,boolean getXMLMetadata) {

		try {

			Trial.getMetaData(db);
			Database database = db.getDatabase();

			// create a string to hit the database
			StringBuffer buf = new StringBuffer();
			buf.append("select t.id, t.experiment, e.application, ");
			buf.append("t.name");

			for (int i = 0; i < database.getTrialFieldNames().length; i++) {
				if(!getXMLMetadata&&(database.getTrialFieldNames()[i].toUpperCase().equals(XML_METADATA_GZ)||database.getTrialFieldNames()[i].toUpperCase().equals(XML_METADATA))){
					continue;
				}
				else
					buf.append(", t." + database.getTrialFieldNames()[i]);
			}

			buf.append(" from " + db.getSchemaPrefix() + "trial t inner join " + db.getSchemaPrefix() + "experiment e ");
			buf.append("on t.experiment = e.id ");
			buf.append(whereClause);
			buf.append(" order by t.name ");

			Vector trials = new Vector();

			ResultSet resultSet = db.executeQuery(buf.toString());
			while (resultSet.next() != false) {
				Trial trial = new Trial();
				trial.setDatabase(db.getDatabase());
				int pos = 1;
				trial.setID(resultSet.getInt(pos++));
				trial.setExperimentID(resultSet.getInt(pos++));
				trial.setApplicationID(resultSet.getInt(pos++));
				trial.setName(resultSet.getString(pos++));

				boolean xmlSet = false;


				for (int i = 0; i < database.getTrialFieldNames().length; i++) {
					if (database.getTrialFieldNames()[i].equalsIgnoreCase(XML_METADATA_GZ)) {
						if(getXMLMetadata){
							InputStream compressedStream = resultSet.getBinaryStream(pos++);
							String tmp = Gzip.decompress(compressedStream);
							//trial.setField(i, tmp);
							if (tmp != null && tmp.length() > 0) {
								trial.setField(XML_METADATA, tmp);
								trial.parseMetaData(tmp);
							}
							xmlSet = true;
							trial.setXmlMetaDataLoaded(true);
						}
					} 
					else 
					{
						if (database.getTrialFieldNames()[i].equalsIgnoreCase(XML_METADATA)) {
							if(getXMLMetadata){
								if (xmlSet == false) {
									trial.setField(i, resultSet.getString(pos++));
									trial.setXmlMetaDataLoaded(true);
								}
							}
						} 
						else 
						{
							trial.setField(i, resultSet.getString(pos++));
						}
					}
				}



				trials.addElement(trial);
			}
			resultSet.close();

			// get the function details
			Enumeration en = trials.elements();
			Trial trial;
			while (en.hasMoreElements()) {
				trial = (Trial) en.nextElement();
				trial.getTrialMetrics(db);
			}

			Collections.sort(trials);

			return trials;

		} catch (Exception ex) {
			ex.printStackTrace();
			return null;
		}

	}

	public int saveTrial(DB db) {
		boolean itExists = exists(db);
		int newTrialID = 0;

		try {
			database = db.getDatabase();
			// determine if we have a data meta-data item
			boolean haveDate = false;
			java.sql.Timestamp timestamp = null;
			String dateString = null;

			if (getMetaData() != null) {
				dateString = (String) getMetaData().get("UTC Time");
			}
			if (dateString == null) {
				if (getDataSource() != null && getDataSource().getAllThreads() != null
						&& ((Thread) getDataSource().getAllThreads().get(0)).getMetaData() != null) {
					dateString = (String) ((Thread) getDataSource().getAllThreads().get(0)).getMetaData().get("UTC Time");
				}
			}
			if (dateString != null) {
				try {
					Date date = DataSource.dateTime.parse(dateString);
					timestamp = new java.sql.Timestamp(date.getTime());
					haveDate = true;
				} catch (java.text.ParseException e) {
					e.printStackTrace();
				}
			}

			// FIRST!  Check if the trial table has a metadata column
			checkForMetadataColumn(db);
			// SECOND!  Check if the trial table has a zipped metadata column
			checkForMetadataColumn2(db);

			// get the fields since this is an insert
			if (!itExists) {
				Trial.getMetaData(db);
				this.fields = new String[database.getTrialFieldNames().length];
			}

			if (this.getDataSource() != null) {
				// If the user is simply manipulating apps/exps/trials in the treeview
				// there may not be a dataSource for this trial (it isn't loaded)
				this.setField("node_count", Integer.toString(1 + this.getDataSource().getMaxNCTNumbers()[0]));
				this.setField("contexts_per_node", Integer.toString(1 + this.getDataSource().getMaxNCTNumbers()[1]));
				this.setField("threads_per_context", Integer.toString(1 + this.getDataSource().getMaxNCTNumbers()[2]));
			}

			// set the other metadata, if it exists
			// UNCOMMENT THIS WHEN WE KNOW FOR SURE WHAT THE TAU
			// METADATA WILL LOOK LIKE 
			if (getDataSource() != null) {
				String tmp = getDataSource().getMetadataString();
				if (tmp != null && tmp.length() > 0) {
					setField(XML_METADATA, null);
					setField(XML_METADATA_GZ, tmp);
				}
			}

			// Check if the date column exists and is a timestamp
			boolean dateColumnFound = false;
			for (int i = 0; i < this.getNumFields(); i++) {
				if (getFieldName(i).equals("date")) {
					if (getFieldType(i) == java.sql.Types.TIMESTAMP) {
						dateColumnFound = true;
					}
				}
			}

			StringBuffer buf = new StringBuffer();
			if (itExists) {
				buf.append("UPDATE " + db.getSchemaPrefix() + "trial SET name = ?, experiment = ?");
				for (int i = 0; i < this.getNumFields(); i++) {
					if (DBConnector.isWritableType(this.getFieldType(i))) {
						buf.append(", " + this.getFieldName(i) + " = ?");
					}
				}

				if (haveDate && dateColumnFound) {
					buf.append(", date = ?");
				}

				buf.append(" WHERE id = ?");
			} else {
				buf.append("INSERT INTO " + db.getSchemaPrefix() + "trial (name, experiment");
				for (int i = 0; i < this.getNumFields(); i++) {
					if (DBConnector.isWritableType(this.getFieldType(i)))
						buf.append(", " + this.getFieldName(i));
				}

				if (haveDate && dateColumnFound) {
					buf.append(", date");
				}
				buf.append(") VALUES (?, ?");
				for (int i = 0; i < this.getNumFields(); i++) {
					if (DBConnector.isWritableType(this.getFieldType(i)))
						buf.append(", ?");
				}
				if (haveDate && dateColumnFound) {
					buf.append(", ?");
				}
				buf.append(")");
			}

			//System.out.println(buf.toString());
			PreparedStatement statement = db.prepareStatement(buf.toString());

			int pos = 1;

			statement.setString(pos++, name);
			statement.setInt(pos++, experimentID);
			for (int i = 0; i < this.getNumFields(); i++) {
				if (DBConnector.isWritableType(this.getFieldType(i))) {
					if (this.getFieldName(i).equalsIgnoreCase(XML_METADATA_GZ)) {
						if (this.getField(i) == null) {
							statement.setNull(pos++, this.getFieldType(i));
						} else {
							byte[] compressed = Gzip.compress(this.getField(i));
							//System.out.println("gzip data is " + compressed.length + " bytes in size");
							ByteArrayInputStream in = new ByteArrayInputStream(compressed);
							statement.setBinaryStream(pos++, in, compressed.length);
						}
					} else {
						int type = this.getFieldType(i);
						if (this.getField(i) == null) {
							statement.setNull(pos++, type);
						} else if (type == java.sql.Types.VARCHAR || type == java.sql.Types.CLOB
								|| type == java.sql.Types.LONGVARCHAR) {
							statement.setString(pos++, this.getField(i));
						} else if (type == java.sql.Types.INTEGER) {
							statement.setInt(pos++, Integer.parseInt(this.getField(i)));
						} else if (type == java.sql.Types.DECIMAL || type == java.sql.Types.DOUBLE
								|| type == java.sql.Types.FLOAT) {
							statement.setDouble(pos++, Double.parseDouble(this.getField(i)));
						} else if (type == java.sql.Types.TIME || type == java.sql.Types.TIMESTAMP) {
							statement.setString(pos++, this.getField(i));
						} else {
							// give up
							statement.setNull(pos++, type);
						}
					}
				}
			}

			if (haveDate && dateColumnFound) {
				statement.setTimestamp(pos++, timestamp);
			}

			if (itExists) {
				statement.setInt(pos, trialID);
			}

			statement.executeUpdate();
			statement.close();

			if (itExists) {
				newTrialID = trialID;
			} else {
				String tmpStr = new String();
				if (db.getDBType().compareTo("mysql") == 0)
					tmpStr = "select LAST_INSERT_ID();";
				else if (db.getDBType().compareTo("db2") == 0)
					tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
				else if (db.getDBType().compareTo("derby") == 0)
					tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
				else if (db.getDBType().compareTo("oracle") == 0)
					tmpStr = "select " + db.getSchemaPrefix() + "trial_id_seq.currval FROM dual";
				else
					tmpStr = "select currval('trial_id_seq');";
				newTrialID = Integer.parseInt(db.getDataItem(tmpStr));
			}

		} catch (SQLException e) {
			System.out.println("An error occurred while saving the trial.");
			e.printStackTrace();
		}
		return newTrialID;
	}

	private static void deleteAtomicLocationProfilesMySQL(DB db, int trialID) throws SQLException {
		Vector atomicEvents = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select id ");
		buf.append("from " + db.getSchemaPrefix() + "atomic_event where trial = ");
		buf.append(trialID);

		// System.out.println(buf.toString());

		StringBuffer deleteString = new StringBuffer();
		deleteString.append("DELETE FROM atomic_location_profile WHERE atomic_event IN (-1");

		ResultSet resultSet = db.executeQuery(buf.toString());
		while (resultSet.next() != false) {
			deleteString.append(", " + resultSet.getInt(1));
		}
		resultSet.close();

		//System.out.println("stmt = " + deleteString.toString() + ")");
		PreparedStatement statement = db.prepareStatement(deleteString.toString() + ")");
		statement.execute();
		statement.close();
	}

	private static void deleteIntervalLocationProfilesMySQL(DB db, int trialID) throws SQLException {
		Vector atomicEvents = new Vector();
		// create a string to hit the database
		StringBuffer buf = new StringBuffer();
		buf.append("select id ");
		buf.append("from " + db.getSchemaPrefix() + "interval_event where trial = ");
		buf.append(trialID);

		// System.out.println(buf.toString());

		StringBuffer deleteString = new StringBuffer();
		deleteString.append(" (-1");

		ResultSet resultSet = db.executeQuery(buf.toString());
		while (resultSet.next() != false) {
			deleteString.append(", " + resultSet.getInt(1));
		}
		resultSet.close();

		PreparedStatement statement = db.prepareStatement("DELETE FROM interval_location_profile WHERE interval_event IN"
				+ deleteString.toString() + ")");
		statement.execute();
		statement.close();

		statement = db.prepareStatement("DELETE FROM interval_mean_summary WHERE interval_event IN" + deleteString.toString()
				+ ")");
		statement.execute();
		statement.close();

		statement = db.prepareStatement("DELETE FROM interval_total_summary WHERE interval_event IN" + deleteString.toString()
				+ ")");
		statement.execute();
		statement.close();

	}

	public static void deleteTrial(DB db, int trialID) throws SQLException {
		// save this trial
		PreparedStatement statement = null;

		// delete from the atomic_location_profile table
		if (db.getDBType().compareTo("mysql") == 0) {

			Trial.deleteAtomicLocationProfilesMySQL(db, trialID);

			//                statement = db.prepareStatement(" DELETE atomic_location_profile.* FROM "
			//                        + db.getSchemaPrefix()
			//                        + "atomic_location_profile LEFT JOIN "
			//                        + db.getSchemaPrefix()
			//                        + "atomic_event ON atomic_location_profile.atomic_event = atomic_event.id WHERE atomic_event.trial = ?");
		} else {
			// Postgresql, oracle, and DB2?
			statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
					+ "atomic_location_profile WHERE atomic_event in (SELECT id FROM " + db.getSchemaPrefix()
					+ "atomic_event WHERE trial = ?)");
			statement.setInt(1, trialID);
			statement.execute();
			statement.close();
		}

		// delete the from the atomic_events table
		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "atomic_event WHERE trial = ?");
		statement.setInt(1, trialID);
		statement.execute();
		statement.close();

		// delete from the interval_location_profile table
		if (db.getDBType().compareTo("mysql") == 0) {

			Trial.deleteIntervalLocationProfilesMySQL(db, trialID);

			//                statement = db.prepareStatement(" DELETE interval_location_profile.* FROM "
			//                        + db.getSchemaPrefix()
			//                        + "interval_location_profile LEFT JOIN "
			//                        + db.getSchemaPrefix()
			//                        + "interval_event ON interval_location_profile.interval_event = interval_event.id WHERE interval_event.trial = ?");
		} else {
			// Postgresql and DB2?
			statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
					+ "interval_location_profile WHERE interval_event IN (SELECT id FROM " + db.getSchemaPrefix()
					+ "interval_event WHERE trial = ?)");
			statement.setInt(1, trialID);
			statement.execute();
			statement.close();
		}

		// delete from the interval_mean_summary table
		if (db.getDBType().compareTo("mysql") == 0) {
			//statement = db.prepareStatement(" DELETE interval_mean_summary.* FROM interval_mean_summary LEFT JOIN interval_event ON interval_mean_summary.interval_event = interval_event.id WHERE interval_event.trial = ?");
		} else {
			// Postgresql and DB2?
			statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
					+ "interval_mean_summary WHERE interval_event IN (SELECT id FROM " + db.getSchemaPrefix()
					+ "interval_event WHERE trial = ?)");
			statement.setInt(1, trialID);
			statement.execute();
			statement.close();
		}

		if (db.getDBType().compareTo("mysql") == 0) {
			//statement = db.prepareStatement(" DELETE interval_total_summary.* FROM interval_total_summary LEFT JOIN interval_event ON interval_total_summary.interval_event = interval_event.id WHERE interval_event.trial = ?");
		} else {
			// Postgresql and DB2?
			statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
					+ "interval_total_summary WHERE interval_event IN (SELECT id FROM " + db.getSchemaPrefix()
					+ "interval_event WHERE trial = ?)");
			statement.setInt(1, trialID);
			statement.execute();
			statement.close();
		}

		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "interval_event WHERE trial = ?");
		statement.setInt(1, trialID);
		statement.execute();
		statement.close();

		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "metric WHERE trial = ?");
		statement.setInt(1, trialID);
		statement.execute();
		statement.close();

		statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "trial WHERE id = ?");
		statement.setInt(1, trialID);
		statement.execute();
		statement.close();
	}

	private boolean exists(DB db) {
		boolean retval = false;
		try {
			PreparedStatement statement = db.prepareStatement("SELECT name FROM " + db.getSchemaPrefix() + "trial WHERE id = ?");
			statement.setInt(1, trialID);
			ResultSet results = statement.executeQuery();
			while (results.next() != false) {
				retval = true;
				break;
			}
			results.close();
		} catch (SQLException e) {
			System.out.println("An error occurred while saving the application.");
			e.printStackTrace();
		}
		return retval;
	}

	private void readObject(ObjectInputStream aInputStream) throws ClassNotFoundException, IOException {
		// always perform the default de-serialization first
		aInputStream.defaultReadObject();
		//        if (fieldNames == null)
		//            fieldNames = (String[]) aInputStream.readObject();
		//        if (fieldTypes == null)
		//            fieldTypes = (int[]) aInputStream.readObject();
	}

	private void writeObject(ObjectOutputStream aOutputStream) throws IOException {
		// always perform the default serialization first
		aOutputStream.defaultWriteObject();
		//        aOutputStream.writeObject(fieldNames);
		//        aOutputStream.writeObject(fieldTypes);
	}

	/**
	 *  hack - needed to delete meta so that it is reloaded each time a new database is created.
	 */
	//    public void removeMetaData() {
	//        fieldNames = null;
	//        fieldTypes = null;
	//    }
	public void checkForMetadataColumn(DB db) {
		String[] columns = Trial.getFieldNames(db);
		boolean found = false;
		// loop through the column names, and see if we have this column already
		for (int i = 0; i < columns.length; i++) {
			if (columns[i].equalsIgnoreCase(XML_METADATA)) {
				found = true;
				break;
			}
		}
		if (!found) {
			StringBuffer sql = new StringBuffer();
			// create the column in the database
			sql.append("ALTER TABLE " + db.getSchemaPrefix() + "trial ADD COLUMN ");
			sql.append(XML_METADATA);
			if (db.getDBType().equalsIgnoreCase("oracle")) {
				sql.append(" CLOB"); // defaults to 4 GB max
			} else if (db.getDBType().equalsIgnoreCase("derby")) {
				sql.append(" CLOB"); // defaults to 1 MB max
			} else if (db.getDBType().equalsIgnoreCase("db2")) {
				sql.append(" CLOB"); // defaults to 1 GB max
			} else if (db.getDBType().equalsIgnoreCase("mysql")) {
				sql.append(" TEXT"); // defaults to 64 KB max
			} else if (db.getDBType().equalsIgnoreCase("postgresql")) {
				sql.append(" TEXT"); // defaults to 4 GB max
			}

			try {
				db.execute(sql.toString());
			} catch (SQLException e) {
				System.err.println("Unable to add " + XML_METADATA + " column to trial table.");
				e.printStackTrace();
			}
		}
	}

	public void checkForMetadataColumn2(DB db) {
		String[] columns = Trial.getFieldNames(db);
		boolean found = false;
		// loop through the column names, and see if we have this column already
		for (int i = 0; i < columns.length; i++) {
			if (columns[i].equalsIgnoreCase(XML_METADATA_GZ)) {
				found = true;
				break;
			}
		}
		if (!found) {
			StringBuffer sql = new StringBuffer();
			// create the column in the database
			sql.append("ALTER TABLE " + db.getSchemaPrefix() + "trial ADD COLUMN ");
			sql.append(XML_METADATA_GZ);
			if (db.getDBType().equalsIgnoreCase("oracle")) {
				sql.append(" BLOB"); // defaults to 4 GB max
			} else if (db.getDBType().equalsIgnoreCase("derby")) {
				sql.append(" BLOB"); // defaults to 1 MB max
			} else if (db.getDBType().equalsIgnoreCase("db2")) {
				sql.append(" BLOB"); // defaults to 1 GB max
			} else if (db.getDBType().equalsIgnoreCase("mysql")) {
				sql.append(" LONGBLOB"); // defaults to 64 KB max
			} else if (db.getDBType().equalsIgnoreCase("postgresql")) {
				sql.append(" BYTEA"); // defaults to 4 GB max
			}

			try {
				db.execute(sql.toString());
			} catch (SQLException e) {
				System.err.println("Unable to add " + XML_METADATA_GZ + " column to trial table.");
				e.printStackTrace();
			}
		}
	}

	public Map getMetaData() {
		return metaData;
	}

	public void setMetaData(Map metaDataMap) {
		this.metaData = metaDataMap;
	}

	public Database getDatabase() {
		return database;
	}

	public void setDatabase(Database database) {
		this.database = database;
		fields = new String[database.getTrialFieldNames().length];

	}

	public Map getUncommonMetaData() {
		return uncommonMetaData;
	}

	public void setUncommonMetaData(Map uncommonMetaData) {
		this.uncommonMetaData = uncommonMetaData;
	}

	public int compareTo(Object arg0) {
		return alphanum.compare(this.getName(), ((Trial) arg0).getName());
	}

}
