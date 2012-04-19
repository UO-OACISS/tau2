package edu.uoregon.tau.perfdmf;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.util.Collections;
import java.util.Vector;

import edu.uoregon.tau.perfdmf.database.DB;

public class TAUdbTrial {
	public static int saveTrialTAUdb(DB db, int trialID, DataSource dataSource,
			String name) {
		boolean itExists = exists(db, trialID);
		int newTrialID = 0;
		if (dataSource == null) {
			System.out
					.println("Given a null Datasource, must be loaded first, TAUdbTrial");
			return trialID;
		}
		if (itExists) {
			// TODO: Write "update" code
			System.out
					.println("Updates to TAUdb trials have not been implemented yet");
			return trialID;
		}

		try {

			java.sql.Timestamp collection_date = new Timestamp(
					System.currentTimeMillis());
			int node_count = dataSource.getMaxNode();
			int contexts_per_node = dataSource.getMaxContextPerNode();
			int threads_per_context =dataSource.getMaxThreadsPerContext();
			int datasource_id = dataSource.getFileType();
			int total_threads = dataSource.getNumThreads();

			String sql = "INSERT INTO "
					+ db.getSchemaPrefix()
					+ "trial (name, collection_date, data_source,  node_count, contexts_per_node, threads_per_context, total_threads)"
					+ "VALUES (?,?,?,?,?,?,?" +
					") ";
			PreparedStatement statement = db.prepareStatement(sql);
			statement.setString(1, name);

			statement.setTimestamp(2, collection_date);
			statement.setInt(3, datasource_id);

			statement.setInt(4, node_count);
			statement.setInt(5, contexts_per_node);
			statement.setInt(6, threads_per_context);
			statement.setInt(7, total_threads);

			statement.executeUpdate();
			statement.close();

			String tmpStr = new String();
			if (db.getDBType().compareTo("mysql") == 0)
				tmpStr = "select LAST_INSERT_ID();";
			else if (db.getDBType().compareTo("db2") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
			else if (db.getDBType().compareTo("derby") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
			else if (db.getDBType().compareTo("h2") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
			else if (db.getDBType().compareTo("oracle") == 0)
				tmpStr = "select " + db.getSchemaPrefix()
						+ "trial_id_seq.currval FROM dual";
			else
				tmpStr = "select currval('trial_id_seq');";
			newTrialID = Integer.parseInt(db.getDataItem(tmpStr));

		} catch (SQLException e) {
			System.out.println("An error occurred while saving the trial.");
			e.printStackTrace();
		}
		return newTrialID;
	}
	    private static boolean exists(DB db, int trialID) {
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
	            System.out.println("An error occurred while checking to see if the trial exists.");
	            e.printStackTrace();
	        }
	        return retval;
	    }
		public static Vector<Trial> getTrialList(DB db, boolean getXMLMetadata, String whereClause ) {
		try {

			Trial.getMetaData(db);

			// create a string to hit the database
			String buf = "SELECT t.id, t.name, metadata.app, metadata.exp, t.collection_date, t.data_source, "
					+ "t.node_count, t.contexts_per_node, t.threads_per_context, t.total_threads FROM "
					+ "( SELECT DISTINCT A.value as app, E.value as exp, A.trial as trial "
					+ "FROM "
					+ db.getSchemaPrefix()
					+ "primary_metadata A, "
					+ db.getSchemaPrefix()
					+ "primary_metadata E "
					+ "WHERE A.trial=E.trial AND A.name='Application' AND E.name='Experiment'"
					+ ") as metadata LEFT JOIN "
					+ db.getSchemaPrefix()
					+ "trial as t ON metadata.trial=t.id  " + whereClause;

			Vector<Trial> trials = new Vector<Trial>();

			ResultSet resultSet = db.executeQuery(buf.toString());
			while (resultSet.next() != false) {
				Trial trial = new Trial();
				trial.setDatabase(db.getDatabase());
				int pos = 1;
				trial.setID(resultSet.getInt(pos++));
				trial.setName(resultSet.getString(pos++));

				String appname = resultSet.getString(pos++);
				// TODO: Figure out what to do about the app ids
				// trial.setApplicationID(resultSet.getInt(pos++));

				String expanme = resultSet.getString(pos++);
				// TODO: Figure out what to do about the experiment ids
				// trial.setExperimentID();

				Database database = db.getDatabase();
				for (int i = 0; i < database.getTrialFieldNames().length; i++) {
					if (database.getTrialFieldNames()[i]
							.equalsIgnoreCase("collection_date")) {
						java.sql.Timestamp time = resultSet.getTimestamp(pos++);
						trial.setField(i, time.toString());
					} else {
						trial.setField(i, resultSet.getString(pos++));
					}
				}
				trials.addElement(trial);
			}
			resultSet.close();
			// TODO: Deal with adding the metrics to the trial
			// // get the function details
			// Enumeration<Trial> en = trials.elements();
			// Trial trial;
			// while (en.hasMoreElements()) {
			// trial = en.nextElement();
			// trial.getTrialMetrics(db);
			// }

			Collections.sort(trials);

			return trials;

		} catch (Exception ex) {
			ex.printStackTrace();
			return null;
		}
	}
		//This is for the Columns in the Trial table
		public static void getMetaData(DB db, boolean allColumns) {
	
				  
		            String[] fieldNames = new String[6];
		           //int[] fieldTypes = new int[typeList.size()];
		           // String[] fieldTypeNames = new String[typeList.size()];
		            fieldNames[0] = "collection_date";
		            fieldNames[1] = "data_source";
		            fieldNames[2] = "node_count";
		            fieldNames[3] = "contexts_per_node";
		            fieldNames[4] = "threads_per_context";
		            fieldNames[5] = "total_threads";
		            db.getDatabase().setTrialFieldNames(fieldNames);
		            //db.getDatabase().setTrialFieldTypes(fieldTypes);
		            //db.getDatabase().setTrialFieldTypeNames(fieldTypeNames);
		
		}
}
