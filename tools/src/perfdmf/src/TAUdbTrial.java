package edu.uoregon.tau.perfdmf;

import java.sql.DatabaseMetaData;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Vector;

import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.database.DBConnector;

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
						System.out.println(time.toString());

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
		public static void getMetaData(DB db, boolean allColumns) {
			  try {
		            ResultSet resultSet = null;

		            //String trialFieldNames[] = null;
		            //int trialFieldTypes[] = null;

		            DatabaseMetaData dbMeta = db.getMetaData();

		            if ((db.getDBType().compareTo("oracle") == 0) || (db.getDBType().compareTo("derby") == 0) || (db.getDBType().compareTo("h2") == 0)
		                    || (db.getDBType().compareTo("db2") == 0)) {
		                resultSet = dbMeta.getColumns(null, null, "TRIAL", "%");
		            } else {
		                resultSet = dbMeta.getColumns(null, null, "trial", "%");
		            }

		            Vector<String> nameList = new Vector<String>();
		            Vector<Integer> typeList = new Vector<Integer>();
		            List<String> typeNames = new ArrayList<String>();
		            List<Integer> columnSizes = new ArrayList<Integer>();
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
		                fieldNames[i] = nameList.get(i);
		                fieldTypes[i] = typeList.get(i).intValue();
		                if (columnSizes.get(i).intValue() > 255) {
		                    fieldTypeNames[i] = typeNames.get(i) + "(" + columnSizes.get(i).toString() + ")";
		                } else {
		                    fieldTypeNames[i] = typeNames.get(i);
		                }
		            }

		            db.getDatabase().setTrialFieldNames(fieldNames);
		            db.getDatabase().setTrialFieldTypes(fieldTypes);
		            db.getDatabase().setTrialFieldTypeNames(fieldTypeNames);
		        } catch (SQLException e) {
		            e.printStackTrace();
		        }			
		}
}
