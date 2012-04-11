package edu.uoregon.tau.perfdmf;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;

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
					+ "VALUES (?,?,?,?,?,?,?) ";
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
}
