package edu.uoregon.tau.perfexplorer.server;

import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.database.DB;

public class QueryManager {
	/**
	 * This method gets a list of trial objects from the database
	 * 
	 * @throws PerfExplorerException
	 */
	@SuppressWarnings("unchecked")  // for Trial.getTrialList() call
	public static List<Trial> getTrialList (String criteria) {
		PerfExplorerServer server = PerfExplorerServer.getServer();
		List<Trial> list = null;
		try {
			// get the database connection
			DB db = server.getDB();
			// PerfDMF has some syntax shortcuts, so make those
			// changes in our criteria
			String whereClause = " where " + fixClause(criteria, db);
			// ask the API for the trials
			list = Trial.getTrialList(db, whereClause);
		} catch (Exception e) {
			System.err.println("ERROR: Couldn't the list of trials!");
            System.err.println(e.getMessage());
			e.printStackTrace();
		}
		if (list == null) {
			// return an empty list
			list = new ArrayList<Trial>();
		}
		return list;
	}

	private static String fixClause (String inString, DB db) {
		// change the table names
		String outString = inString.replaceAll("trial.", "t.").replaceAll("experiment.", "e.").replaceAll("application.", "a.");
		// fix the oracle specific stuff
		if (db.getDBType().equalsIgnoreCase("oracle")) {
			outString = outString.replaceAll("exclusive", "exec");
		}
		// and so forth
		return outString;
	}

}
