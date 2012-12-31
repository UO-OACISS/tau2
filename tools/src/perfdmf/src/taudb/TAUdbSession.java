/**
 * 
 */
package edu.uoregon.tau.perfdmf.taudb;

import java.io.File;
import java.sql.SQLException;
import java.util.Map;

import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.database.DB;

/**
 * @author khuck
 *
 */
public class TAUdbSession {

	private Map<Integer, TAUdbDataSourceType> sources = null;
	private TAUdbDatabaseAPI api = null;
	private boolean connected = false;
	private TAUdbTrial currentTrial = null;
	
	private DB db;
	
	/**
	 * 
	 */
	public TAUdbSession(String configName, boolean prompt) {
		api = new TAUdbDatabaseAPI();
		String homedir = System.getenv("HOME");
		String configFile = homedir + File.separator + ".ParaProf" + File.separator + "perfdmf.cfg." + configName;
		try {
			api.initialize(configFile, false);
			connected = true;
			sources = TAUdbDataSourceType.getDataSourceTypes(this);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	public TAUdbSession(DB db){
		this.db = db;
	}
	
	public void close() {
		if (api != null)
			api.db().close();
		connected = false;
	}
	
	public TAUdbTrial getCurrentTrial() {
		return currentTrial;
	}
	
	public void finalize () {
		if (connected)
			System.err.println("Forgot to close the database connection!");
		close();
	}
	
	public Map<Integer, TAUdbDataSourceType> getDataSources() {
		return sources;
	}
	
	public DB getDB() {
		if(api == null)
			return db;
		return api.db();
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TAUdbSession session = new TAUdbSession("callpath", false);
		System.out.print("Connected to: ");
		System.out.println(session.api.db().getConnectString());
		session.close();
	}

}
