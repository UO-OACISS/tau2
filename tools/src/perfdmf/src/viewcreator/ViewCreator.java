package edu.uoregon.tau.perfdmf.viewcreator;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfdmf.database.DB;

public class ViewCreator {
	DB db;
	List<String> metadataNames;
	public ViewCreator(DB db){
		this.db = db;
		this.metadataNames = new ArrayList<String>();
	}
	public List<String> getMetadataNames()  {
		metadataNames = new ArrayList<String>();
		try {
		String sql = "SELECT DISTINCT name FROM "
				+ db.getSchemaPrefix()
				+ "primary_metadata ORDER BY name";
		PreparedStatement statement;
		
			statement = db.prepareStatement(sql);

		ResultSet results = statement.executeQuery();
		while(results.next()){
			metadataNames.add(results.getString(1));
		}
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	System.out.println(metadataNames);
    	
		return metadataNames;
	}
	public int saveView(String parent, String name, String conjoin){
		
		// /* simple view where the metadata field "Application" is equal to "application" */
		// INSERT INTO taudb_view (parent, name, conjoin) VALUES (NULL, 'Test View', 'and');
		// INSERT INTO taudb_view_parameter (taudb_view, table_name, column_name, operator, value)
		//             VALUES (2, 'primary_metadata', 'Application', '=', 'application');
		
		StringBuffer buf = new StringBuffer();
		buf.append("INSERT INTO " + db.getSchemaPrefix()
				+ "taudb_view  (parent, name, conjoin) ");

		buf.append(" VALUES (?,?,?)");
		PreparedStatement statement;
		try {
			statement = db.prepareStatement(buf.toString());
if (parent == null)
	statement.setNull(1,java.sql.Types.VARCHAR);
else
			statement.setString(1, parent);
			statement.setString(2, name);
			statement.setString(3, conjoin);
			System.out.println(statement.toString());
 
			statement.executeUpdate();
			statement.close();
			
            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0) {
                tmpStr = "select LAST_INSERT_ID();";
            } else if (db.getDBType().compareTo("db2") == 0) {
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM taudb_view";
            } else if (db.getDBType().compareTo("sqlite") == 0) {
                tmpStr = "select seq from sqlite_sequence where name = 'taudb_view'";
            } else if (db.getDBType().compareTo("derby") == 0) {
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM taudb_view";
            } else if (db.getDBType().compareTo("h2") == 0) {
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM taudb_view";
            } else if (db.getDBType().compareTo("oracle") == 0) {
                tmpStr = "SELECT " + db.getSchemaPrefix() + "taudb_view_id_seq.currval FROM DUAL";
            } else { // postgresql 
                tmpStr = "select currval('taudb_view_id_seq');";
            }
            return Integer.parseInt(db.getDataItem(tmpStr));
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return -1;
	}
	
	public void saveViewParameter(int viewID, String table, String column,
			String operator, String value) {

		StringBuffer buf = new StringBuffer();
		buf.append("INSERT INTO "
				+ db.getSchemaPrefix()
				+ "taudb_view_parameter   (taudb_view, table_name, column_name, operator, value) ");
		buf.append(" VALUES (?,?,?,?,?)");
		PreparedStatement statement;
		try {
			statement = db.prepareStatement(buf.toString());

			statement.setInt(1, viewID);
			statement.setString(2, table);
			statement.setString(3, column);
			statement.setString(4, operator);
			statement.setString(5, value);
			System.out.println(statement.toString());

			statement.executeUpdate();
			statement.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
