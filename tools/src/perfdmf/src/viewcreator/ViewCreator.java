package edu.uoregon.tau.perfdmf.viewcreator;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.database.DB;

public class ViewCreator {
	DB db;
	List<String> metadataNames;
	public ViewCreator(DB db){
		this.db = db;
		this.metadataNames = new ArrayList<String>();
	}
	public List<String> getMetadataNames()  {
		
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
		
	

}
