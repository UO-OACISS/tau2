package edu.uoregon.tau.perfdmf.viewcreator;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import edu.uoregon.tau.perfdmf.database.DB;

public class ViewCreator {
	DB db;
	public ViewCreator(DB db){
		this.db = db;
	}
	public List<String> getMetadataNames()  {
		ArrayList<String> names = new ArrayList<String>();
		try {
		String sql = "SELECT DISTINCT name FROM "
				+ db.getSchemaPrefix()
				+ "primary_metadata";
		PreparedStatement statement;
		
			statement = db.prepareStatement(sql);

		ResultSet results = statement.executeQuery();
		while(results.next()){
			names.add(results.getString(0));
		}
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	System.out.println(names);

		return names;
	}
		
	

}
