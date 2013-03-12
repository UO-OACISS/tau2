package edu.uoregon.tau.perfdmf.taudb;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

/**
 * @author khuck
 * 
 */
public class TAUdbDataSourceType {

	private int id = 0; // database ID
	private String name = null; // datasource name
	private String description = null; // datasource description

	public TAUdbDataSourceType() {
		super();
	}

	public TAUdbDataSourceType(int id, String name, String description) {
		this.id = id;
		this.name = name;
		this.description = description;
	}

	/**
	 * @return the id
	 */
	public int getId() {
		return id;
	}

	/**
	 * @param id
	 *            the id to set
	 */
	public void setId(int id) {
		this.id = id;
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @param name
	 *            the name to set
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * @return the description
	 */
	public String getDescription() {
		return description;
	}

	/**
	 * @param description
	 *            the description to set
	 */
	public void setDescription(String description) {
		this.description = description;
	}

	public String toString() {
		StringBuffer b = new StringBuffer();
		b.append("id = " + id + ", ");
		b.append("name = " + name + ", ");
		b.append("description = " + description);
		return b.toString();
	}

	public static Map<Integer, TAUdbDataSourceType> getDataSourceTypes(
			TAUdbSession session) {
		Map<Integer, TAUdbDataSourceType> sources = new HashMap<Integer, TAUdbDataSourceType>();
		String query = "select id, name, description from data_source;";
		try {
			PreparedStatement statement = session.getDB().prepareStatement(
					query);
			ResultSet results = statement.executeQuery();
			while (results.next()) {
				Integer id = results.getInt(1);
				String name = results.getString(2);
				String description = results.getString(3);
				TAUdbDataSourceType source = new TAUdbDataSourceType(id, name,
						description);
				sources.put(id, source);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return sources;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TAUdbSession session = new TAUdbSession("callpath", false);
		Map<Integer, TAUdbDataSourceType> sources = TAUdbDataSourceType
				.getDataSourceTypes(session);
		for (Integer id : sources.keySet()) {
			System.out.println(sources.get(id).toString());
		}
		session.close();
	}

}
