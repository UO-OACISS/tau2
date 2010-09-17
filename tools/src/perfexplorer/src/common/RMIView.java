package edu.uoregon.tau.perfexplorer.common;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import javax.swing.tree.DefaultMutableTreeNode;

import edu.uoregon.tau.perfdmf.database.DB;

/**
 * This class is the RMI class which contains the tree of views to be 
 * constructed in the PerfExplorerClient.
 *
 * <P>CVS $Id: RMIView.java,v 1.11 2009/02/27 00:45:09 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class RMIView implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7198343642137106238L;
	private static List<String> fieldNames = null;
	private List<String> fields = null;
	private DefaultMutableTreeNode node = null;

	public RMIView () {
		fields = new ArrayList<String>();
	}

	public static Iterator<String> getFieldNames(DB db) {
		if (fieldNames == null) {
			fieldNames = new ArrayList<String>();
			try {
				ResultSet resultSet = null;
				DatabaseMetaData dbMeta = db.getMetaData();
				if (db.getDBType().compareTo("oracle") == 0) {
					resultSet = dbMeta.getColumns(null, null, "TRIAL_VIEW", "%");
				} else if (db.getDBType().compareTo("derby") == 0) {
					resultSet = dbMeta.getColumns(null, null, "TRIAL_VIEW", "%");
				} else if (db.getDBType().compareTo("db2") == 0) {
					resultSet = dbMeta.getColumns(null, null, "TRIAL_VIEW", "%");
				} else {
					resultSet = dbMeta.getColumns(null, null, "trial_view", "%");
				}

				int i = 0;
				while (resultSet.next() != false) {
					String name =
					resultSet.getString("COLUMN_NAME").toUpperCase();
					fieldNames.add(name);
					i++;
				}
				resultSet.close();

			} catch (SQLException e) {
				System.err.println("DATABASE EXCEPTION: " + e.toString());
				e.printStackTrace();
			}
		}
		return fieldNames.iterator();
	}

	public static Iterator<String> getFieldNames() {
		// assumes not null!
		return fieldNames.iterator();
	}

	public static int getFieldCount() {
		return fieldNames.size();
	}

	public void addField(String value) {
		fields.add(value);
	}

	public String getField(String fieldName) {
		int i = fieldNames.indexOf(fieldName.toUpperCase());
		if (i == -1)
			return new String("");
		else
			return fields.get(i);
	}

	public String getField(int i) {
		return fields.get(i);
	}

	public static String getFieldName(int i) {
		return fieldNames.get(i);
	}

	// suppress warning about aInputStream.readObject() call.
	@SuppressWarnings("unchecked")
	private void readObject (ObjectInputStream aInputStream) throws ClassNotFoundException, IOException {
		// perform the default serialization for this object
		aInputStream.defaultReadObject();
		if (fieldNames == null)
			fieldNames = (List<String>) aInputStream.readObject();
	}

	private void writeObject (ObjectOutputStream aOutputStream) throws IOException {
		// perform the default serialization for this object
		aOutputStream.defaultWriteObject();
		aOutputStream.writeObject(fieldNames);
	}

	public String toString() {
		return getField("NAME");
	}

	public void setDMTN(DefaultMutableTreeNode node) {
		this.node = node;
	}

	public DefaultMutableTreeNode getDMTN() {
		return this.node;
	}
}
