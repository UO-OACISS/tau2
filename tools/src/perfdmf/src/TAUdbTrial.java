package edu.uoregon.tau.perfdmf;

import java.io.ByteArrayInputStream;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Date;

import edu.uoregon.tau.common.Gzip;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.database.DBConnector;

public class TAUdbTrial {
	 public static int saveTrialTAUdb(DB db, int trialID, DataSource dataSource) {
	        boolean itExists = exists(db, trialID);
	        int newTrialID = 0;
	  
//	        try {
	            Database database = db.getDatabase();


	            // get the fields since this is an insert
//	            if (!itExists) {
//	                Trial.getMetaData(db);
//	                this.fields = new String[database.getTrialFieldNames().length];
//	            }
	            int node_count = 0;
	            int contexts_per_node = 0;
	            int threads_per_context = 0;
	            if (dataSource != null) {
	                // If the user is simply manipulating apps/exps/trials in the treeview
	                // there may not be a dataSource for this trial (it isn't loaded)
	                node_count = 1 + dataSource.getMaxNCTNumbers()[0];
	                contexts_per_node=1 + dataSource.getMaxNCTNumbers()[1];
	                threads_per_context=1 + dataSource.getMaxNCTNumbers()[2];
	            }




	            StringBuffer buf = new StringBuffer();
	            //TODO: Write "update" code
//	            if (itExists) {
//	                buf.append("UPDATE " + db.getSchemaPrefix() + "trial SET name = ?, experiment = ?");
//	                for (int i = 0; i < this.getNumFields(); i++) {
//	                    if (DBConnector.isWritableType(this.getFieldType(i))) {
//	                        buf.append(", " + this.getFieldName(i) + " = ?");
//	                    }
//	                }
//
//	                if (haveDate && dateColumnFound) {
//	                    buf.append(", date = ?");
//	                }
//
//	                buf.append(" WHERE id = ?");
//	            } else {
	                buf.append("INSERT INTO " + db.getSchemaPrefix() 
	                		+ "trial (name, collection_date, data_source, node_count, contexts_per_node, threads_per_context");
	          
	                buf.append(")");
//	            }

//	            //System.out.println(buf.toString());
//	            PreparedStatement statement = db.prepareStatement(buf.toString());
//
//	            int pos = 1;
//
//	            statement.setString(pos++, name);
//	            statement.setInt(pos++, experimentID);
//	            for (int i = 0; i < this.getNumFields(); i++) {
//	                if (DBConnector.isWritableType(this.getFieldType(i))) {
//	                    if (this.getFieldName(i).equalsIgnoreCase(XML_METADATA_GZ)) {
//	                        if (this.getField(i) == null) {
//	                            statement.setNull(pos++, this.getFieldType(i));
//	                        } else {
//	                            byte[] compressed = Gzip.compress(this.getField(i));
//	                            //System.out.println("gzip data is " + compressed.length + " bytes in size");
//	                            ByteArrayInputStream in = new ByteArrayInputStream(compressed);
//	                            statement.setBinaryStream(pos++, in, compressed.length);
//	                        }
//	                    } else {
//	                        int type = this.getFieldType(i);
//	                        if (this.getField(i) == null) {
//	                            statement.setNull(pos++, type);
//	                        } else if (type == java.sql.Types.VARCHAR || type == java.sql.Types.CLOB
//	                                || type == java.sql.Types.LONGVARCHAR) {
//	                            statement.setString(pos++, this.getField(i));
//	                        } else if (type == java.sql.Types.INTEGER) {
//	                            statement.setInt(pos++, Integer.parseInt(this.getField(i)));
//	                        } else if (type == java.sql.Types.DECIMAL || type == java.sql.Types.DOUBLE
//	                                || type == java.sql.Types.FLOAT) {
//	                            statement.setDouble(pos++, Double.parseDouble(this.getField(i)));
//	                        } else if (type == java.sql.Types.TIME || type == java.sql.Types.TIMESTAMP) {
//	                            statement.setString(pos++, this.getField(i));
//	                        } else {
//	                            // give up
//	                            statement.setNull(pos++, type);
//	                        }
//	                    }
//	                }
//	            }
//
//	            if (haveDate && dateColumnFound) {
//	                statement.setTimestamp(pos++, timestamp);
//	            }
//
//	            if (itExists) {
//	                statement.setInt(pos, trialID);
//	            }
//
//	            statement.executeUpdate();
//	            statement.close();

//	            if (itExists) {
//	                newTrialID = trialID;
//	            } else {
//	                String tmpStr = new String();
//	                if (db.getDBType().compareTo("mysql") == 0)
//	                    tmpStr = "select LAST_INSERT_ID();";
//	                else if (db.getDBType().compareTo("db2") == 0)
//	                    tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
//	                else if (db.getDBType().compareTo("derby") == 0)
//	                    tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
//	                else if (db.getDBType().compareTo("h2") == 0)
//	                    tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
//	                else if (db.getDBType().compareTo("oracle") == 0)
//	                    tmpStr = "select " + db.getSchemaPrefix() + "trial_id_seq.currval FROM dual";
//	                else
//	                    tmpStr = "select currval('trial_id_seq');";
//	                newTrialID = Integer.parseInt(db.getDataItem(tmpStr));
	           // }

//	        } catch (SQLException e) {
//	            System.out.println("An error occurred while saving the trial.");
//	            e.printStackTrace();
//	        }
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
