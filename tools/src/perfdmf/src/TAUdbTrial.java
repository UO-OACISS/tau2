package edu.uoregon.tau.perfdmf;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.common.MetaDataMap.MetaDataKey;
import edu.uoregon.tau.common.MetaDataMap.MetaDataValue;
import edu.uoregon.tau.perfdmf.database.DB;

public class TAUdbTrial extends Trial {
	public  int saveTrial(DB db) {
		return TAUdbTrial.saveTrialTAUdb(db, trialID, dataSource, name);
	}
	public static  int saveTrialTAUdb(DB db, int trialID, DataSource dataSource, String name) {
		if(db.getSchemaVersion()<1){
			System.err.println("You can't save a TAUdbTrial to a PerfDMF database.");
			return -1;
		}
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
			int node_count = dataSource.getMaxNode();
			int contexts_per_node = dataSource.getMaxContextPerNode();
			int threads_per_context =dataSource.getMaxThreadsPerContext();
			int datasource_id = dataSource.getFileType();
			int total_threads = dataSource.getNumThreads();

			String sql = "INSERT INTO "
					+ db.getSchemaPrefix()
					+ "trial (name, data_source,  node_count, contexts_per_node, threads_per_context, total_threads)"
					+ "VALUES (?,?,?,?,?,?" +
					") ";
			PreparedStatement statement = db.prepareStatement(sql);
			statement.setString(1, name);

			statement.setInt(2, datasource_id);

			statement.setInt(3, node_count);
			statement.setInt(4, contexts_per_node);
			statement.setInt(5, threads_per_context);
			statement.setInt(6, total_threads);

			statement.executeUpdate();
			statement.close();

			String tmpStr = new String();
			if (db.getDBType().compareTo("mysql") == 0)
				tmpStr = "select LAST_INSERT_ID();";
			else if (db.getDBType().compareTo("db2") == 0)
				tmpStr = "select IDENTITY_VAL_LOCAL() FROM trial";
            else if (db.getDBType().compareTo("sqlite") == 0)
                tmpStr = "select seq from sqlite_sequence where name = 'trial'";
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
		            String buf = new String();
		            if (whereClause.contains("application") || whereClause.contains("experiment")) {
		            	buf = "SELECT t.id, t.name, t.data_source, t.node_count, t.contexts_per_node, t.threads_per_context, t.total_threads, metadata.app, metadata.exp FROM " +
			            "( SELECT DISTINCT A.value as app, E.value as exp, A.trial as trial " +
			            "FROM " +
			            db.getSchemaPrefix()+"primary_metadata A, " +
			            		db.getSchemaPrefix()+"primary_metadata E " +
			            "WHERE A.trial=E.trial AND A.name='Application' AND E.name='Experiment'" +
			            ") as metadata LEFT JOIN " +
			            db.getSchemaPrefix()+"trial as t ON metadata.trial=t.id  " + whereClause ;
		            } else {// faster query for selecting trials
			            buf = "SELECT t.id, t.name, t.data_source, t.node_count, t.contexts_per_node, " +
			            "t.threads_per_context, t.total_threads, m1.value as app, m2.value as exp " +
			            "FROM " +
			            db.getSchemaPrefix()+"trial t " +
			            "left outer join " +
			            db.getSchemaPrefix()+"primary_metadata m1 on t.id = m1.trial and m1.name = 'Application' "+
			            "left outer join " +
			            db.getSchemaPrefix()+"primary_metadata m2 on t.id = m2.trial and m2.name = 'Experiment' " + whereClause ;
		            }
		            Vector<Trial> trials = new Vector<Trial>();

		            //System.out.println(buf.toString());
		            ResultSet resultSet = db.executeQuery(buf.toString());
		            while (resultSet.next() != false) {
		                Trial trial = new TAUdbTrial();
		                trial.setDatabase(db.getDatabase());
		                int pos = 1;
		                trial.setID(resultSet.getInt(pos++));
		                trial.setName(resultSet.getString(pos++));
		                trial.setField("data_source", resultSet.getString(pos++));
		                trial.setField("node_count", resultSet.getString(pos++));
		                trial.setField("contexts_per_node", resultSet.getString(pos++));
		                trial.setField("threads_per_context", resultSet.getString(pos++));
		                trial.setField("total_threads", resultSet.getString(pos++));
		                
		                String appname = resultSet.getString(pos++);
	//TODO: Figure out what to do about the app ids	                
//		                trial.setApplicationID(resultSet.getInt(pos++));

		                String expanme = resultSet.getString(pos++);
	//TODO: Figure out what to do about the experiment ids	                
//		                trial.setExperimentID();
		                


		                trials.addElement(trial);
		            }
		            resultSet.close();
//TODO: Deal with adding the metrics to the trial
		            // get the function details
		            Enumeration<Trial> en = trials.elements();
		            Trial trial;
		            while (en.hasMoreElements()) {
		                trial = en.nextElement();
		                trial.getTrialMetrics(db);
		            }

		            Collections.sort(trials);

		            return trials;

		        } catch (Exception ex) {
		            ex.printStackTrace();
	            return null;
		        }
	}
	    // gets the metric data for the trial
	    public void getTrialMetrics(DB db) {
	        // create a string to hit the database
	        StringBuffer buf = new StringBuffer();
	        buf.append("select id, name, derived ");
	        buf.append("from " + db.getSchemaPrefix() + "metric ");
	        buf.append("where trial = ");
	        buf.append(getID());
	        buf.append(" order by id ");
	        // System.out.println(buf.toString());

	        // get the results
	        try {
	            ResultSet resultSet = db.executeQuery(buf.toString());
	            while (resultSet.next() != false) {
	                Metric tmp = new Metric();
	                tmp.setID(resultSet.getInt(1));
	                tmp.setName(resultSet.getString(2));
	                tmp.setDerivedMetric(resultSet.getBoolean(3));
	                tmp.setTrialID(getID());
	                addMetric(tmp);
	            }
	            resultSet.close();
	        } catch (Exception ex) {
	            ex.printStackTrace();
	            return;
	        }
	        return;
	    }


//		    private static int getDBMetric(int trialID, int metric) {
//		        return 0;
//		    }

		    public static void deleteMetric(DB db, int trialID, int metricID) throws SQLException {
		        PreparedStatement statement = null;

		        // delete from the interval_location_profile table
		        statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "timer_value WHERE metric = ?");
		        statement.setInt(1, metricID);
		        statement.execute();

		        statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "metric WHERE id = ?");
		        statement.setInt(1, metricID);
		        statement.execute();
		    }
		 public static void deleteTrial(DB db, int trialID) throws SQLException {

		        PreparedStatement statement = null;
		        
//There's a chances that these might not work with MySQL, but after reading the manual 

		            // Postgresql, oracle, and DB2?
		            statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
		                    + "counter_value WHERE counter in (SELECT id FROM " + db.getSchemaPrefix()
		                    + "counter WHERE trial = ?)");
		            statement.setInt(1, trialID);
		            statement.execute();
		            
		            statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
		                    + "timer_value tv WHERE tv.timer_call_data IN (SELECT tcd.id FROM " + db.getSchemaPrefix()
		                    + "timer_call_data tcd WHERE tcd.timer_callpath IN (SELECT tcp.id FROM " + db.getSchemaPrefix()
		                    + "timer_callpath tcp WHERE tcp.timer IN (SELECT t.id FROM " + db.getSchemaPrefix()
		                    + "timer t WHERE trial = ?)))");
		            statement.setInt(1, trialID);
		            statement.execute();
		            
		            statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
		                    + "timer_call_data tcd WHERE tcd.timer_callpath IN (SELECT tcp.id FROM " + db.getSchemaPrefix()
		                    + "timer_callpath tcp WHERE tcp.timer IN (SELECT t.id FROM " + db.getSchemaPrefix()
		                    + "timer t WHERE trial = ?))");
		            statement.setInt(1, trialID);
		            statement.execute();
		            
		            statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
		                    + "timer_callpath WHERE timer IN (SELECT id FROM " + db.getSchemaPrefix()
		                    + "timer WHERE trial = ?)");
		            statement.setInt(1, trialID);
		            statement.execute();
		            statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
		                    + "timer_parameter WHERE timer IN (SELECT id FROM " + db.getSchemaPrefix()
		                    + "timer WHERE trial = ?)");
		            statement.setInt(1, trialID);
		            statement.execute();
		            
		            statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
		                    + "timer_group WHERE timer IN (SELECT id FROM " + db.getSchemaPrefix()
		                    + "timer WHERE trial = ?)");
		            statement.setInt(1, trialID);
		            statement.execute();
		            
		            statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix()
		                    + "timer_group WHERE timer IN (SELECT id FROM " + db.getSchemaPrefix()
		                    + "timer WHERE trial = ?)");
		            statement.setInt(1, trialID);
		            statement.execute();
		            
		        		     

		        statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "counter WHERE trial = ?");
		        statement.setInt(1, trialID);
		        statement.execute();

		        statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "primary_metadata WHERE trial = ?");
		        statement.setInt(1, trialID);
		        statement.execute();
		        
		        statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "secondary_metadata WHERE trial = ?");
		        statement.setInt(1, trialID);
		        statement.execute();
		        
		        statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "thread WHERE trial = ?");
		        statement.setInt(1, trialID);
		        statement.execute();
		        
		        statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "timer WHERE trial = ?");
		        statement.setInt(1, trialID);
		        statement.execute();
		        
		        statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "metric WHERE trial = ?");
		        statement.setInt(1, trialID);
		        statement.execute();

		        statement = db.prepareStatement(" DELETE FROM " + db.getSchemaPrefix() + "trial WHERE id = ?");
		        statement.setInt(1, trialID);
		        statement.execute();
		    }
		 
		public void loadXMLMetadata(DB db, Map<Integer, Function> ieMap) {
			loadMetadata(db, ieMap);
		}
		public void loadMetadata(DB db) {
			Map<Integer, Function> ieMap = new HashMap<Integer, Function>();
			loadMetadata(db, ieMap);
		}
//Shouldn't this method override loadXMLMetadata?
		public void loadMetadata(DB db, Map<Integer, Function> ieMap) {
			StringBuffer iHateThis = new StringBuffer();
			iHateThis.append("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>");
			iHateThis.append("<tau:metadata xmlns:tau=\"http://www.cs.uoregon.edu/research/tau\">");
			iHateThis.append("<tau:CommonProfileAttributes>");
	        try {
	            PreparedStatement statement = db.prepareStatement("SELECT name, value FROM " + db.getSchemaPrefix() + "primary_metadata WHERE trial = ?");
	            statement.setInt(1, this.trialID);
	            ResultSet results = statement.executeQuery();
	            while (results.next() != false) {
	                String name = results.getString(1);
	                String value = results.getString(2);
	                this.metaData.put(name, value);
	                iHateThis.append("<tau:attribute><tau:name>");
	                iHateThis.append(name);
	                iHateThis.append("</tau:name><tau:value>");
	                iHateThis.append(value);
	                iHateThis.append("</tau:value></tau:attribute>");
	            }
	            results.close();
				iHateThis.append("</tau:CommonProfileAttributes>");
	            // TODO: need to get the secondary metadata and either populate uncommonMetaData, or 
	            // something similar.
				
				int node = -1;
				int context = -1;
				int thread = -1;
				boolean inThread = false;
				
	            statement = db.prepareStatement("SELECT sm.name, sm.value, t.node_rank, t.context_rank, t.thread_rank, timer_callpath, iteration_start, time_start parent FROM " + 
	            		db.getSchemaPrefix() + "secondary_metadata sm left outer join " + db.getSchemaPrefix() + 
	            		"thread t on sm.thread = t.id left outer join " + db.getSchemaPrefix() + 
	            		"time_range tr on sm.time_range = tr.id WHERE sm.trial = ? order by t.node_rank, t.context_rank, t.thread_rank");
	            statement.setInt(1, this.trialID);
	            results = statement.executeQuery();
	            Thread currentThread = null;
	            while (results.next() != false) {
	            	if (node != results.getInt(3) || context != results.getInt(4) || thread != results.getInt(5)) {
	            		node = results.getInt(3);
	            		context = results.getInt(4);
	            		thread = results.getInt(5);
	            		if (this.getDataSource() != null) {
	            			currentThread = this.getDataSource().getThread(node, context, thread);
	            		}
	            		if (inThread) {
	            			iHateThis.append("</tau:ProfileAttributes>");
	            		}
            			iHateThis.append("<tau:ProfileAttributes context=\"" + context + "\" node=\"" + node + "\" thread=\"" + thread + "\">");
            			inThread = true;
	            	}
	                MetaDataKey key = this.uncommonMetaData.new MetaDataKey(results.getString(1));
	                Function f = ieMap.get(results.getInt(6));
	                if (f == null) {
	                	key.timer_context = "";
	                } else {
	                	key.timer_context = f.getName();
	                }
	                key.call_number = results.getInt(7);
	                key.timestamp = results.getLong(8);
	                String value = results.getString(2);
	                // put this value in the trial's uncommon metadata map
	                this.uncommonMetaData.put(key, value);
	                // put this value in the thread's metadata map
	                if (currentThread != null) {
		                currentThread.getMetaData().put(key, value);
	                }
	                iHateThis.append("<tau:attribute><tau:name>");
	                // for now, we need to build a super long string. Ugh.
	                String tmpName = key.timer_context + " : " + key.call_number + " : " + key.timestamp + " : " + key.name; 
	                iHateThis.append(tmpName);
	                // iHateThis.append(key.name);
	                iHateThis.append("</tau:name><tau:timer_context>");
	                iHateThis.append(key.timer_context);
	                iHateThis.append("</tau:timer_context><tau:call_number>");
	                iHateThis.append(key.call_number);
	                iHateThis.append("</tau:call_number><tau:timestamp>");
	                iHateThis.append(key.timestamp);
	                iHateThis.append("</tau:timestamp><tau:value>");
	                iHateThis.append(value);
	                iHateThis.append("</tau:value></tau:attribute>");
	            }
	            results.close();
	            if (inThread) {
	            	iHateThis.append("</tau:ProfileAttributes>");
	            }
				iHateThis.append("</tau:metadata>");
				this.setField(XML_METADATA, iHateThis.toString());
                this.setXmlMetaDataLoaded(true);
	        } catch (SQLException e) {
	            System.out.println("An error occurred loading metadata for trial object from TAUdb database.");
	            e.printStackTrace();
	        }
	        return;
	    }

}
