/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import javax.swing.JFrame;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Database;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.DatabaseException;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.View;
import edu.uoregon.tau.perfdmf.View.ViewRule;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerClient;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerModel;
import edu.uoregon.tau.perfexplorer.common.RMISortableIntervalEvent;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;




/**
 * @author khuck
 *
 */
public class Utilities {
	public static Trial getTrialFromView (View v, String tName) {
		boolean message = false;
    PerfExplorerServer server = getServer();
    List<View> views = new ArrayList();
    views.add(v);
  	List<Trial> trials = server.getTrialsForView(views, false);//TODO: Is this metadata ever needed?
    for (Trial trial : trials) {
      System.out.println(trial);
      if (trial.getName().trim().equals(tName.trim())) {
        if (!trial.isXmlMetaDataLoaded()) {
          try {
      trial.loadXMLMetadata(server.getDB());
    } catch (SQLException e) {
      System.err.println("Error getting metadata for trial");
      e.printStackTrace();
    }
        }
        return trial;
      }
    }
		message = true;
    return null;
  }

  public static List<Trial> getTrialByName (String tName) {
	boolean message = false;
    PerfExplorerServer server = getServer();
    // shortcut - assume this is a TAUdb database.
	String whereClause = "";
	int i = 0;
	whereClause += " where t.name = '" + tName + "'";
    List<Trial> trials = Trial.getTrialList(server.getDB(), whereClause, false); 
    message = true;
    return trials;
  }

  public static List<Trial> getTrialsFromMetadata (Map<String,String> metadata, String conjoin) {
      boolean message = false;
      PerfExplorerServer server = getServer();
	  // shortcut - assume this is a TAUdb database.
	  String whereClause = "";
	  int i = 0;
	  for (String key : metadata.keySet()) {
	    whereClause += " inner join primary_metadata pm" + i + " on pm" + i;
		whereClause += ".trial = t.id and pm" + i;
		whereClause += ".name = '" + key + "' ";
		i++;
	  }
	  whereClause += " where ";
	  i = 0;
	  for (String key : metadata.keySet()) {
	    if (i > 0) {
		  whereClause += conjoin;
		}
		whereClause += "pm" + i + ".value " + metadata.get(key) + " ";
		i++;
	  }
      List<Trial> trials = Trial.getTrialList(server.getDB(), whereClause, false); 
      message = true;
      return trials;
  }

  public static Trial getTrial (String aName, String eName, String tName) {
		boolean message = false;
        PerfExplorerServer server = getServer();
		List<Application> apps = server.getApplicationList();
        for (Application app : apps ) {
            if (app.getName().equals(aName)) {
            	//System.out.println("Found app");
            	List<Experiment> exps = server.getExperimentList(app.getID());
            	for (Experiment exp : exps) {
            		if (exp.getName().trim().equals(eName.trim())) {
                    	//System.out.println("Found exp");
            			List<Trial> trials = server.getTrialList(exp.getID(), false);//TODO: Is this metadata ever needed?
            			for (Trial trial : trials) {
            				if (trial.getName().trim().equals(tName.trim())) {
            					if (!trial.isXmlMetaDataLoaded()) {
            						try {
										trial.loadXMLMetadata(server.getDB());
									} catch (SQLException e) {
										System.err.println("Error getting metadata for trial");
										e.printStackTrace();
									}
            					}
            					return trial;
            				}
            			}
						System.out.println("Could not find trial: " + tName);
						message = true;
            		}
            	}
				if (!message)
					System.out.println("Could not find experiment: " + eName);
				message = true;
			}
        }
		if (!message)
		System.out.println("Could not find application: " + aName);
        return null;
	}

	public static List<View> getViews () {
		boolean message = false;

    PerfExplorerServer server = getServer();
    return server.getViews(0); 
  }

	public static List<View> getSubViews (View view) {
		boolean message = false;

    PerfExplorerServer server = getServer();
    return server.getViews(view.getID()); 
  }

	public static List<View> getAllViews () {
		boolean message = false;

    PerfExplorerServer server = getServer();
    return server.getAllSubViews(0); 
  }

	
	public static void createView(String name, boolean requireAll, List<ViewRule> rules) {
			createSubView(name, requireAll, -1, rules);
		}
	
	public static void createSubView(String name, boolean requireAll, int parent, List<ViewRule> rules) {
			
			PerfExplorerServer server = getServer();
			DB db = server.getDB();
			try {
				View.createView(db, name, requireAll, parent, rules);
			} catch (SQLException e) {
				e.printStackTrace();
			}
		}
	
	
	public static View getView (String name) {
		boolean message = false;

    PerfExplorerServer server = getServer();
    List<View> views = server.getAllSubViews(0); 
    View found = new View();
    for (View view : views ) {
        if (view.toString().equals(name)) {
          found = view;
          break;
        }
    }
    return found;
  }
	
  public static View getSubView (View view, String name) {
		boolean message = false;

    PerfExplorerServer server = getServer();
    List<View> views = server.getAllSubViews(view.getID()); 
    View found = new View();
    for (View subView : views ) {
        if (subView.toString().equals(name)) {
          found = subView;
          break;
        }
    }
    return found;
  }

  public static List<Trial> getTrialsForView(View view)
  {
		return getTrialsForView(view, false);
	}

	public static List<Trial> getTrialsForView(View view, boolean getXMLMetadata) {
		boolean message = false;

    PerfExplorerServer server = getServer();

    List<View> views = new ArrayList<View>();
		View parent = view.getParent();
		while (parent != null) {
			views.add(parent);
			parent = parent.getParent();
		}
    views.add(view);

		return server.getTrialsForTAUdbView(views, getXMLMetadata);
  }

	public static List<Trial> getTrialsForExperiment (String aName, String eName) {
		boolean message = false;
        PerfExplorerServer server = getServer();
		List<Application> apps = server.getApplicationList();
        for (Application app : apps ) {
            if (app.getName().equals(aName)) {
            	//System.out.println("Found app");
            	List<Experiment> exps = server.getExperimentList(app.getID());
            	for (Experiment exp : exps) {
            		if (exp.getName().trim().equals(eName.trim())) {
                    	//System.out.println("Found exp");
            			List<Trial> trials = server.getTrialList(exp.getID(),false);//TODO: Is this metadata ever needed?
            			for (Trial trial : trials) {
            				if (!trial.isXmlMetaDataLoaded()) {
            					try {
									trial.loadXMLMetadata(server.getDB());
								} catch (SQLException e) {
									System.err.println("Error getting metadata for trial");
									e.printStackTrace();
								}
            				}
            			}
						return trials;
            		}
            	}
				if (!message)
					System.out.println("Could not find experiment: " + eName);
				message = true;
			}
        }
		if (!message)
		System.out.println("Could not find application: " + aName);
        return null;
	}

	public static void  deleteTrial (Trial t) {
		boolean message = false;
    PerfExplorerServer server = getServer();
		try {
		Trial.deleteTrial(server.getDB(), t.getID());
		} catch (SQLException e) {}
	}
	
	public static void saveMetric(Trial t, Metric metric){
		boolean message = false;
		if (metric != null) {
			Database database = getServer().getDB().getDatabase();
			
			
			DatabaseAPI databaseAPI = new DatabaseAPI();
			try {
				databaseAPI.initialize(database);
			} catch (SQLException e) {
				e.printStackTrace();
			}

			if (databaseAPI != null) {
				databaseAPI.saveTrial(t,metric);
				databaseAPI.terminate();
			}
		}
	}

	public static List<Application> getApplications () {
		boolean message = false;
    PerfExplorerServer server = getServer();
		return server.getApplicationList();
	}
	
	public static List<Experiment> getExperimentsForApplication (String aName) {
		boolean message = true;
    PerfExplorerServer server = getServer();
		List<Application> apps = server.getApplicationList();
        for (Application app : apps ) {
            if (app.getName().equals(aName)) {
            	List<Experiment> exps = server.getExperimentList(app.getID());
				return exps;
			}
        }
		if (!message)
		System.out.println("Could not find application: " + aName);
        return null;
	}

	public static Trial getCurrentTrial () {
		Trial trial = PerfExplorerModel.getModel().getTrial();
		// should we check for a valid trial?
		if (trial == null) {
			System.out.println("Utilities.getCurrentTrial() failed: No trial selected.");
		}
		return trial;
	}

	private static PerfExplorerServer getServer() {
		return PerfExplorerServer.getServer(null, "", "");
	}

	private static PerfExplorerServer getServer(String configName, String tauHome) {
		String home = System.getProperty("user.home");
		String slash = System.getProperty("file.separator");
		String configFile = home + slash + ".ParaProf" + slash + "perfdmf.cfg";
		if(configName!=null&&configName.length()>0){
			configFile=configFile+"." + configName;
		}
		return PerfExplorerServer.getServer(configFile, tauHome, "");
	}

	public static JFrame getClient() {
		String home = System.getProperty("user.home");
		String slash = System.getProperty("file.separator");
		String configFile = home + slash + ".ParaProf" + slash + "perfdmf.cfg";
		JFrame frame = new PerfExplorerClient(true, configFile, true);
		frame.pack();
		frame.setVisible(true);
		return frame;
	}

	private static boolean isSet=false;
	public static int setSession (String name) {
		try {
			
			if(isSet){
				PerfExplorerServer.hardResetServer();
			}
			PerfExplorerServer server = getServer(name, "");
			isSet=true;
			List<String> configNames = server.getConfigNames();
			for (int i = 0 ; i < server.getSessionCount() ; i++) {
				server.setConnectionIndex(i);
//				System.out.println(server.getConnectionString());
				String tmpname = (String)configNames.get(i);
				if (tmpname.equals(name)||((name==null||name.length()==0)&&tmpname.endsWith("/perfdmf.cfg"))) {
					// getting the schema version forces the connection to be made
					server.getSchemaVersion(i);
//					System.out.println("selected: " + server.getConnectionString());
					return i;
				}
			}
		} catch (Exception e) {}
		return 0;
	}
	
	public static LinkedHashMap<String, Double> sortHashMapByValues(Map<String, Double> passedMap, boolean ascending) {
		List<String> mapKeys = new ArrayList<String>(passedMap.keySet());
		List<Double> mapValues = new ArrayList<Double>(passedMap.values());
		Collections.sort(mapValues);
		Collections.sort(mapKeys);

		if (!ascending) {
			Collections.reverse(mapValues);
		}

		LinkedHashMap<String, Double> someMap = new LinkedHashMap<String, Double>();
		for (Double val : mapValues) {
			for (String key : mapKeys) {
				if (passedMap.get(key).toString().equals(val.toString())) {
					passedMap.remove(key);
					mapKeys.remove(key);
					someMap.put(key, val);
					break;
				}
			}
		}
		return someMap;
	} 
	
	public static List<RMISortableIntervalEvent> getEventsForTrial(Trial trial, int metricIndex) {
		List<RMISortableIntervalEvent> events = null;
        PerfExplorerServer server = getServer();
        events = server.getEventList(trial.getID(), metricIndex);
		return events;
	}

	/**
	 * Shorten a function name
	 * @param longName the long function name
	 * @return the shortened function name
	 */
	public static String shortenEventName(String longName) {
		StringTokenizer st = new StringTokenizer(longName, "(");
		String shorter = null;
		// trim the function arguments
		try {
			shorter = st.nextToken();
			if (shorter.length() < longName.length()) {
				shorter = shorter + "()";
			}
		} catch (NoSuchElementException e) {
			shorter = longName;
		}
		longName = shorter;
		// trim the source location
		int index = longName.indexOf(" [{");
		if (index >= 0) {
		    int last = longName.lastIndexOf("/");
			if (last >= 0) {
			    shorter = longName.substring(0,index+3) + longName.substring(last+1,longName.length());
			} else {
			    shorter = longName.substring(0,index);
			}
		} else {
			shorter = longName;
		}
		// remove any OPENMP annotation
		shorter = shorter.replace("[OPENMP] ", "");
		shorter = shorter.replace("OpenMP_", "");
		return shorter.trim();
	}


}
