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

import javax.swing.JFrame;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerClient;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerModel;
import edu.uoregon.tau.perfexplorer.common.RMISortableIntervalEvent;
import edu.uoregon.tau.perfexplorer.server.PerfExplorerServer;




/**
 * @author khuck
 *
 */
public class Utilities {
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

	public static List<Experiment> getExperimentsForApplication (String aName) {
		boolean message = false;
        PerfExplorerServer server = getServer();
		List<Application> apps = server.getApplicationList();
        for (Application app : apps ) {
            if (app.getName().equals(aName)) {
            	//System.out.println("Found app");
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
		return PerfExplorerServer.getServer(null);
	}

	private static PerfExplorerServer getServer(String configName) {
		String home = System.getProperty("user.home");
		String slash = System.getProperty("file.separator");
		String configFile = home + slash + ".ParaProf" + slash + "perfdmf.cfg." + configName;
		return PerfExplorerServer.getServer(configFile);
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

	public static int setSession (String name) {
		try {
			PerfExplorerServer server = getServer(name);
			List<String> configNames = server.getConfigNames();
			for (int i = 0 ; i < server.getSessionCount() ; i++) {
				server.setConnectionIndex(i);
//				System.out.println(server.getConnectionString());
				String tmpname = (String)configNames.get(i);
				if (tmpname.equals(name)) {
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

}
