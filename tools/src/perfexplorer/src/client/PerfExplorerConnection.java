package client;

import java.rmi.*;
import common.*;
import server.*;
import java.util.*;
import javax.swing.*;

public class PerfExplorerConnection {

    private static PerfExplorerConnection theConnection = null;

    private RMIPerfExplorer server = null;
    private static boolean standalone = false;
    private static String configFile = null;
    private static int analysisEngine = 0;

    private PerfExplorerConnection () {
		makeConnection();
    }

	private void makeConnection() {
		if (standalone) {
	    	server = PerfExplorerServer.getServer(configFile, analysisEngine);
		} else {
	    	if (System.getSecurityManager() == null) {
				System.setSecurityManager(new RMISecurityManager());
	    	}
	    	try {
				String hostname = System.getProperty("java.rmi.server.hostname");
				String name = "PerfExplorerServer";
				System.out.println("Connecting to rmi://" + hostname + "/" + name);
				//server = (RMIPerfExplorer)Naming.lookup("//utonium.cs.uoregon.edu/" + name);
				server = (RMIPerfExplorer)Naming.lookup("//" + hostname + "/" + name);
				System.out.println("Bound to " + name);
	    	} catch (Exception e) {
				System.err.println("createServer Exception: " + e.getMessage());
				e.printStackTrace();
				server = null;
	    	}
		}
	}

    public static void setStandalone (boolean standalone) {
	PerfExplorerConnection.standalone = standalone;
    }

    public static void setConfigFile (String configFile) {
	PerfExplorerConnection.configFile = configFile;
    }
	
    public static void setAnalysisEngine(int analysisEngine) {
	PerfExplorerConnection.analysisEngine = analysisEngine;
    }

    public static PerfExplorerConnection getConnection() {
	if (theConnection == null) {
	    theConnection = new PerfExplorerConnection();
	}
	if (theConnection.server == null) {
	    return null;
	}
	return theConnection;
    }

    private void handleError (RemoteException e, String functionName) {
		System.out.println("PerfExplorerConnection." + functionName + " Exception: ");
		System.out.println(e.getMessage());
		e.printStackTrace();
		JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), 
			"An error occurred communicating with the server.", 
			"Server Error", JOptionPane.ERROR_MESSAGE);
		// try to reconnect
		server = null;
		makeConnection();
		if (server == null) {
			JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), 
			"The connection could not be restored.\n" +
			"Please check to make sure the server is running.", 
			"Server Error", JOptionPane.ERROR_MESSAGE);
		} else {
			JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(),
			"Connection restored - please try your request again.",
			"Connection Restored", JOptionPane.INFORMATION_MESSAGE);
		}
    }

    public String sayHello() {
	String tmpStr = null;
	try {
	    tmpStr = server.sayHello();
	} catch (RemoteException e) {
	    handleError(e, "sayHello()");
	}
	return tmpStr;
    }

    public ListIterator getApplicationList() {
	ListIterator tmpIterator = null;
	try {
	    tmpIterator = server.getApplicationList().listIterator();
	} catch (RemoteException e) {
	    handleError(e, "getApplicationList()");
	}
	return tmpIterator;
    }

    public ListIterator getExperimentList(int applicationID) {
	ListIterator tmpIterator = null;
	try {
	    tmpIterator =
		server.getExperimentList(applicationID).listIterator();
	} catch (RemoteException e) {
	    handleError(e, "getExperimentList(" + applicationID + ")");
	}
	return tmpIterator;
    }

    public ListIterator getTrialList(int experimentID) {
	ListIterator tmpIterator = null;
	try {
	    tmpIterator =
		server.getTrialList(experimentID).listIterator();
	} catch (RemoteException e) {
	    handleError(e, "getTrialList(" + experimentID + ")");
	}
	return tmpIterator;
    }

    public String requestAnalysis(RMIPerfExplorerModel model, boolean force) {
	String tmpString = null;
	try {
	    tmpString = server.requestAnalysis(model, force);
	} catch (RemoteException e) {
	    handleError(e, "requestAnalysis(" + model.toString() + ")");
	}
	return tmpString;
    }

    public RMIPerformanceResults getPerformanceResults(PerfExplorerModel model) {
	RMIPerformanceResults results = null;
	try {
	    results = server.getPerformanceResults(model);
	} catch (RemoteException e) {
	    handleError(e, "getPerformanceResults(" + model.toString() + ")");
	}
	return results;
    }

    public RMIPerformanceResults getCorrelationResults(PerfExplorerModel model) {
	RMIPerformanceResults results = null;
	try {
	    results = server.getCorrelationResults(model);
	} catch (RemoteException e) {
	    handleError(e, "getCorrelationResults(" + model.toString() + ")");
	}
	return results;
    }

    public void stopServer() {
	try {
	    server.stopServer();
	} catch (RemoteException e) {
	    handleError(e, "stopServer()");
	}
    }

    public RMIChartData requestChartData(PerfExplorerModel model, int dataType) {
	RMIChartData data = null;
	try {
	    data = server.requestChartData(model, dataType);
	} catch (RemoteException e) {
	    handleError(e, "requestChartData(" + model.toString() + ")");
	}
	return data;
    }

    public List getPotentialGroups(PerfExplorerModel model) {
	List groups = null;
	try {
	    groups = server.getPotentialGroups(model);
	} catch (RemoteException e) {
	    handleError(e, "getPotentialGroups(" + model.toString() + ")");
	}
	return groups;
    }

    public List getPotentialMetrics(PerfExplorerModel model) {
	List metrics = null;
	try {
	    metrics = server.getPotentialMetrics(model);
	} catch (RemoteException e) {
	    handleError(e, "getPotentialGroups(" + model.toString() + ")");
	}
	return metrics;
    }

    public List getPotentialEvents(PerfExplorerModel model) {
	List events = null;
	try {
	    events = server.getPotentialEvents(model);
	} catch (RemoteException e) {
	    handleError(e, "getPotentialEvents(" + model.toString() + ")");
	}
	return events;
    }

    public String[] getMetaData(String tableName) {
	String[] columns = null;
	try {
	    columns = server.getMetaData(tableName);
	} catch (RemoteException e) {
	    handleError(e, "getMetaData(" + tableName + ")");
	}
	return columns;
    }

    public List getPossibleValues(String tableName, String columnName) {
	List values = null;
	try {
	    values = server.getPossibleValues(tableName, columnName);
	} catch (RemoteException e) {
	    handleError(e, "getPossibleValues(" + tableName + ", " + columnName + ")");
	}
	return values;
    }

    public int createNewView (String name, int parent, String tableName, String columnName, String oper, String value) {
	int viewID = 0;
	try {
	    viewID = server.createNewView(name, parent, tableName, columnName, oper, value);
	} catch (RemoteException e) {
	    handleError(e, "createNewView(" + tableName + ", " + columnName + ", " + value + ")");
	}
	return viewID;
    }

    public List getViews(int parent) {
	List views = null;
	try {
	    views = server.getViews(parent);
	} catch (RemoteException e) {
	    handleError(e, "getViews(" + parent + ")");
	}
	return views;
    }

    public ListIterator getTrialsForView(List views) {
	ListIterator trials = null;
	try {
	    trials = server.getTrialsForView(views).listIterator();
	} catch (RemoteException e) {
	    handleError(e, "getTrialsForView(" + views + ")");
	}
	return trials;
    }

    public RMIVarianceData requestVariationAnalysis(PerfExplorerModel model) {
	RMIVarianceData results = null;
	try {
	    results = server.getVariationAnalysis(model);
	} catch (RemoteException e) {
	    handleError(e, "getVariationAnalysis(" + model.toString() + ")");
	}
	return results;
    }

    public RMICubeData requestCubeData(PerfExplorerModel model) {
	RMICubeData results = null;
	try {
	    results = server.getCubeData(model);
	} catch (RemoteException e) {
	    handleError(e, "getCubeData(" + model.toString() + ")");
	}
	return results;
    }


}
