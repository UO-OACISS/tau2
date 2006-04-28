package common;

import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.List;

/**
 * This is the main RMI object which is used to send requests to the 
 * PerfExplorerServer object.  This interface defines the API for
 * passing requests to the server.
 * 
 * <P>CVS $Id: RMIPerfExplorer.java,v 1.2 2006/04/28 22:01:21 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public interface RMIPerfExplorer extends Remote {
	public String sayHello() throws RemoteException;
	public List getApplicationList() throws RemoteException; 
	public List getExperimentList(int applicationID) throws RemoteException; 
	public List getTrialList(int experimentID) throws RemoteException; 
	public String requestAnalysis(RMIPerfExplorerModel model, boolean force) throws RemoteException;
	public RMIPerformanceResults getPerformanceResults(RMIPerfExplorerModel model) throws RemoteException;
	public void stopServer() throws RemoteException;
	public RMIChartData requestChartData(RMIPerfExplorerModel model, int dataType) throws RemoteException;
	public List getPotentialGroups(RMIPerfExplorerModel model) throws RemoteException;
	public List getPotentialMetrics(RMIPerfExplorerModel model) throws RemoteException;
	public List getPotentialEvents(RMIPerfExplorerModel model) throws RemoteException;
	public String[] getMetaData(String tableName) throws RemoteException;
	public List getPossibleValues(String tableName, String columnName) throws RemoteException;
	public int createNewView(String name, int parent, String tableName, String columnName, String oper, String value) throws RemoteException;
	public List getViews(int parent) throws RemoteException;
	public List getTrialsForView(List views) throws RemoteException;
	public RMIPerformanceResults getCorrelationResults(RMIPerfExplorerModel model) throws RemoteException;
	public RMIVarianceData getVariationAnalysis(RMIPerfExplorerModel model) throws RemoteException;
	public RMICubeData getCubeData(RMIPerfExplorerModel model) throws RemoteException;
	public String getConnectionString() throws RemoteException;
}

