package edu.uoregon.tau.perfexplorer.common;

import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.List;


/**
 * This is the main RMI object which is used to send requests to the 
 * PerfExplorerServer object.  This interface defines the API for
 * passing requests to the server.
 * 
 * <P>CVS $Id: RMIPerfExplorer.java,v 1.16 2009/02/25 19:51:46 wspear Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public interface RMIPerfExplorer extends Remote {
    /**
     * RMI connection test method.
     *
     * @return
     * @throws RemoteException
     */
    public String sayHello() throws RemoteException;

    /**
     * Returns the full list of applications in the database.
     *
     * @return
     * @throws RemoteException
     */
    public List getApplicationList() throws RemoteException;

    /**
     * Returns the full list of experiments in the database for the
     * specified application.
     *
     * @param applicationID
     * @return
     * @throws RemoteException
     */
    public List getExperimentList(int applicationID) throws RemoteException;

    /**
     * Returns the full list of trials in the database for the specified
     * experiment.
     *
     * @param experimentID
     * @return
     * @throws RemoteException
     */
    public List getTrialList(int experimentID) throws RemoteException;

    /**
     * Requests analysis using the settings specified in the model.
     *
     * @param model
     * @param force If the request already exists, enter it again.
     * @return
     * @throws RemoteException
     */
    public String requestAnalysis(RMIPerfExplorerModel model, boolean force)
        throws RemoteException;

    /**
     * Requests the results of cluster or correlation analysis.
     *
     * @param model
     * @return
     * @throws RemoteException
     */
    public RMIPerformanceResults getPerformanceResults(
            RMIPerfExplorerModel model) throws RemoteException;

    /**
     * Method to stop execution of the server.
     *
     * @throws RemoteException
     */
    public void stopServer() throws RemoteException;

    /**
     * Requests data for generating comparison charts for the specified model.
     *
     * @param model
     * @param dataType
     * @return
     * @throws RemoteException
     */
    public RMIChartData requestChartData(RMIPerfExplorerModel model,
        ChartDataType dataType) throws RemoteException;

    /**
     * Requests data for generating comparison charts for the specified model.
     *
     * @param model
     * @param dataType
     * @return
     * @throws RemoteException
     */
    public RMIGeneralChartData requestGeneralChartData(RMIPerfExplorerModel model,
        ChartDataType dataType) throws RemoteException;

    /**
     * Method to request the common groups between the selected trials in
     * the specified model.
     *
     * @param model
     * @return
     * @throws RemoteException
     */
    public List getPotentialGroups(RMIPerfExplorerModel model)
        throws RemoteException;

    /**
     * Method to request the common metrics between the selected trials in
     * the specified model.
     *
     * @param model
     * @return
     * @throws RemoteException
     */
    public List<String> getPotentialMetrics(RMIPerfExplorerModel model)
        throws RemoteException;

    /**
     *
     * @param model
     * @return
     * @throws RemoteException
     */
    public List getPotentialEvents(RMIPerfExplorerModel model)
        throws RemoteException;

    /**
     * Requests the metadata for the specified table.
     *
     * @param tableName
     * @return
     * @throws RemoteException
     */
    public String[] getMetaData(String tableName) throws RemoteException;

    /**
     * Gets the possible values for the specified table, column.
     *
     * @param tableName
     * @param columnName
     * @return
     * @throws RemoteException
     */
    public List getPossibleValues(String tableName, String columnName)
        throws RemoteException;

    /**
     * Creates a new view in the view hierarchy.
     *
     * @param name
     * @param parent
     * @param tableName
     * @param columnName
     * @param oper
     * @param value
     * @return
     * @throws RemoteException
     */
    public int createNewView(String name, int parent, String tableName,
        String columnName, String oper, String value) throws RemoteException;

    /**
     * Gets the sub-views for the specified parent.
     *
     * @param parent If 0, get the top level views.
     * @return
     * @throws RemoteException
     */
    public List getViews(int parent) throws RemoteException;

    /**
     * Get the trials which are filtered by the specifed views.
     *
     * @param views
     * @return
     * @throws RemoteException
     */
    public List getTrialsForView(List views) throws RemoteException;

    /**
     * Gets the correlation results for the specified model.
     *
     * @param model
     * @return
     * @throws RemoteException
     */
    public RMIPerformanceResults getCorrelationResults(
        RMIPerfExplorerModel model) throws RemoteException;

    /**
     * Gets the variation data for the specified model.
     * @param model
     * @return
     * @throws RemoteException
     */
    public RMIVarianceData getVariationAnalysis(RMIPerfExplorerModel model)
    throws RemoteException;

    /**
     * Gets the top 4 events for the specified trial.
     *
     * @param model
     * @return
     * @throws RemoteException
     */
    public RMICubeData getCubeData(RMIPerfExplorerModel model)
        throws RemoteException;

    /**
     * Gets the URL for the JDBC connection.
     *
     * @return The connection URL
     * @throws RemoteException
     */
    public String getConnectionString() throws RemoteException;

    /**
     * Gets the URLs for the JDBC connections.
     *
     * @return The connection URLs
     * @throws RemoteException
     */
    public List getConnectionStrings() throws RemoteException;

    /**
     * Gets the configuration names for the JDBC connections.
     *
     * @return The configuration names
     * @throws RemoteException
     */
    public List getConfigNames() throws RemoteException;

    /**
     * Gets the list of events for the specified trial, metric
     * 
     * @param trialID
     * @param metricIndex
     * @return a list of event names
     * @throws RemoteException
     */
	public List getEventList(int trialID, int metricIndex) throws RemoteException; 

    /**
     * Returns the full list of trials in the database for the specified
     * where clause.
     *
     * @param criteria
     * @return
     * @throws RemoteException
     */
    public List getTrialList(String criteria) throws RemoteException;

    /**
     * Deletes a view from the view hierarchy.
     *
     * @param id
     * @return
     * @throws RemoteException
     */
    public void deleteView(String id) throws RemoteException;

    /**
     * Returns the full list of possible table.column selections for graphs.
     *
     * @return a list of table.column names
     * @throws RemoteException
     */
    public List getChartFieldNames() throws RemoteException;

    /**
     * Requests data for generating comparison charts for the specified model.
     *
     * @param model
     * @return list of XML fields
     * @throws RemoteException
     */
    public List getXMLFields(RMIPerfExplorerModel model) throws RemoteException;

    /**
     * Because PerfExplorer now supports multiple database connections, specify
     * which connection index should be used for subsequent commands.
     *
     * @param connectionIndex
     * @return 
     * @throws RemoteException
     */
    public void setConnectionIndex(int connectionIndex) throws RemoteException;
	
    /**
     * Because PerfExplorer now supports multiple database connections, allow
     * for dropping all connections and restarting
     *
     * @return 
     * @throws RemoteException
     */
    public void resetServer() throws RemoteException;

    /**
    *
    * @param model
    * @return
    * @throws RemoteException
    */
   public List<String> getPotentialAtomicEvents(RMIPerfExplorerModel model)
       throws RemoteException;
}

