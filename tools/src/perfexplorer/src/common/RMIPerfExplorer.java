package edu.uoregon.tau.perfexplorer.common;

import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.List;
import java.util.Map;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Trial;

/**
 * This is the main RMI object which is used to send requests to the 
 * PerfExplorerServer object.  This interface defines the API for
 * passing requests to the server.
 * 
 * <P>CVS $Id: RMIPerfExplorer.java,v 1.22 2009/04/05 23:59:27 khuck Exp $</P>
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
    public List<Application> getApplicationList() throws RemoteException;

    /**
     * Returns the full list of experiments in the database for the
     * specified application.
     *
     * @param applicationID
     * @return
     * @throws RemoteException
     */
    public List<Experiment> getExperimentList(int applicationID) throws RemoteException;

    /**
     * Returns the full list of trials in the database for the specified
     * experiment.
     *
     * @param experimentID
     * @return
     * @throws RemoteException
     */
    public List<Trial> getTrialList(int experimentID, boolean getXMLMetadata) throws RemoteException;

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
    public List<String> getPotentialGroups(RMIPerfExplorerModel model)
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
    public List<String> getPotentialEvents(RMIPerfExplorerModel model)
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
    public List<String> getPossibleValues(String tableName, String columnName)
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
    public List<RMIView> getViews(int parent) throws RemoteException;

    /**
     * Get the trials which are filtered by the specifed views.
     *
     * @param views
     * @return
     * @throws RemoteException
     */
    public List<Trial> getTrialsForView(List<RMIView> views, boolean getXMLMetadata) throws RemoteException;

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
    public List<String> getConnectionStrings() throws RemoteException;

    /**
     * Gets the configuration names for the JDBC connections.
     *
     * @return The configuration names
     * @throws RemoteException
     */
    public List<String> getConfigNames() throws RemoteException;

    /**
     * Gets the list of events for the specified trial, metric
     * 
     * @param trialID
     * @param metricIndex
     * @return a list of event names
     * @throws RemoteException
     */
	public List<RMISortableIntervalEvent> getEventList(int trialID, int metricIndex) throws RemoteException; 

    /**
     * Returns the full list of trials in the database for the specified
     * where clause.
     *
     * @param criteria
     * @return
     * @throws RemoteException
     */
    public List<Trial> getTrialList(String criteria, boolean getXMLMetadata) throws RemoteException;

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
    public List<String> getChartFieldNames() throws RemoteException;

    /**
     * Requests data for generating comparison charts for the specified model.
     *
     * @param model
     * @return list of XML fields
     * @throws RemoteException
     */
    public List<String> getXMLFields(RMIPerfExplorerModel model) throws RemoteException;

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

   /**
   * Get the counts for each thread count value for all the selected trials
   * 
   * @param model
   * @return
   * @throws RemoteException
   */
   public Map<String, Integer> checkScalabilityChartData(RMIPerfExplorerModel model)
   	throws RemoteException;

   public Map<String, double[][]> getUserEventData(RMIPerfExplorerModel model)
   	throws RemoteException;
}

