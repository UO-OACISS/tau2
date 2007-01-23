

/**
 * 
 */
package server;

import clustering.AnalysisFactory;
import clustering.ClusterException;
import clustering.DataNormalizer;
import clustering.KMeansClusterInterface;
import clustering.PrincipalComponentsAnalysisInterface;
import clustering.RawDataInterface;

import common.EngineType;
import common.AnalysisType;
import common.ChartType;
import common.TransformationType;
import common.PerfExplorerException;
import common.PerfExplorerOutput;
import common.RMIPerfExplorerModel;

import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.database.DB;

import java.awt.Color;

import java.awt.Shape;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.TimerTask;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.DefaultXYItemRenderer;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.Range;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.function.Function2D;
import org.jfree.data.function.LineFunction2D;
import org.jfree.data.function.PowerFunction2D;
import org.jfree.data.general.DatasetUtilities;
import org.jfree.data.statistics.Regression;
import org.jfree.data.xy.XYDataset;


/**
 * This is a rewrite of the AnalysisTask class.
 * This class is intended to be a wrapper around data mining operations
 * available in Weka, R and Octave.  The orignal AnalysisTask class
 * only supported R directly.  This is intended to be an improvement...
 *
 * <P>CVS $Id: AnalysisTask.java,v 1.4 2007/01/23 18:46:29 khuck Exp $</P>
 * @author Kevin Huck
 * @version 0.1
 * @since 0.1
 */
public class AnalysisTask extends TimerTask {
    
    private ChartType chartType = ChartType.DENDROGRAM;
    private AnalysisFactory factory = null;
    
    private RMIPerfExplorerModel modelData = null;
    private PerfExplorerServer server = null;
    private DatabaseAPI session = null;
    private int analysisID = 0;
    private int numRows = 0;
    private int numTotalThreads = 0;
    private int numEvents = 0;
    private int nodes = 0;
    private int contexts = 0;
    private int threads = 0;
    private RawDataInterface rawData = null;
    private double maximum = 0.0;
    private List eventIDs = null;
    private double rCorrelation = 0.0;
    private boolean correlateToMain = false;

    /**
    * Constructor.  The engine parameter passed in specifies which analysis
    * engine to use. 
    * 
    * @param server
    * @param engine
    */
    public AnalysisTask (PerfExplorerServer server, DatabaseAPI session) {
        super();
        this.server = server;
        this.session = session;
        this.factory = server.getAnalysisFactory();
    }

    /**
     * This method saves the analysis results in the database.
     * 
     * @param description
     * @param fileName
     * @param thumbnail
     * @param big
     * @throws PerfExplorerException
     */
    public void saveAnalysisResult (String description, String fileName, 
        String thumbnail, boolean big) throws PerfExplorerException {
        // create the thumbnail
        Thumbnail.createThumbnail(fileName, thumbnail, big);
        // save the image in the database
        try {
            DB db = session.db();
            //db.setAutoCommit(false);
            PreparedStatement statement = null;
            StringBuffer buf = new StringBuffer();
            buf.append("insert into analysis_result (analysis_settings, ");
            buf.append("description, thumbnail_size, image_size, thumbnail, ");
            buf.append("image, result_type) values ");
            buf.append("(?, ?, ?, ?, ?, ?, ?) ");
            statement = db.prepareStatement(buf.toString());
            statement.setInt(1, analysisID);
            statement.setString(2, fileName);
            File v_file = new File(fileName);
            FileInputStream v_fis = new FileInputStream(v_file);
            File v_thumb = new File(thumbnail);
            FileInputStream v_tis = new FileInputStream(v_thumb);
            statement.setInt(3,(int)v_thumb.length());
            statement.setInt(4,(int)v_file.length());
            statement.setBinaryStream(5,v_tis,(int)v_thumb.length());
            statement.setBinaryStream(6,v_fis,(int)v_file.length());
            statement.setString(7, chartType.toString());
            statement.executeUpdate();
            //db.commit();
            v_fis.close();
            statement.close();
            v_file.delete();
            v_thumb.delete();
        } catch (Exception e) {
            String error = "ERROR: Couldn't insert the analysis results into the database!";
            System.err.println(error);
            System.err.println(e.getMessage());
            e.printStackTrace();
            throw new PerfExplorerException(error, e);
        }
    }

	/**
	 * This method saves the analysis results in the database.
	 * 
	 * @param centroids
	 * @param deviations
	 * @param thumbnail
	 * @param outfile
	 * @throws PerfExplorerException
	 */
	public void saveAnalysisResult (RawDataInterface centroids, RawDataInterface deviations, File thumbnail, File outfile) throws PerfExplorerException {
		// save the image in the database
		try {
			DB db = session.db();
			db.setAutoCommit(false);
			PreparedStatement statement = null;
			// for each centroid, save the data
			// TODO - MAKE THIS A GENERAL USE LATER!
			StringBuffer buf = new StringBuffer();
			buf.append("insert into analysis_result ");
			buf.append(" (analysis_settings, description, thumbnail_size, thumbnail, image_size, image, result_type) values (?, ?, ?, ?, ?, ?, ?)");
			statement = db.prepareStatement(buf.toString());
			statement.setInt(1, analysisID);
			statement.setString(2, new String("analysis_result"));
       		FileInputStream inStream = new FileInputStream(thumbnail);
       		statement.setInt(3,(int)outfile.length());
       		statement.setBinaryStream(4,inStream,(int)thumbnail.length());
       		FileInputStream inStream2 = new FileInputStream(outfile);
       		statement.setInt(5,(int)outfile.length());
       		statement.setBinaryStream(6,inStream2,(int)outfile.length());
			statement.setString(7, chartType.toString());
			//PerfExplorerOutput.println(statement.toString());
       		statement.executeUpdate();
       		statement.close();
       		db.commit();
		} catch (Exception e) {
			String error = "ERROR: Couldn't insert the analysis results into the database!";
			System.err.println(error);
            System.err.println(e.getMessage());
			e.printStackTrace();
			throw new PerfExplorerException(error, e);
		}
	}

	/**
	 * This method gets the constant values for the data.  In order to retrieve
	 * the data correctly, it is necessary to get the number of events, metrics, 
	 * threads, and the total number of rows of data we are expecting.
	 * 
	 * @throws PerfExplorerException
	 */
	private void getConstants () throws PerfExplorerException {
		PerfExplorerOutput.print("Getting constants...");
		try {
			DB db = session.db();
			PreparedStatement statement = null;
			// First, get the total number of rows we are expecting
			StringBuffer sql = new StringBuffer();

            if (db.getDBType().compareTo("oracle") == 0) {
                sql.append("select count(p.excl) ");
            } else {
                sql.append("select count(p.exclusive) ");
            }

			sql.append("from interval_event e ");
			sql.append("left outer join interval_location_profile p ");
			sql.append("on e.id = p.interval_event ");
			if (modelData.getDimensionReduction().equals(TransformationType.OVER_X_PERCENT)) {
				sql.append("inner join interval_mean_summary s ");
				sql.append("on e.id = s.interval_event and s.metric = p.metric ");
				sql.append("and s.exclusive_percentage > ");
				sql.append("" + modelData.getXPercent() + "");	
			//} else if (modelData.getCurrentSelection() instanceof Metric) {
				//sql.append("inner join interval_mean_summary s ");
				//sql.append("on e.id = s.interval_event and s.metric = p.metric ");
			}
			sql.append("where e.trial = ?");
			sql.append(" and e.group_name not like '%TAU_CALLPATH%' ");
			if (modelData.getCurrentSelection() instanceof Metric) {
				sql.append(" and p.metric = ?");
			}
			statement = db.prepareStatement(sql.toString());
			statement.setInt(1, modelData.getTrial().getID());
			if (modelData.getCurrentSelection() instanceof Metric) {
				statement.setInt(2, ((Metric)(modelData.getCurrentSelection())).getID());
			}
			// PerfExplorerOutput.println(statement.toString());
			ResultSet results = statement.executeQuery();
			if (results.next() != false) {
				numRows = results.getInt(1);
			}
			results.close();
			statement.close();

			if (modelData.getCurrentSelection() instanceof Metric) {
				// Next, get the event names, and count them
				sql = new StringBuffer();
				sql.append("select e.id, e.name from interval_event e ");
				if (modelData.getDimensionReduction().equals(TransformationType.OVER_X_PERCENT)) {
					sql.append("inner join interval_mean_summary s on ");
					sql.append("e.id = s.interval_event ");
					sql.append("and s.exclusive_percentage > ");
					sql.append("" + modelData.getXPercent() + "");
					sql.append(" where e.trial = ? ");
					if (modelData.getCurrentSelection() instanceof Metric) {
						sql.append(" and s.metric = ? ");
					}
				} else {
					sql.append("where e.trial = ?");
				}
			sql.append(" and e.group_name not like '%TAU_CALLPATH%' ");
				sql.append(" order by 1");
				statement = db.prepareStatement(sql.toString());
				statement.setInt(1, modelData.getTrial().getID());
				if (modelData.getDimensionReduction().equals(TransformationType.OVER_X_PERCENT)) {
					if (modelData.getCurrentSelection() instanceof Metric) {
						statement.setInt(2, ((Metric)(modelData.getCurrentSelection())).getID());
					}
				}
				//PerfExplorerOutput.println(statement.toString());
				results = statement.executeQuery();
				numEvents = 0;
				eventIDs = new ArrayList();
				while (results.next() != false) {
					numEvents++;
					eventIDs.add(results.getString(2));
				}
				results.close();
				statement.close();
			} else {

				// Next, get the metric names, and count them
				sql = new StringBuffer();
				sql.append("select m.id, m.name from metric m ");
				sql.append("where m.trial = ?");
				sql.append(" order by 1");
				statement = db.prepareStatement(sql.toString());
				statement.setInt(1, modelData.getTrial().getID());
				//PerfExplorerOutput.println(statement.toString());
				results = statement.executeQuery();
				numEvents = 0;
				eventIDs = new ArrayList();
				while (results.next() != false) {
					numEvents++;
					eventIDs.add(results.getString(2));
				}
				results.close();
				statement.close();
			}

			// get the number of threads
			sql = new StringBuffer();
			sql.append("select max(node), max(context), max(thread) ");
			sql.append("from interval_location_profile ");
			sql.append("inner join interval_event ");
			sql.append("on id = interval_event where trial = ? ");
			if (modelData.getCurrentSelection() instanceof Metric) {
				sql.append(" and metric = ? ");
			}
			statement = db.prepareStatement(sql.toString());
			statement.setInt(1, modelData.getTrial().getID());
			if (modelData.getCurrentSelection() instanceof Metric) {
				statement.setInt(2, ((Metric)(modelData.getCurrentSelection())).getID());
			}
			results = statement.executeQuery();
			//PerfExplorerOutput.println(statement.toString());
			if (results.next() != false) {
				nodes = results.getInt(1) + 1;
				contexts = results.getInt(2) + 1;
				threads = results.getInt(3) + 1;
				numTotalThreads = nodes * contexts * threads;
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			String error = "ERROR: Couldn't the constant settings from the database!";
			System.err.println(error);
            System.err.println(e.getMessage());
			e.printStackTrace();
			throw new PerfExplorerException(error, e);
		}
		/*
		PerfExplorerOutput.println("\nnumRows: " + numRows);
		PerfExplorerOutput.println("numCenterRows: " + numCenterRows);
		PerfExplorerOutput.println("nodes: " + nodes);
		PerfExplorerOutput.println("contexts: " + contexts);
		PerfExplorerOutput.println("threads: " + threads);
		PerfExplorerOutput.println("numTotalThreads: " + numTotalThreads);
		PerfExplorerOutput.println("numEvents: " + numEvents);
		PerfExplorerOutput.println(" Done!");
		*/
	}

	/**
	 * This method gets the raw performance data from the database.
	 * 
	 * @throws PerfExplorerException
	 */
	private void getRawData () throws PerfExplorerException {
		PerfExplorerOutput.print("Getting raw data...");
		rawData = factory.createRawData("Cluster Test", eventIDs, numTotalThreads, numEvents);
		ResultSet results = null;
		int currentFunction = 0;
		int functionIndex = -1;
		int rowIndex = 0;
		int threadIndex = 0;
		maximum = 0.0;
		try {
			DB db = session.db();
			PreparedStatement statement = null;
			StringBuffer sql = new StringBuffer();
			if (modelData.getDimensionReduction().equals(TransformationType.OVER_X_PERCENT)) {
				sql.append("select e.id, (p.node*");
				sql.append(contexts * threads);
				sql.append(") + (p.context*");
				sql.append(threads);
                
                if (db.getDBType().compareTo("oracle") == 0) {
                    sql.append(") + p.thread as thread, p.metric as metric, p.excl/1000000, ");
                } else {
                    sql.append(") + p.thread as thread, p.metric as metric, p.exclusive/1000000, ");
                }
				sql.append("p.inclusive/1000000, s.inclusive_percentage, s.exclusive_percentage ");
				sql.append("from interval_event e ");
				sql.append("inner join interval_mean_summary s ");
				sql.append("on e.id = s.interval_event and (s.exclusive_percentage > ");
				sql.append(modelData.getXPercent());
				sql.append("or s.inclusive_percentage = 100.0) ");
				sql.append(" left outer join interval_location_profile p ");
				sql.append("on e.id = p.interval_event ");
				sql.append("and p.metric = s.metric where e.trial = ? ");
			} else {
				sql.append("select e.id, (p.node*" + (contexts * threads) + "");
				sql.append(") + (p.context*" + threads + "");
                
                if (db.getDBType().compareTo("oracle") == 0) {
                    sql.append(") + p.thread as thread, p.metric as metric, p.excl, ");
                } else {
                    sql.append(") + p.thread as thread, p.metric as metric, p.exclusive, ");
                }

				sql.append("p.inclusive/1000000, p.inclusive_percentage ");
				sql.append("from interval_event e ");
				sql.append("left outer join interval_location_profile p ");
				sql.append("on e.id = p.interval_event where e.trial = ? ");
			}
			if (modelData.getCurrentSelection() instanceof Metric) {
				sql.append(" and p.metric = ? ");
			}
			sql.append(" and e.group_name not like '%TAU_CALLPATH%' ");
			sql.append(" order by 3,1,2 ");
			statement = db.prepareStatement(sql.toString());
			statement.setInt(1, modelData.getTrial().getID());
			if (modelData.getCurrentSelection() instanceof Metric) {
				statement.setInt(2, ((Metric)(modelData.getCurrentSelection())).getID());
			}
			//PerfExplorerOutput.println(statement.toString());
			results = statement.executeQuery();

			// if we are getting data for one metric, we are focussed on the events.
			// if we are getting data for all metrics, we are focussed on the metrics.
			int importantIndex = (modelData.getCurrentSelection() instanceof Metric) ? 1 : 3 ;

			// get the rows
			while (results.next() != false) {
				if (!(modelData.getDimensionReduction().equals(
					TransformationType.OVER_X_PERCENT)) || 
					(results.getDouble(7) > modelData.getXPercent())) {
					if (currentFunction != results.getInt(importantIndex)) {
						functionIndex++;
					}
					currentFunction = results.getInt(importantIndex);
					threadIndex = results.getInt(2);
					rawData.addValue(threadIndex, functionIndex, results.getDouble(4));
					if (maximum < results.getDouble(4))
						maximum = results.getDouble(4);
				} 
				// if this is the main method, save its values
				if (results.getDouble(6) == 100.0) {
					rawData.addMainValue(threadIndex, functionIndex, results.getDouble(5));
				}
				rowIndex++;
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			String error = "ERROR: Couldn't the raw data from the database!";
			System.err.println(error);
            System.err.println(e.getMessage());
			e.printStackTrace();
			throw new PerfExplorerException(error, e);
		} catch (ArrayIndexOutOfBoundsException e2) {
            System.err.println(e2.getMessage());
			e2.printStackTrace();
			PerfExplorerOutput.println("\ncurrentFunction: " + currentFunction);
			PerfExplorerOutput.println("functionIndex: " + functionIndex);
			PerfExplorerOutput.println("rowIndex: " + rowIndex);
			PerfExplorerOutput.println("threadIndex: " + threadIndex);
			System.exit(1);
		}
		PerfExplorerOutput.println(" Done!");
	}

	/**
	 * This method creates the dendrogram tree which represents the results
	 * of the hierarchical clustering, if it was done.
	 * 
	 * @param merge
	 * @param height
	 * @return
	 */
	public DendrogramTree createDendrogramTree (int[] merge, double[] height) {
	// public DendrogramTree createDendrogramTree (int[] merge) {
		DendrogramTree leftLeaf = null;
		DendrogramTree rightLeaf = null;
		DendrogramTree newTree = null;
		//Hashtable finder = new Hashtable(height.length);
		Hashtable finder = new Hashtable(merge.length);
		int j = 0;
		for (int i = 0 ; i < height.length ; i++) {
			if (merge[j] < 0)
				leftLeaf = new DendrogramTree(merge[j], 0.0);
			else
				leftLeaf = (DendrogramTree)finder.get(new Integer(merge[j]));
			j++;
			if (merge[j] < 0)
				rightLeaf = new DendrogramTree(merge[j], 0.0);
			else
				rightLeaf = (DendrogramTree)finder.get(new Integer(merge[j]));
			j++;
			newTree = new DendrogramTree(i+1, height[i]);
			newTree.setLeftAndRight(leftLeaf, rightLeaf);
			finder.put(new Integer(i+1), newTree);
		}
		return newTree;
	}

	/**
	 * @return
	 */
	public RawDataInterface doDimensionReduction() {
		// TODO: Implement dimension reduction.
		// for now, just return the original data.
		return rawData;
	}
	
	/* (non-Javadoc)
	 * @see java.lang.Runnable#run()
	 */
	public void run() {
		modelData = server.getNextRequest();
		if (modelData != null) {
			analysisID = modelData.getAnalysisID();
			try {
				PerfExplorerOutput.println("Processing " + modelData.toString());
				// get the number of events, threads, etc.
				getConstants();
				// get the raw data
				getRawData();
				// do dimension reduction, if requested
				RawDataInterface reducedData = doDimensionReduction();
				// find good initial centers
				//makeDendrogram(reducedData);
				// do it!
				if (modelData.getClusterMethod().equals(AnalysisType.K_MEANS)) {
					int maxClusters = (numTotalThreads <= modelData.getNumberOfClusters()) ? (numTotalThreads-1) : modelData.getNumberOfClusters();
					for (int i = 2 ; i <= maxClusters ; i++) {
						PerfExplorerOutput.println("Doing " + i + " clusters:" + modelData.toString());
						// create a cluster engine
						KMeansClusterInterface clusterer = factory.createKMeansEngine();
						clusterer.setInputData(reducedData);
						clusterer.setK(i);
						clusterer.findClusters();
						// get the centroids
						RawDataInterface centroids = clusterer.getClusterCentroids();
						RawDataInterface deviations = clusterer.getClusterStandardDeviations();
						int[] clusterSizes = clusterer.getClusterSizes();
						// do histograms
						File thumbnail = ImageUtils.generateThumbnail(modelData, clusterSizes, eventIDs);
						File chart = ImageUtils.generateImage(modelData, clusterSizes, eventIDs);
						// TODO - fix this to save the cluster sizes, too!
						chartType = ChartType.HISTOGRAM;
						saveAnalysisResult(centroids, deviations, thumbnail, chart);
						
						if (modelData.getCurrentSelection() instanceof Metric) {
						// do PCA breakdown
						PrincipalComponentsAnalysisInterface pca =
						factory.createPCAEngine(server.getCubeData(modelData));
						pca.setInputData(reducedData);
						pca.doPCA();
						// get the components
						RawDataInterface components = pca.getResults();
						pca.setClusterer(clusterer);
						RawDataInterface[] clusters = pca.getClusters();
						// do a scatterplot
						rCorrelation = 0.0;
						//for (int m = 0 ; m < i ; m++)
							//clusters[m].normalizeData(true);
						//PerfExplorerOutput.println("PCA Dimensions: " + components.numDimensions());
						thumbnail = ImageUtils.generateThumbnail(ChartType.PCA_SCATTERPLOT, modelData, clusters);
						chart = ImageUtils.generateImage(ChartType.PCA_SCATTERPLOT, modelData, components, clusters);
						saveAnalysisResult(components, components, thumbnail, chart);
						}
						
						// do virtual topology
						VirtualTopology vt = new VirtualTopology(modelData, clusterer);
						String filename = vt.getImage();
						String nail = vt.getThumbnail();
						saveAnalysisResult("Virtual Topology", filename, nail, false);
						
						// do mins
						chartType = ChartType.CLUSTER_MINIMUMS;
						thumbnail = ImageUtils.generateThumbnail(modelData, clusterer.getClusterMinimums(), deviations, eventIDs);
						chart = ImageUtils.generateImage(chartType, modelData, clusterer.getClusterMinimums(), deviations, eventIDs);
						saveAnalysisResult(clusterer.getClusterMinimums(), deviations, thumbnail, chart);
						// do averages
						chartType = ChartType.CLUSTER_AVERAGES;
						thumbnail = ImageUtils.generateThumbnail(modelData, centroids, deviations, eventIDs);
						chart = ImageUtils.generateImage(chartType, modelData, centroids, deviations, eventIDs);
						saveAnalysisResult(centroids, deviations, thumbnail, chart);
						// do maxes
						chartType = ChartType.CLUSTER_MAXIMUMS;
						thumbnail = ImageUtils.generateThumbnail(modelData, clusterer.getClusterMaximums(), deviations, eventIDs);
						chart = ImageUtils.generateImage(chartType, modelData, clusterer.getClusterMaximums(), deviations, eventIDs);
						saveAnalysisResult(clusterer.getClusterMaximums(), deviations, thumbnail, chart);
					}
				} else {
					PerfExplorerOutput.println("Doing Correlation Analysis...");
					// get the inclusive data
					//reducedData;
					chartType = ChartType.CORRELATION_SCATTERPLOT;
					/*
					for (int i = 0 ; i < reducedData.numDimensions() ; i++) {
						correlateToMain = true;
						rCorrelation = reducedData.getCorrelation(i,i);
						File thumbnail = generateThumbnail(reducedData, i, i);
						File chart = generateImage(reducedData, i, i);
						saveAnalysisResult(reducedData, reducedData, thumbnail, chart);	
						correlateToMain = false;
					}
					*/
					for (int i = 0 ; i < reducedData.numDimensions() ; i++) {
						for (int j = 0 ; j < reducedData.numDimensions() ; j++) {
							rCorrelation = reducedData.getCorrelation(i,j);
							File thumbnail = ImageUtils.generateThumbnail(chartType, modelData, reducedData, i, j, correlateToMain);
							File chart = ImageUtils.generateImage(chartType, modelData, reducedData, i, j, correlateToMain, rCorrelation);
							saveAnalysisResult(reducedData, reducedData, thumbnail, chart);	
						}
						//PerfExplorerOutput.println("Finished: " + (i+1) + " of " + reducedData.numDimensions());
					}
				}
			}catch (PerfExplorerException pee) {
            	System.err.println(pee.getMessage());
				pee.printStackTrace();
			}catch (ClusterException ce) {
            	System.err.println(ce.getMessage());
				ce.printStackTrace();
			}
			// let the server (main thread) know we are done
			server.taskFinished();
			modelData = null;
		} // else 
			//PerfExplorerOutput.println("nothing to do... ");
			
	}
}

