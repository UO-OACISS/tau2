

/**
 * 
 */
package edu.uoregon.tau.perfexplorer.server;



import java.io.File;
import java.io.FileInputStream;
import java.sql.PreparedStatement;
import java.util.Hashtable;
import java.util.List;
import java.util.TimerTask;

import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfexplorer.clustering.ClusterException;
import edu.uoregon.tau.perfexplorer.clustering.ClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.DBScanClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.DataNormalizer;
import edu.uoregon.tau.perfexplorer.clustering.HierarchicalCluster;
import edu.uoregon.tau.perfexplorer.clustering.KMeansClusterInterface;
import edu.uoregon.tau.perfexplorer.clustering.PrincipalComponentsAnalysisInterface;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.clustering.weka.AnalysisFactory;
import edu.uoregon.tau.perfexplorer.common.AnalysisType;
import edu.uoregon.tau.perfexplorer.common.ChartType;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerException;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;
import edu.uoregon.tau.perfexplorer.common.RMIPerfExplorerModel;


/**
 * This is a rewrite of the AnalysisTask class.
 * This class is intended to be a wrapper around data mining operations
 * available in Weka, R and Octave.  The orignal AnalysisTask class
 * only supported R directly.  This is intended to be an improvement...
 *
 * <P>CVS $Id: AnalysisTask.java,v 1.25 2009/11/25 09:15:35 khuck Exp $</P>
 * @author Kevin Huck
 * @version 0.1
 * @since 0.1
 */
public class AnalysisTask extends TimerTask {
    
    private ChartType chartType = ChartType.DENDROGRAM;
    
    private RMIPerfExplorerModel modelData = null;
    private PerfExplorerServer server = null;
    private DatabaseAPI session = null;
    private int analysisID = 0;
    private int numTotalThreads = 0;
    private RawDataInterface rawData = null;
    private List<String> eventIDs = null;
    private double rCorrelation = 0.0;
    private boolean correlateToMain = false;
    private int connectionIndex = 0;

    /**
    * Constructor.  The engine parameter passed in specifies which analysis
    * engine to use. 
    * 
    * @param server
    * @param engine
    */
    public AnalysisTask (PerfExplorerServer server, DatabaseAPI session, int connectionIndex) {
        super();
        this.server = server;
        this.session = session;
        this.connectionIndex = connectionIndex;
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
            PreparedStatement statement = null;
            StringBuilder buf = new StringBuilder();
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
            statement.setInt(7, Integer.parseInt(chartType.toString()));
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
	public void saveAnalysisResult (RawDataInterface centroids, RawDataInterface deviations, File thumbnail, File outfile, int i, int j) throws PerfExplorerException {
		// save the image in the database
		DB db = session.db();
		try {
			db.setAutoCommit(false);
			PreparedStatement statement = null;
			// for each centroid, save the data
			// TODO - MAKE THIS A GENERAL USE LATER!
			StringBuilder buf = new StringBuilder();
			buf.append("insert into analysis_result ");
			buf.append(" (analysis_settings, description, thumbnail_size, thumbnail, image_size, image, result_type) values (?, ?, ?, ?, ?, ?, ?)");
			statement = db.prepareStatement(buf.toString());
			statement.setInt(1, analysisID);
			if(i>=0 && j>=0){
			List<String> s= centroids.getEventNames();
			
			statement.setString(2, new String(s.get(i).substring(0, Math.min(s.get(i).length(),120))+" vs "+s.get(j).substring(0,Math.min(s.get(j).length(),120))));
			}
			else statement.setString(2, "analysis_result");
			
       		FileInputStream inStream = new FileInputStream(thumbnail);
       		statement.setInt(3,(int)outfile.length());
       		statement.setBinaryStream(4,inStream,(int)thumbnail.length());
       		FileInputStream inStream2 = new FileInputStream(outfile);
       		statement.setInt(5,(int)outfile.length());
       		statement.setBinaryStream(6,inStream2,(int)outfile.length());
            statement.setInt(7, Integer.parseInt(chartType.toString()));
			//PerfExplorerOutput.println(statement.toString());
       		statement.executeUpdate();
       		statement.close();
			db.commit();
			db.setAutoCommit(true);
		} catch (Exception e) {
			String error = "ERROR: Couldn't insert the analysis results into the database!";
			System.err.println(error);
            System.err.println(e.getMessage());
			e.printStackTrace();
			try { 
				db.setAutoCommit(true);
			} catch (Exception e2) {}
			throw new PerfExplorerException(error, e);
		}
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
		Hashtable<Integer,DendrogramTree> finder = new Hashtable<Integer,DendrogramTree>(merge.length);
		int j = 0;
		for (int i = 0 ; i < height.length ; i++) {
			if (merge[j] < 0)
				leftLeaf = new DendrogramTree(merge[j], 0.0);
			else
				leftLeaf = finder.get(new Integer(merge[j]));
			j++;
			if (merge[j] < 0)
				rightLeaf = new DendrogramTree(merge[j], 0.0);
			else
				rightLeaf = finder.get(new Integer(merge[j]));
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
		modelData = server.getNextRequest(connectionIndex);
		if (modelData != null) {
			analysisID = modelData.getAnalysisID();
			try {
				PerfExplorerOutput.println("Processing " + modelData.toString());
				// get the raw data
				rawData = DataUtils.getRawData(session, modelData);
				eventIDs = rawData.getEventNames();
				numTotalThreads = rawData.numVectors();
				// do dimension reduction, if requested
				RawDataInterface reducedData = doDimensionReduction();

				// do data normalization, if requested
				//DataNormalizer normalizer = AnalysisFactory.createDataNormalizer(reducedData);
				//reducedData=normalizer.getNormalizedData();
				
				if (modelData.getClusterMethod().equals(AnalysisType.K_MEANS)) {
					int maxClusters = (numTotalThreads <= modelData.getNumberOfClusters()) ? (numTotalThreads-1) : modelData.getNumberOfClusters();
					for (int i = 2 ; i <= maxClusters ; i++) {
						PerfExplorerOutput.println("Doing " + i + " clusters:" + modelData.toString());
						// create a cluster engine
						KMeansClusterInterface clusterer = AnalysisFactory.createKMeansEngine();
						clusterer.setInputData(reducedData);
						clusterer.setK(i);
						clusterer.findClusters();
						saveClusterResults (clusterer, reducedData);
					}
					System.out.println("...done clustering.");
				} else if (modelData.getClusterMethod().equals(AnalysisType.HIERARCHICAL)) {
					int maxClusters = (numTotalThreads <= modelData.getNumberOfClusters()) ? (numTotalThreads-1) : modelData.getNumberOfClusters();
					// create a cluster engine
					HierarchicalCluster clusterer = AnalysisFactory.createHierarchicalClusteringEngine();
					//long start = System.currentTimeMillis();
					clusterer.setInputData(reducedData);
					//long end = System.currentTimeMillis();
					for (int i = 2 ; i <= maxClusters ; i++) {
						PerfExplorerOutput.println("Doing " + i + " clusters:" + modelData.toString());
						// because all we are doing is changing the cut in the tree, we don't
						// have to the clustering more than once.  but we do have to reset
						// to cut with a new k.
						clusterer.reset();
						clusterer.setK(i);
						clusterer.findClusters();
						saveClusterResults (clusterer, reducedData);
					}
					System.out.println("...done clustering.");
				} else if (modelData.getClusterMethod().equals(AnalysisType.DBSCAN)) {
					// create a cluster engine
					DBScanClusterInterface clusterer = AnalysisFactory.createDBScanEngine();
					//long start = System.currentTimeMillis();
					// force normalization for DBSCAN, it seems to work better when guessing epsilon
					DataNormalizer normalizer = AnalysisFactory.createDataNormalizer(reducedData);
					reducedData=normalizer.getNormalizedData();
					clusterer.setInputData(reducedData);
					//long end = System.currentTimeMillis();

					double[] kDistances = clusterer.getKDistances();
					int epsilonIndex = clusterer.guessEpsilonIndex();
					//double epsilon = kDistances[epsilonIndex];
					double epsilon = kDistances[kDistances.length-1];

					int numClusters = 1;
					int maxTries = Math.min(50, epsilonIndex);
					int[] previousClusterSizes = null;
					while (maxTries > 0 && epsilonIndex > 0) {
						boolean newResult = false;
						PerfExplorerOutput.println("Clustering with Epsilon = " + epsilon + " : " + modelData.toString());
						clusterer.setError(epsilon);
						clusterer.findClusters();
						numClusters = clusterer.getClusterSizes().length;
						if (numClusters < 2) {
							System.out.println("No clusters found.");
							epsilonIndex--;
							epsilon = kDistances[epsilonIndex];
							maxTries--;
							continue;
						}
						int[] clusterSizes = clusterer.getClusterSizes();
						if (previousClusterSizes == null) {
							newResult = true;
						} else {
						    if (clusterSizes.length != previousClusterSizes.length && 
							  clusterSizes.length > previousClusterSizes.length) {
								newResult = true;
							/*} else {
								for (int i = 0 ; i < clusterSizes.length ; i++) {
									if (clusterSizes[i] != previousClusterSizes[i])
										newResult = true;
								}*/
							}
						}
						
						if (newResult) {
							System.out.println(Integer.toString(numClusters) + " clusters found - saving results.");
							saveClusterResults (clusterer, reducedData);
							previousClusterSizes = clusterSizes;
						} else {
							System.out.println("No change in cluster membership - ignoring results.");
						}
						epsilonIndex--;
						epsilon = kDistances[epsilonIndex];
						maxTries--;
					}

					System.out.println("...done clustering.");
				} else {
					PerfExplorerOutput.println("Doing Correlation Analysis...");
					chartType = ChartType.CORRELATION_SCATTERPLOT;
					for (int i = 0 ; i < reducedData.numDimensions() ; i++) {
						for (int j = 0 ; j < reducedData.numDimensions() ; j++) {
							rCorrelation = reducedData.getCorrelation(i,j);
							File thumbnail = ImageUtils.generateCorrelationScatterplotThumbnail(chartType, modelData, reducedData, i, j, correlateToMain);
							File chart = ImageUtils.generateCorrelationScatterplotImage(chartType, modelData, reducedData, i, j, correlateToMain, rCorrelation);
							saveAnalysisResult(reducedData, reducedData, thumbnail, chart, i, j);	
						}
						PerfExplorerOutput.println("Finished: " + (i+1) + " of " + reducedData.numDimensions());
					}
					System.out.println("...done with correlation.");
				}
			}catch (PerfExplorerException pee) {
            	System.err.println(pee.getMessage());
				pee.printStackTrace();
			}catch (ClusterException ce) {
            	System.err.println(ce.getMessage());
				ce.printStackTrace();
			}catch (Exception e) {
            	System.err.println(e.getMessage());
				e.printStackTrace();
			}
			// let the server (main thread) know we are done
			server.taskFinished(connectionIndex);
			modelData = null;
		} // else 
			//PerfExplorerOutput.println("nothing to do... ");			
	}
	
	private void saveClusterResults (ClusterInterface clusterer, RawDataInterface reducedData) 
			throws PerfExplorerException, ClusterException {
		// get the centroids
		RawDataInterface centroids = clusterer.getClusterCentroids();
		RawDataInterface deviations = clusterer.getClusterStandardDeviations();
		int[] clusterSizes = clusterer.getClusterSizes();

		// do histograms
		File thumbnail = ImageUtils.generateClusterSizeThumbnail(modelData, clusterSizes);//, eventIDs //TODO: These aren't used.
		File chart = ImageUtils.generateClusterSizeImage(modelData, clusterSizes);//, eventIDs
		chartType = ChartType.HISTOGRAM;
		saveAnalysisResult(centroids, deviations, thumbnail, chart, -1,-1);
		
		if (modelData.getCurrentSelection() instanceof Metric) {
			// do PCA breakdown
			PrincipalComponentsAnalysisInterface pca =
			AnalysisFactory.createPCAEngine(server.getCubeData(modelData, 4));
			pca.setInputData(reducedData);
			pca.doPCA();
			// get the components
			RawDataInterface components = pca.getResults();
			pca.setClusterer(clusterer);
			RawDataInterface[] clusters = pca.getClusters();
			// do a scatterplot
			rCorrelation = 0.0;
			thumbnail = ImageUtils.generateClusterScatterplotThumbnail(ChartType.PCA_SCATTERPLOT, modelData, clusters);
			boolean hasNoise = false;
			if (clusterer instanceof DBScanClusterInterface) 
				hasNoise = true;
			chart = ImageUtils.generateClusterScatterplotImage(ChartType.PCA_SCATTERPLOT, modelData, components, clusters, hasNoise);
			saveAnalysisResult(components, components, thumbnail, chart, -1, -1);
		}
		
		// do virtual topology
		VirtualTopology vt = new VirtualTopology(modelData, clusterer);
		String filename = vt.getImage();
		String nail = vt.getThumbnail();
		saveAnalysisResult("Virtual Topology", filename, nail, false);
		
		// do mins
		chartType = ChartType.CLUSTER_MINIMUMS;
		thumbnail = ImageUtils.generateBreakdownThumbnail(modelData, clusterer.getClusterMinimums(), deviations, eventIDs);
		chart = ImageUtils.generateBreakdownImage(chartType, modelData, clusterer.getClusterMinimums(), deviations, eventIDs);
		saveAnalysisResult(clusterer.getClusterMinimums(), deviations, thumbnail, chart, -1,-1);
		// do averages
		chartType = ChartType.CLUSTER_AVERAGES;
		thumbnail = ImageUtils.generateBreakdownThumbnail(modelData, centroids, deviations, eventIDs);
		chart = ImageUtils.generateBreakdownImage(chartType, modelData, centroids, deviations, eventIDs);
		saveAnalysisResult(centroids, deviations, thumbnail, chart, -1,-1);
		// do maxes
		chartType = ChartType.CLUSTER_MAXIMUMS;
		thumbnail = ImageUtils.generateBreakdownThumbnail(modelData, clusterer.getClusterMaximums(), deviations, eventIDs);
		chart = ImageUtils.generateBreakdownImage(chartType, modelData, clusterer.getClusterMaximums(), deviations, eventIDs);
		saveAnalysisResult(clusterer.getClusterMaximums(), deviations, thumbnail, chart, -1,-1);
	}
}

