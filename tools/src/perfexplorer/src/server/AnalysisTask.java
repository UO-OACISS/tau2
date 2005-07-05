package server;

import clustering.DendrogramTree;
import common.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.database.*;
import java.sql.SQLException;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.Hashtable;
import java.util.TimerTask;
import java.io.File;
import java.io.FileInputStream;
import org.omegahat.R.Java.REvaluator;
import org.omegahat.R.Java.ROmegahatInterpreter;

/**
 * Description
 *
 * <P>CVS $Id: AnalysisTask.java,v 1.1 2005/07/05 22:29:53 amorris Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class AnalysisTask extends TimerTask {

	private RMIPerfExplorerModel modelData = null;
	private PerfExplorerServer server = null;
	private ROmegahatInterpreter rInterpreter = null;
	private REvaluator rEvaluator = null;
	private DendrogramTree dendrogramTree = null;
	private int analysisID = 0;
	private int numRows = 0;
	private int numCenterRows = 0;
	private int numTotalThreads = 0;
	private int numEvents = 0;
	private int nodes = 0;
	private int contexts = 0;
	private int threads = 0;
	private Object rawData[][] = null;
	private int eventID[] = null;
	private double maximum = 0.0;
	private static final int reducedDimension = 12;

	public AnalysisTask (PerfExplorerServer server) {
		super();
		this.server = server;
	}

	private void makeDendrogram() throws PerfExplorerException {
		System.out.print("Copying data...");
		rEvaluator.voidEval("raw <- matrix(0, nrow=" + numTotalThreads + ", ncol=" + numEvents + ")");
		for (int i = 0 ; i < numTotalThreads ; i++) {
			for (int j = 0 ; j < numEvents ; j++) {
				if (rawData[i][j] != null)
					rEvaluator.voidEval("raw[" + (i+1) + "," + (j+1) + "] <- " + rawData[i][j]);
			}
		}
		System.out.println(" Done!");
		if (modelData.getDimensionReduction().equals(RMIPerfExplorerModel.LINEAR_PROJECTION)) {
			System.out.print("Reducing Dimensions...");
			int numReduced = numTotalThreads * reducedDimension;
			rEvaluator.voidEval("reducer <- matrix((runif("+numReduced+",0,1)), nrow=" + numEvents + ", ncol="+reducedDimension+")");
			rEvaluator.voidEval("raw <- crossprod(t(raw), reducer)");
			numEvents = reducedDimension;
			System.out.println(" Done!");
		}
		if (numTotalThreads < 4098) { // arbitrary choice...
			System.out.print("Getting distances...");
			rEvaluator.voidEval("threads <- dist(raw, method=\"manhattan\")");
			System.out.println(" Done!");
			System.out.print("Hierarchical clustering...");
			rEvaluator.voidEval("hcgtr <- hclust(threads, method=\"average\")");
			int[] merge = (int[])rEvaluator.eval("t(hcgtr$merge)");
			double[] height = (double[])rEvaluator.eval("hcgtr$height");
			// int[] merge = rInterpreter.call("makeDendrogram", rawData, numTotalThreads, numEvents);
			dendrogramTree = createDendrogramTree(merge, height);
			// dendrogramTree = createDendrogramTree(merge);
			rEvaluator.voidEval("dend <- as.dendrogram(hcgtr)");
			System.out.println(" Done!");
			System.out.print("Making png image...");
			String description = "dendrogram." + 
				modelData.toString();
			description = description.replaceAll(":",".");
			String shortDescription =
			modelData.toShortString();
			String fileName = "/tmp/dendrogram." + shortDescription + ".png";
			String thumbnail = "/tmp/dendrogram.thumb." + shortDescription + ".jpg";
			rEvaluator.voidEval("png(\"" + fileName + "\",width=800, height=400)");
			rEvaluator.voidEval("plot (dend, main=\"" + description + "\", edge.root=FALSE,horiz=FALSE,axes=TRUE)");
			rEvaluator.voidEval("dev.off()");
			System.out.println(" Done!");
			saveAnalysisResult(description, fileName, thumbnail, true);
		}
	}

	public void saveAnalysisResult (String description, String fileName, String thumbnail, boolean big) throws PerfExplorerException {
		// create the thumbnail
		Thumbnail.createThumbnail(fileName, thumbnail, big);
		// save the image in the database
		try {
			PerfExplorerServer.getServer().getControl().WAIT("saveAnalysisResult");
			DB db = PerfExplorerServer.getServer().getDB();
			//db.setAutoCommit(false);
			PreparedStatement statement = null;
			statement = db.prepareStatement("insert into analysis_result (analysis_settings, description, thumbnail_size, image_size, thumbnail, image, result_type) values (?, ?, ?, ?, ?, ?, 0)");
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
       		statement.executeUpdate();
			//db.commit();
       		v_fis.close();
       		statement.close();
			v_file.delete();
			v_thumb.delete();
			PerfExplorerServer.getServer().getControl().SIGNAL("saveAnalysisResult");
		} catch (Exception e) {
			String error = "ERROR: Couldn't insert the analysis results into the database!";
			System.out.println(error);
			e.printStackTrace();
			PerfExplorerServer.getServer().getControl().SIGNAL("saveAnalysisResult");
			throw new PerfExplorerException(error, e);
		}
	}

	private void doKMeansClustering(boolean haveCenters) throws PerfExplorerException {
		rEvaluator.voidEval("traw <- t(raw)");
		// create a palette!
		rEvaluator.voidEval("first <- rgb(0,0,0:15,max=15)");
		rEvaluator.voidEval("second <- rgb(0,0:15,15:0,max=15)");
		rEvaluator.voidEval("third <- rgb(0:15,15:0,0,max=15)");
		rEvaluator.voidEval("fourth <- rgb(15,0:15,0,max=15)");
		//rEvaluator.voidEval("fifth <- rgb(15,15,0:15,max=15)");
		rEvaluator.voidEval("all <- c(first, second, third, fourth)");
		int maxClusters = (numTotalThreads <= modelData.getNumberOfClusters()) ? (numTotalThreads-1) : modelData.getNumberOfClusters();
		for (int i = 2 ; i <= maxClusters ; i++) {
			if (haveCenters) {
				System.out.print("Making " + i + " centers...");
				int[]centerIndexes = dendrogramTree.findCenters(i);
				rEvaluator.voidEval("centers <- matrix(0, nrow=" + i + ", ncol=" + numEvents + ")");
				System.out.print("centers: ");
				for (int j = 1 ; j <= i ; j++) {
					System.out.print(centerIndexes[j-1]);
					rEvaluator.voidEval("centers[" + j + ",] <- raw[" + centerIndexes[j-1] + ",]");
					if (j != i)
						System.out.print(",");
				}
				System.out.println(" Done!");
			}

			System.out.print("Doing k-means clustering...");
			if (haveCenters)
				rEvaluator.voidEval("cl <- kmeans(raw, centers, 20)");
			else
				rEvaluator.voidEval("cl <- kmeans(raw, " + i + ", 20)");
			System.out.println(" Done!");

			String description = modelData.toString() +
				i + "_clusters";
			description = description.replaceAll(":",".");
			String shortDescription = modelData.toShortString() + i;

			System.out.print("Making png image...");
			String fileName = "/tmp/clusterSizes." + shortDescription + ".png";
			String thumbnail = "/tmp/clusterSizes.thumb." + shortDescription + ".jpg";
			rEvaluator.voidEval("png(\"" + fileName + "\")");
			rEvaluator.voidEval("barplot (cl$size, main=\"cluster sizes: " + description + "\", xlab=\"count\", ylab=\"cluster\", horiz=TRUE)");
			rEvaluator.voidEval("dev.off()");
			System.out.println(" Done!");
			saveAnalysisResult(description, fileName, thumbnail, false);

			System.out.print("Making colormap image...");
			if (haveCenters) {
				if (numTotalThreads > 32 && threads != 1) {
					rEvaluator.voidEval("mymat <- matrix(cl$cluster, nrow="+threads * contexts+", ncol="+nodes+")");
					rEvaluator.voidEval("mymat <- t(mymat)");
				} else {
					rEvaluator.voidEval("mymat <- matrix(cl$cluster, nrow="+(numTotalThreads/16)+", ncol=16)");
				}
			} else {
				rEvaluator.voidEval("mymat <- matrix(cl$cluster, nrow="+(numTotalThreads/32)+", ncol=32)");
			}
			fileName = "/tmp/clusterimage." + shortDescription + ".png";
			thumbnail = "/tmp/clusterimage.thumb." + shortDescription + ".jpg";
			rEvaluator.voidEval("png(\"" + fileName + "\")");
			rEvaluator.voidEval("image(mymat, col=all, axes=FALSE)");
			rEvaluator.voidEval("dev.off()");
			System.out.println(" Done!");
			saveAnalysisResult(description, fileName, thumbnail, false);

			System.out.print("Getting averages, mins, maxes...");
			// get the averages for each cluster, and graph them
			rEvaluator.voidEval("maxes <- matrix(0.0, nrow=" + i + ", ncol=" + numEvents + ")");
			rEvaluator.voidEval("mins <- matrix(" + maximum + ", nrow=" + i + ", ncol=" + numEvents + ")");
			rEvaluator.voidEval("totals <- matrix(0.0, nrow=" + i + ", ncol=" + numEvents + ")");
			rEvaluator.voidEval("counts <- matrix(0.0, nrow=" + i + ", ncol=" + numEvents + ")");
			rEvaluator.voidEval("ci <- 0");
			for (int m = 1 ; m <= numTotalThreads ; m++) {
  				rEvaluator.voidEval("ci <- cl$cluster["+m+"]");
				for (int n = 1 ; n <= numEvents ; n++) {
   					rEvaluator.voidEval("if (raw["+m+","+n+"] > maxes[ci,"+n+"]) maxes[ci,"+n+"] <- raw["+m+","+n+"]");
   					rEvaluator.voidEval("if (raw["+m+","+n+"] < mins[ci,"+n+"]) mins[ci,"+n+"] <- raw["+m+","+n+"]");
   					rEvaluator.voidEval("totals[ci,"+n+"] = totals[ci,"+n+"] + raw["+m+","+n+"]");
   					rEvaluator.voidEval("counts[ci,"+n+"] = counts[ci,"+n+"] + 1");
				}
			}
			rEvaluator.voidEval("avgs <- totals / counts");
			System.out.println(" Done!");
/*
			rEvaluator.voidEval("for (i in 1:" + numTotalThreads + ") {\nci <- cl$cluster[i] for (j in 1:" + numEvents + ") {\nif (raw[i,j] > maxes[ci,j]) maxes[ci,j] <- raw[i,j]\nif (raw[i,j] < mins[ci,j]) mins[ci,j] <- raw[i,j]\ntotals[ci,j] = totals[ci,j] + raw[i,j]\ncounts[ci,j] = counts[ci,j] + 1 } } ");

			double[] val = (double[])rEvaluator.eval("mins");
			for (int m = 0 ; m < 10 ; m++)
				System.out.println("val[" + m + "]: " + val[m]);
*/

			System.out.print("Making png image of averages...");
			fileName = "/tmp/barplot_averages." + shortDescription + ".png";
			thumbnail = "/tmp/barplot_averages.thumb." + shortDescription + ".jpg";
			rEvaluator.voidEval("png(\"" + fileName + "\")");
			if (modelData.getCurrentSelection() instanceof Metric)
				rEvaluator.voidEval("barplot(t(avgs), main=\"barplot averages: " + description + "\", xlab=\"cluster\", ylab=\"" + ((Metric)(modelData.getCurrentSelection())).getName() + "\", col=1:" + numEvents + ", horiz=TRUE)");
			else
				rEvaluator.voidEval("barplot(t(avgs), main=\"barplot averages: " + description + "\", xlab=\"cluster\", ylab=\"" + "all metrics" + "\", col=1:" + numEvents + ", horiz=TRUE)");
			rEvaluator.voidEval("dev.off()");
			System.out.println(" Done!");
			saveAnalysisResult(description, fileName, thumbnail, false);

			System.out.print("Making png image of maxes...");
			fileName = "/tmp/barplot_maxes." + shortDescription + ".png";
			thumbnail = "/tmp/barplot_maxes.thumb." + shortDescription + ".jpg";
			rEvaluator.voidEval("png(\"" + fileName + "\")");
			if (modelData.getCurrentSelection() instanceof Metric)
				rEvaluator.voidEval("barplot(t(maxes), main=\"barplot maxes: " + description + "\", xlab=\"cluster\", ylab=\"" + ((Metric)(modelData.getCurrentSelection())).getName() + "\", col=1:" + numEvents + ", horiz=TRUE)");
			else
				rEvaluator.voidEval("barplot(t(maxes), main=\"barplot maxes: " + description + "\", xlab=\"cluster\", ylab=\"" + "all metrics" + "\", col=1:" + numEvents + ", horiz=TRUE)");
			rEvaluator.voidEval("dev.off()");
			System.out.println(" Done!");
			saveAnalysisResult(description, fileName, thumbnail, false);

			System.out.print("Making png image of mins...");
			fileName = "/tmp/barplot_mins." + shortDescription + ".png";
			thumbnail = "/tmp/barplot_mins.thumb." + shortDescription + ".jpg";
			rEvaluator.voidEval("png(\"" + fileName + "\")");
			if (modelData.getCurrentSelection() instanceof Metric)
				rEvaluator.voidEval("barplot(t(mins), main=\"barplot mins: " + description + "\", xlab=\"cluster\", ylab=\"" + ((Metric)(modelData.getCurrentSelection())).getName() + "\", col=1:" + numEvents + ", horiz=TRUE)");
			else
				rEvaluator.voidEval("barplot(t(mins), main=\"barplot mins: " + description + "\", xlab=\"cluster\", ylab=\"" + "all metrics" + "\", col=1:" + numEvents + ", horiz=TRUE)");
			rEvaluator.voidEval("dev.off()");
			System.out.println(" Done!");
			saveAnalysisResult(description, fileName, thumbnail, false);
		}
	}

	private void getConstants () throws PerfExplorerException {
		System.out.print("Getting constants...");
		try {
			PerfExplorerServer.getServer().getControl().WAIT("getContstants");
			DB db = PerfExplorerServer.getServer().getDB();
			PreparedStatement statement = null;
			String sql = null;
			if (modelData.getDimensionReduction().equals(RMIPerfExplorerModel.OVER_X_PERCENT)) {
				sql = new String("select count(p.exclusive) from interval_event e inner join interval_mean_summary s on e.id = s.interval_event and s.exclusive_percentage > " + modelData.getXPercent() + " left outer join interval_location_profile p on e.id = p.interval_event and s.metric = p.metric where e.trial = ?");
			} else {
				sql = new String("select count(p.exclusive) from interval_event e left outer join interval_location_profile p on e.id = p.interval_event where e.trial = ?");
			}
			if (modelData.getCurrentSelection() instanceof Metric) {
				sql += " and p.metric = ?";
			}
			//System.out.println(sql);
			statement = db.prepareStatement(sql);
			statement.setInt(1, modelData.getTrial().getID());
			if (modelData.getCurrentSelection() instanceof Metric) {
				statement.setInt(2, ((Metric)(modelData.getCurrentSelection())).getID());
			}
			ResultSet results = statement.executeQuery();
			// get the number of rows
			if (results.next() != false) {
				numRows = results.getInt(1);
			}
			results.close();
			PerfExplorerServer.getServer().getControl().SIGNAL("getConstants");
			sql = null;

			PerfExplorerServer.getServer().getControl().WAIT("getContstants");
			if (modelData.getDimensionReduction().equals(RMIPerfExplorerModel.OVER_X_PERCENT)) {
				sql = new String("select count(e.id) from interval_event e inner join interval_mean_summary s on e.id = s.interval_event and s.exclusive_percentage > " + modelData.getXPercent() + " where e.trial = ? ");
				if (modelData.getCurrentSelection() instanceof Metric) {
					sql += " and s.metric = ? ";
				}
			} else {
				sql = new String("select count(e.id) from interval_event e where e.trial = ?");
			}
			//System.out.println(sql);
			statement = db.prepareStatement(sql);
			statement.setInt(1, modelData.getTrial().getID());
			if (modelData.getDimensionReduction().equals(RMIPerfExplorerModel.OVER_X_PERCENT)) {
				if (modelData.getCurrentSelection() instanceof Metric) {
					statement.setInt(2, ((Metric)(modelData.getCurrentSelection())).getID());
				}
			}
			results = statement.executeQuery();
			// get the number of events
			if (results.next() != false) {
				numEvents = results.getInt(1);
			}
			results.close();
			PerfExplorerServer.getServer().getControl().SIGNAL("getConstants");
			sql = null;

			PerfExplorerServer.getServer().getControl().WAIT("getContstants");
			// get the number of threads
			sql = new String("select max(node), max(context), max(thread) from interval_location_profile inner join interval_event on id = interval_event where trial = ? ");
			if (modelData.getCurrentSelection() instanceof Metric) {
				sql += " and metric = ? ";
			}
			statement = db.prepareStatement(sql);
			statement.setInt(1, modelData.getTrial().getID());
			if (modelData.getCurrentSelection() instanceof Metric) {
				statement.setInt(2, ((Metric)(modelData.getCurrentSelection())).getID());
			}
			results = statement.executeQuery();
			if (results.next() != false) {
				nodes = results.getInt(1) + 1;
				contexts = results.getInt(2) + 1;
				threads = results.getInt(3) + 1;
				numTotalThreads = nodes * contexts * threads;
			}
			results.close();

			// 

			statement.close();
			PerfExplorerServer.getServer().getControl().SIGNAL("getConstants");
		} catch (SQLException e) {
			String error = "ERROR: Couldn't the constant settings from the database!";
			System.out.println(error);
			e.printStackTrace();
			PerfExplorerServer.getServer().getControl().SIGNAL("getConstants");
			throw new PerfExplorerException(error, e);
		}
		/*System.out.println("\nnumRows: " + numRows);
		System.out.println("numCenterRows: " + numCenterRows);
		System.out.println("nodes: " + nodes);
		System.out.println("contexts: " + contexts);
		System.out.println("threads: " + threads);
		System.out.println("numTotalThreads: " + numTotalThreads);
		System.out.println("numEvents: " + numEvents);*/
		System.out.println(" Done!");
	}

	private void getRawData () throws PerfExplorerException {
		System.out.print("Getting raw data...");
		rawData = new Double[numTotalThreads][numEvents];
		eventID = new int[numEvents];
		ResultSet results = null;
		int currentFunction = 0;
		int functionIndex = -1;
		int rowIndex = 0;
		int threadIndex = 0;
		maximum = 0.0;
		try {
			PerfExplorerServer.getServer().getControl().WAIT("getRawData");
			DB db = PerfExplorerServer.getServer().getDB();
			PreparedStatement statement = null;
			String sql = null;
			if (modelData.getDimensionReduction().equals(RMIPerfExplorerModel.OVER_X_PERCENT)) {
				sql = new String("select e.id, (p.node*" + (contexts * threads) + ") + (p.context*" + threads + ") + p.thread as thread, p.exclusive from interval_event e inner join interval_mean_summary s on e.id = s.interval_event and s.exclusive_percentage > " + modelData.getXPercent() + " left outer join interval_location_profile p on e.id = p.interval_event and p.metric = s.metric where e.trial = ? ");
			} else {
				sql = new String("select e.id, (p.node*" + (contexts * threads) + ") + (p.context*" + threads + ") + p.thread as thread, p.exclusive from interval_event e left outer join interval_location_profile p on e.id = p.interval_event where e.trial = ? ");
			}
			if (modelData.getCurrentSelection() instanceof Metric) {
				sql += " and p.metric = ? ";
			}
			sql += " order by 1,2 ";
			//System.out.println(sql);
			statement = db.prepareStatement(sql);
			statement.setInt(1, modelData.getTrial().getID());
			if (modelData.getCurrentSelection() instanceof Metric) {
				statement.setInt(2, ((Metric)(modelData.getCurrentSelection())).getID());
			}
			results = statement.executeQuery();

			// get the rows
			while (results.next() != false) {
				if (currentFunction != results.getInt(1)) {
					functionIndex++;
					eventID[functionIndex] = results.getInt(1);
				}
				currentFunction = results.getInt(1);
				threadIndex = results.getInt(2);
				rawData[threadIndex][functionIndex] = new Double(results.getDouble(3));
				if (maximum < results.getDouble(3))
					maximum = results.getDouble(3);
				rowIndex++;
			}
			results.close();
			statement.close();
			PerfExplorerServer.getServer().getControl().SIGNAL("getRawData");
		} catch (SQLException e) {
			String error = "ERROR: Couldn't the raw data from the database!";
			System.out.println(error);
			e.printStackTrace();
			PerfExplorerServer.getServer().getControl().SIGNAL("getRawData");
			throw new PerfExplorerException(error, e);
		} catch (ArrayIndexOutOfBoundsException e2) {
			e2.printStackTrace();
			System.out.println("\ncurrentFunction: " + currentFunction);
			System.out.println("functionIndex: " + functionIndex);
			System.out.println("rowIndex: " + rowIndex);
			System.out.println("threadIndex: " + threadIndex);
			PerfExplorerServer.getServer().getControl().SIGNAL("getRawData");
			System.exit(1);
		}
		System.out.println(" Done!");
	}

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

	public void run() {
		modelData = server.getNextRequest();
		if (modelData != null) {
			analysisID = modelData.getAnalysisID();
			try {
				System.out.println("Processing " + modelData.toString());
				getConstants();
				getRawData();
				rInterpreter = server.getRInterpreter();
				rEvaluator = server.getREvaluator();
				makeDendrogram();
				doKMeansClustering(numTotalThreads < 4098);
			}catch (PerfExplorerException e) {
			}
			// let the server (main thread) know we are done
			server.taskFinished();
			modelData = null;
		} // else 
			//System.out.println("nothing to do... ");
			
	}

}

