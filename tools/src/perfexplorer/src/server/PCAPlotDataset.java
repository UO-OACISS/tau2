package server;

import org.jfree.data.xy.AbstractXYDataset;
import clustering.KMeansClusterInterface;
import clustering.RawDataInterface;
import weka.core.Instances;

/**
 * Dataset to store scatterplot data.
 * The JFreeChart API requires that client applications extend the 
 * AbstractXYDataset class to implement the data to be plotted in a scatterplot.
 * This is essentially a wrapper around the RawDataInterface class.
 * 
 * <P>CVS $Id: PCAPlotDataset.java,v 1.2 2005/07/15 20:56:44 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class PCAPlotDataset extends AbstractXYDataset {

	private RawDataInterface pcaData = null;
	private RawDataInterface rawData = null;
	private KMeansClusterInterface clusterer = null;
	private Instances[] clusters = null;
	private int x = 0;
	private int y = 1;
	private int k = 0;

	/**
	 * Constructor.
	 * 
	 * @param pcaData
	 * @param rawData
	 * @param clusterer
	 */
	public PCAPlotDataset(RawDataInterface pcaData, RawDataInterface rawData, KMeansClusterInterface clusterer) {
		super();
		this.pcaData = rawData;
		// get a reference to the clusterer
		this.clusterer = clusterer;
		// get the number of clusters
		this.k = clusterer.getK();
		this.clusters = new Instances[k];
		Instances raw = (Instances) rawData.getData();
		Instances pca = (Instances) pcaData.getData();
		/*
		 * after PCA, the two greatest components are at the END of the list
		 * of components.  Therefore, get the last and second-to-last
		 * components.
		 */
		//System.out.println("numAttributes: " + pca.numAttributes());
		if (pca.numAttributes() > 1) {
			x = pca.numAttributes() - 1;
			y = pca.numAttributes() - 2;
		} else {
			y = 0;
		}
		
		for (int i = 0 ; i < k ; i++) 
			this.clusters[i] = new Instances(pca, 0);
		/*
		 * For each element in the raw data, determine which cluster it
		 * belongs in.  That will determine what color the point should be. 
		 */
		for (int i = 0 ; i < rawData.numVectors() ; i++) {
			int location = clusterer.clusterInstance(i);
			clusters[location].add(pca.instance(i));
		}
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesCount()
	 */
	public int getSeriesCount() {
		return k;
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.general.SeriesDataset#getSeriesName(int)
	 */
	public String getSeriesName(int arg0) {
		return new String("Cluster " + arg0);
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getItemCount(int)
	 */
	public int getItemCount(int arg0) {
		return clusters[arg0].numInstances();
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getX(int, int)
	 */
	public Number getX(int arg0, int arg1) {
		return new Double(clusters[arg0].instance(arg1).value(x));
		//return new Double(data.getValue(arg1, x));
	}

	/* (non-Javadoc)
	 * @see org.jfree.data.xy.XYDataset#getY(int, int)
	 */
	public Number getY(int arg0, int arg1) {
		//System.out.println("Getting Y: " + arg0 + ", " + arg1 + ", " + y);
		return new Double(clusters[arg0].instance(arg1).value(y));
		//return new Double(data.getValue(arg1, y));
	}
}
