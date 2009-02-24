package edu.uoregon.tau.perfexplorer.common;

import java.util.List;
import java.util.ArrayList;
import java.io.Serializable;

/**
 * This class is the main RMI object which contains the performance analysis
 * results from either cluster or some other type of background
 * analysis.
 *
 * <P>CVS $Id: RMIPerformanceResults.java,v 1.2 2009/02/24 00:53:37 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class RMIPerformanceResults implements Serializable {
	protected List images = null;  // of File objects
	protected List thumbnails = null;  // of File objects
	protected List ids = null; // of Strings
	protected List ks = null; // of Strings
	protected List descriptions = null;  // of String objects
	protected List clusterCentroids = null; // of Lists of Centroids
	protected List clusterDeviations = null; // of Lists of Deviations

	public RMIPerformanceResults () {
		images = new ArrayList();
		thumbnails = new ArrayList();
		ids = new ArrayList();
		ks = new ArrayList();
		descriptions = new ArrayList();
		clusterCentroids = new ArrayList();
		clusterDeviations = new ArrayList();
	}

	public List getImages() {
		return images;
	}

	public List getThumbnails() {
		return thumbnails;
	}

	public List getDescriptions() {
		return descriptions;
	}

	public List getIDs() {
		return ids;
	}

	public List getKs() {
		return ks;
	}

	public List getClusterCentroids() {
		return clusterCentroids;
	}

	public List getClusterDeviations() {
		return clusterDeviations;
	}

	public void setImages(List images) {
		this.images = images;
	}

	public void setThumbnails(List thumbnails) {
		this.thumbnails = thumbnails;
	}

	public void setClusterCentroids(List clusterCentroids) {
		this.clusterCentroids = clusterCentroids;
	}

	public void setClusterDeviations(List clusterDeviations) {
		this.clusterDeviations = clusterDeviations;
	}

	public void setDescriptions(List descriptions) {
		this.descriptions = descriptions;
	}

	public void setIDs(List ids) {
		this.ids = ids;
	}

	public void setKs(List ks) {
		this.ks = ks;
	}

	public int getResultCount () {
		return (descriptions == null) ? 0 : descriptions.size();
	}
}
