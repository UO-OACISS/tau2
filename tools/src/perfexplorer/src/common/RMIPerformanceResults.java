package edu.uoregon.tau.perfexplorer.common;

import java.util.List;
import java.util.ArrayList;
import java.io.File;
import java.io.Serializable;

/**
 * This class is the main RMI object which contains the performance analysis
 * results from either cluster or some other type of background
 * analysis.
 *
 * <P>CVS $Id: RMIPerformanceResults.java,v 1.3 2009/02/27 00:45:09 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class RMIPerformanceResults implements Serializable {
	protected List<byte[]> images = null;  // of File objects
	protected List<byte[]> thumbnails = null;  // of File objects
	protected List<String> ids = null; // of Strings
	protected List<String> ks = null; // of Strings
	protected List<String> descriptions = null;  // of String objects
	protected List<String> clusterCentroids = null; // of Lists of Centroids
	protected List<String> clusterDeviations = null; // of Lists of Deviations

	public RMIPerformanceResults () {
		images = new ArrayList<byte[]>();
		thumbnails = new ArrayList<byte[]>();
		ids = new ArrayList<String>();
		ks = new ArrayList<String>();
		descriptions = new ArrayList<String>();
		clusterCentroids = new ArrayList<String>();
		clusterDeviations = new ArrayList<String>();
	}

	public List<byte[]> getImages() {
		return images;
	}

	public List<byte[]> getThumbnails() {
		return thumbnails;
	}

	public List<String> getDescriptions() {
		return descriptions;
	}

	public List<String> getIDs() {
		return ids;
	}

	public List<String> getKs() {
		return ks;
	}

	public List<String> getClusterCentroids() {
		return clusterCentroids;
	}

	public List<String> getClusterDeviations() {
		return clusterDeviations;
	}

	public void setImages(List<byte[]> images) {
		this.images = images;
	}

	public void setThumbnails(List<byte[]> thumbnails) {
		this.thumbnails = thumbnails;
	}

	public void setClusterCentroids(List<String> clusterCentroids) {
		this.clusterCentroids = clusterCentroids;
	}

	public void setClusterDeviations(List<String> clusterDeviations) {
		this.clusterDeviations = clusterDeviations;
	}

	public void setDescriptions(List<String> descriptions) {
		this.descriptions = descriptions;
	}

	public void setIDs(List<String> ids) {
		this.ids = ids;
	}

	public void setKs(List<String> ks) {
		this.ks = ks;
	}

	public int getResultCount () {
		return (descriptions == null) ? 0 : descriptions.size();
	}
}
