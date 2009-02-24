
package edu.uoregon.tau.perfexplorer.clustering.weka;

import edu.uoregon.tau.perfexplorer.clustering.Utilities;
import weka.core.Utils;

public class WekaUtilities implements Utilities {

	public double doCorrelation (double[] y1, double[] y2, int arrayLength) {
		return (Utils.correlation(y1, y2, arrayLength));
	}
}