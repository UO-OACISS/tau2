
package edu.uoregon.tau.perfexplorer.clustering.weka;

import weka.core.Utils;
import edu.uoregon.tau.perfexplorer.clustering.Utilities;

public class WekaUtilities implements Utilities {

	public double doCorrelation (double[] y1, double[] y2, int arrayLength) {
		return (Utils.correlation(y1, y2, arrayLength));
	}
}