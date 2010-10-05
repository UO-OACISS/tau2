package edu.uoregon.tau.perfdmf.analysis;

import java.util.Vector;

public class ScalabilityResults extends Vector<ScalabilityResult> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8157675547698994075L;

	public String toString(boolean speedup) {
		// loop through the results, and print them out CSV style
		StringBuffer output = new StringBuffer();

		for (int i = 0 ; i < this.size() ; i++) {
			ScalabilityResult next = (ScalabilityResult)elementAt(i);
			output.append(next.toString(speedup));
			output.append("\n");
		}

		return output.toString();
	}

	public double[] getAverageData (String function, boolean speedup) {
		for (int i = 0 ; i < this.size() ; i++) {
			ScalabilityResult next = (ScalabilityResult)elementAt(i);
			if (next.functionName.equals(function)) {
				if (speedup) {
					return next.getAverageSpeedup();
				} else {
					return next.average;
				}
			}
		}
		return null;
	}

	public double[] getMaximumData (String function, boolean speedup) {
		for (int i = 0 ; i < this.size() ; i++) {
			ScalabilityResult next = (ScalabilityResult)elementAt(i);
			if (next.functionName.equals(function)) {
				if (speedup) {
					return next.getMaximumSpeedup();
				} else {
					return next.maximum;
				}
			}
		}
		return null;
	}

	public double[] getMinimumData (String function, boolean speedup) {
		for (int i = 0 ; i < this.size() ; i++) {
			ScalabilityResult next = (ScalabilityResult)elementAt(i);
			if (next.functionName.equals(function)) {
				if (speedup) {
					return next.getMinimumSpeedup();
				} else {
					return next.minimum;
				}
			}
		}
		return null;
	}
}
