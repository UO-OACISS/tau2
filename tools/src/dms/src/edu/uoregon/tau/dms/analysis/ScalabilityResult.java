package edu.uoregon.tau.dms.analysis;

public class ScalabilityResult {
	public String functionName = null;
	public int trialCount = 0;
	public int[] threadCount = null;
	public double[] minimum = null;
	public double[] average = null;
	public double[] maximum = null;
	public double[] stddev = null;

	public ScalabilityResult (String functionName, int trialCount) {
		this.trialCount = trialCount;
		this.functionName = new String(functionName);
		this.threadCount = new int[trialCount];
		this.minimum = new double[trialCount];
		this.average = new double[trialCount];
		this.maximum = new double[trialCount];
		this.stddev = new double[trialCount];
	}

	public String toString(boolean speedup) {
		StringBuffer output = new StringBuffer();

		output.append(functionName.replace(',',' ') + " minimum");
		for (int i = 0 ; i < trialCount ; i++) {
			if (speedup)
				output.append("," + minimum[0] / minimum[i]);
			else
				output.append("," + minimum[i]);
		}
		output.append("\n");

		output.append(functionName.replace(',',' ') + " average");
		for (int i = 0 ; i < trialCount ; i++) {
			if (speedup)
				output.append("," + average[0] / average[i]);
			else
				output.append("," + average[i]);
		}
		output.append("\n");

		output.append(functionName.replace(',',' ') + " maximum");
		for (int i = 0 ; i < trialCount ; i++) {
			if (speedup)
				output.append("," + maximum[0] / maximum[i]);
			else
				output.append("," + maximum[i]);
		}
		output.append("\n");

		output.append(functionName.replace(',',' ') + " stddev");
		for (int i = 0 ; i < trialCount ; i++) {
			if (speedup)
				output.append("," + stddev[i]);
			else // is this right?
				output.append("," + stddev[i]);
		}
		return output.toString();
	}

	public double[] getMinimumSpeedup () {
		double[] data = new double[trialCount];
		for (int i = 0 ; i < trialCount ; i++) {
			data[i] = minimum[0] / minimum[i];
		}
		return data;
	}

	public double[] getAverageSpeedup () {
		double[] data = new double[trialCount];
		for (int i = 0 ; i < trialCount ; i++) {
			data[i] = average[0] / average[i];
		}
		return data;
	}

	public double[] getMaximumSpeedup () {
		double[] data = new double[trialCount];
		for (int i = 0 ; i < trialCount ; i++) {
			data[i] = maximum[0] / maximum[i];
		}
		return data;
	}
}
