/**
 * 
 */
package glue.psl;

import java.lang.Math;

/**
 * @author khuck
 *
 */
public abstract class Statistics {

	protected RegionSummaryIterator iterator = null;
	protected double min = Double.MAX_VALUE;
	protected double max = Double.MIN_VALUE;
	protected double avg = 0.0;
	protected double variance = 0.0;
	protected double stdDev = 0.0;
	protected double sum = 0.0;
	protected double squareSum = 0.0;
	protected int groupSize = 0;
	
	/**
	 * 
	 */
	public Statistics(RegionSummaryIterator iterator) {
		this.iterator = iterator;
		buildStats();
	}
	
	public double getMin() {
		return min;
	}
	
	public double getMax() {
		return max;
	}
	
	public double getAvg() {
		return avg;
	}
	
	public double getStdDev() {
		return stdDev;
	}
	
	public double getSum() {
		return sum;
	}
	
	public double getSquareSum() {
		return squareSum;
	}
	
	public int getGroupSize() {
		return groupSize;
	}
	
	protected abstract double getValue(RegionSummary summary);

	private void buildStats() {
		while (this.iterator.hasNext()) {
			RegionSummary summary = this.iterator.next();
			
			// get the value for this implementation
			double value = getValue(summary);
			
			// increment the count
			this.groupSize++;
			
			// do min and max
			if (value > this.max)
				this.max = value;
			if (value < this.min)
				this.min = value;
			
			// increment the sums
			this.sum += value;
			this.squareSum += Math.pow(value, 2.0);
		}
		// get the average
		this.avg = this.sum / this.groupSize;
		
		// reset the iterator to do variance and standard deviation
		this.iterator.reset();
		
		while (this.iterator.hasNext()) {
			RegionSummary summary = this.iterator.next();
			this.variance += Math.pow((this.avg-getValue(summary)),2.0);
		}
		this.variance = this.variance / (this.groupSize - 1);
		this.stdDev = Math.sqrt(this.variance);
	}


}
