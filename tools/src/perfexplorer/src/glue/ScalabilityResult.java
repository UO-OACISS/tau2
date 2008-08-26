/**
 * 
 */
package glue;

import java.math.*;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * @author khuck
 *
 */
public class ScalabilityResult extends AbstractResult {

	public enum Scaling {WEAK, STRONG};
	public enum Measure {SPEEDUP, EFFICIENCY};

	
	private double idealRatio = 1.0;
	private double actualRatio = 0.0;
	private String mainEvent = null;
	private String timeMetric = null;
	private Measure measure = Measure.SPEEDUP;
	private Scaling scaling = Scaling.STRONG;
		
	/**
	 * 
	 */
	public ScalabilityResult() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param input
	 */
	public ScalabilityResult(PerformanceResult input) {
		super(input);
		// TODO Auto-generated constructor stub
	}

	public ScalabilityResult(PerformanceResult input, boolean doFullCopy) {
		super(input, doFullCopy);
	}

	/**
	 * @return the idealRatio
	 */
	public double getIdealRatio() {
		return idealRatio;
	}

	/**
	 * @param idealRatio the idealRatio to set
	 */
	public void setIdealRatio(double idealRatio) {
		this.idealRatio = idealRatio;
	}

	/**
	 * @return the mainEvent
	 */
	public String getMainEvent() {
		return mainEvent;
	}

	/**
	 * @param mainEvent the mainEvent to set
	 */
	public void setMainEvent(String mainEvent) {
		this.mainEvent = mainEvent;
	}

	/**
	 * @return the timeMetric
	 */
	public String getTimeMetric() {
		return timeMetric;
	}

	/**
	 * @param timeMetric the timeMetric to set
	 */
	public void setTimeMetric(String timeMetric) {
		this.timeMetric = timeMetric;
	}

	/**
	 * @return the actualRatio
	 */
	public double getActualRatio() {
		return actualRatio;
	}

	/**
	 * @param actualRatio the actualRatio to set
	 */
	public void setActualRatio(double actualRatio) {
		this.actualRatio = actualRatio;
	}

	public String findPositiveReasons() {
		StringBuffer buf = new StringBuffer();
		for (String event : this.getEvents()) {
			for (String metric : this.getMetrics()) {
				double value = this.getExclusive(0, event, metric);
				if (!metric.equals(this.getTimeMetric())) {
					if (value > Math.max(this.actualRatio, this.idealRatio)) {
						buf.append(event + " " + metric + " " + value + '\n');
					}
				}
			}
		}
		return buf.toString();
	}

	public String findNegativeReasons() {
		StringBuffer buf = new StringBuffer();
		for (String event : this.getEvents()) {
			for (String metric : this.getMetrics()) {
				double value = this.getExclusive(0, event, metric);
				if (!metric.equals(this.getTimeMetric())) {
					if (value < Math.min(this.actualRatio, this.idealRatio)) {
						buf.append(event + " " + metric + " " + value + '\n');
					}
				}
			}
		}
		return buf.toString();
	}

}
