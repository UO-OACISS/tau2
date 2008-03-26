/**
 * 
 */
package glue;

import java.text.DecimalFormat;

import rules.RuleHarness;

/**
 * @author khuck
 *
 */
public class MeanEventFact {

	public static final int BETTER = 0;
	public static final int WORSE = 1;
	public static final int HIGHER = 2;
	public static final int LOWER = 3;
	private int betterWorse = BETTER;
	private String metric = null;
	private String meaningfulMetricName = null;
	private double mainValue = 0.0;
	private double eventValue = 0.0;
	private double severity = 0.0;
	private String eventName = null;
	
	/**
	 * 
	 */
	private MeanEventFact(int betterWorse, String metric, String meaningfulMetricName, double mainValue, double eventValue, String eventName, double severity) {
		this.betterWorse = betterWorse;
		this.metric = metric;
		this.meaningfulMetricName = meaningfulMetricName;
		this.mainValue = mainValue;
		this.eventValue = eventValue;
		this.eventName = eventName;
		this.severity = severity;
	}

	public static void compareEventToMain(PerformanceResult mainInput, String mainEvent, PerformanceResult eventInput, String event) {
		compareEventToMain(mainInput, mainEvent, eventInput, event, mainInput.getTimeMetric());
	}
	public static void compareEventToMain(PerformanceResult mainInput, String mainEvent, PerformanceResult eventInput, String event, String timeMetric) {
		// don't compare main to self
		if (mainEvent.equals(event)) {
			return;
		}
		
		double mainTime = mainInput.getInclusive(0, mainEvent, timeMetric);
		double eventTime = mainInput.getExclusive(0, event, timeMetric);
		double severity = eventTime / mainTime;
		//System.out.println(timeMetric + " " + mainTime + " " + eventTime + " " + severity);
		for (String metric : mainInput.getMetrics()) {
			double mainValue = mainInput.getInclusive(0, mainEvent, metric);
			double eventValue = eventInput.getExclusive(0, event, metric);
			if (metric.equals(DerivedMetrics.L1_HIT_RATE)) {
				// L1 cache hit rate
				if (mainValue > eventValue) {
					// this event has poor memory access
					RuleHarness.assertObject(new MeanEventFact(WORSE, metric, "L1 cache hit rate", mainValue, eventValue, event, severity));
				}
			} else if (metric.equals(DerivedMetrics.L2_HIT_RATE)) {
				// L2 cache hit rate
				if (mainValue > eventValue) {
					// this event has poor memory access
					RuleHarness.assertObject(new MeanEventFact(WORSE, metric, "L2 cache hit rate", mainValue, eventValue, event, severity));
				}
			} else if (metric.equals(DerivedMetrics.MFLOP_RATE)) {
				// FLOP rate
				if (mainValue < eventValue) {
					// this event has higher than average FLOP rate
					RuleHarness.assertObject(new MeanEventFact(BETTER, metric, "MFLOP/s", mainValue, eventValue, event, severity));
				}
			} else if (metric.equals(DerivedMetrics.L1_CACHE_HITS)) {
				// L1 cache hits
			} else if (metric.equals(DerivedMetrics.MEM_ACCESSES)) {
				// L1 cache access rate (aka memory accesses)
				if (mainValue > eventValue) {
					// this event has higher than average memory accesses
					RuleHarness.assertObject(new MeanEventFact(WORSE, metric, "L1 cache access rate", mainValue, eventValue, event, severity));
				}
			} else if (metric.equals(DerivedMetrics.L2_CACHE_HITS)) {
				// L2 cache hits
			} else if (metric.equals(DerivedMetrics.L2_ACCESSES)) {
				// L2 cache access rate
			} else if (metric.equals(DerivedMetrics.TOT_INS_RATE)) {
				// Total instruction rate
			} else { 
				// any other metric combination
				if (mainValue < eventValue) {
					RuleHarness.assertObject(new MeanEventFact(HIGHER, metric, metric, mainValue, eventValue, event, severity));
				} else if (mainValue < eventValue) {
					RuleHarness.assertObject(new MeanEventFact(LOWER, metric, metric, mainValue, eventValue, event, severity));
				}
			}
				
		}
	}
	
	public String toString () {
		// TODO: MAKE THIS PRETTY!
		StringBuffer buf = new StringBuffer();
		if (betterWorse == BETTER) {
			buf.append("Better ");
		} else if (betterWorse == WORSE) {
			buf.append("Worse ");
		} else if (betterWorse == HIGHER) {
			buf.append("Higher ");
		} else { // if (betterWorse == LOWER) {
			buf.append("Lower ");
		}
		//buf.append(meaningfulMetricName + " ");
		buf.append(mainValue + " ");
		buf.append(eventValue + " ");
		buf.append(eventName + " ");
		buf.append(metric + " ");
		buf.append(severity + " ");
		
		return buf.toString();
	}

	/**
	 * @return the betterWorse
	 */
	public int isBetterWorse() {
		return betterWorse;
	}

	/**
	 * @param betterWorse the betterWorse to set
	 */
	public void setBetterWorse(int betterWorse) {
		this.betterWorse = betterWorse;
	}

	/**
	 * @return the eventValue
	 */
	public double getEventValue() {
		return eventValue;
	}

	/**
	 * @param eventValue the eventValue to set
	 */
	public void setEventValue(double eventValue) {
		this.eventValue = eventValue;
	}

	/**
	 * @return the mainValue
	 */
	public double getMainValue() {
		return mainValue;
	}

	/**
	 * @param mainValue the mainValue to set
	 */
	public void setMainValue(double mainValue) {
		this.mainValue = mainValue;
	}

	/**
	 * @return the meaningfulMetricName
	 */
	public String getMeaningfulMetricName() {
		return meaningfulMetricName;
	}

	/**
	 * @param meaningfulMetricName the meaningfulMetricName to set
	 */
	public void setMeaningfulMetricName(String meaningfulMetricName) {
		this.meaningfulMetricName = meaningfulMetricName;
	}

	/**
	 * @return the metric
	 */
	public String getMetric() {
		return metric;
	}

	/**
	 * @param metric the metric to set
	 */
	public void setMetric(String metric) {
		this.metric = metric;
	}

	/**
	 * @return the severity
	 */
	public double getSeverity() {
		return severity;
	}

	/**
	 * @param severity the severity to set
	 */
	public void setSeverity(double severity) {
		this.severity = severity;
	}

	/**
	 * @return the eventName
	 */
	public String getEventName() {
		return eventName;
	}

	/**
	 * @param eventName the eventName to set
	 */
	public void setEventName(String eventName) {
		this.eventName = eventName;
	}
	
	public String getPercentage() {
		DecimalFormat format = new DecimalFormat("00.00%");
		String p = format.format(this.severity);
		return p;
	}
}
