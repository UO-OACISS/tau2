/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.drools.FactHandle;

import edu.uoregon.tau.perfexplorer.rules.RuleHarness;
import edu.uoregon.tau.perfexplorer.rules.SelfAsserting;


/**
 * @author khuck
 *
 */
public class CorrelationResult extends DefaultResult implements SelfAsserting {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7362707779938777066L;
	private String baselineMetric = null;
	private String comparisonMetric = null;
	private String baselineEvent = null;
	private String comparisonEvent = null;
	private int baselineType = AbstractResult.EXCLUSIVE;
	private int comparisonType = AbstractResult.EXCLUSIVE;
	private Double intercept = 0.0;
	private Double slope = 0.0;
	private Double correlation = 0.0;
	Hashtable<String,FactHandle> assertedFacts = new Hashtable<String,FactHandle>();
	public final static String NAME = edu.uoregon.tau.perfexplorer.glue.CorrelationResult.class.getName();
	//private RuleHarness ruleHarness = null;
	
	public static final int CORRELATION = AbstractResult.USEREVENT_SUMSQR + 1;
	public static final int SLOPE = CORRELATION + 1;
	public static final int INTERCEPT = CORRELATION + 2;
	public static final int METADATA = CORRELATION + 3;

	private static List<Integer> types = null;
	
	public static List<Integer> getTypes() {
		if (types == null) {
			types = new ArrayList<Integer>();
			types.add(CORRELATION);
			types.add(SLOPE);
			types.add(INTERCEPT);
		}
		return types;
	}
	
	public static String typeToString(int type) {
		switch (type) {
		case CORRELATION:
			return "CORRELATION";
		case SLOPE:
			return "SLOPE";
		case INTERCEPT:
			return "INTERCEPT";
		}
		return "";
	}

	/**
	 * 
	 */
	public CorrelationResult() {
		super();
		// TODO Auto-generated constructor stub
	}

	/**
	 * 
	 */
	public CorrelationResult(String baselineMetric, String comparisonMetric, 
			String baselineEvent, String comparisonEvent, 
			int baselineType, int comparisonType, Double correlation) {
		this.baselineMetric = baselineMetric;
		this.comparisonMetric = comparisonMetric;
		this.baselineEvent = baselineEvent;
		this.comparisonEvent = comparisonEvent;
		this.baselineType = baselineType;
		this.comparisonType = comparisonType;
		this.correlation = correlation;
	}

	public CorrelationResult(PerformanceResult input, boolean notFullCopy) {
		super(input, notFullCopy);
	}

	public String getBaselineEvent() {
		return baselineEvent;
	}

	public void setBaselineEvent(String baselineEvent) {
		this.baselineEvent = baselineEvent;
	}

	public String getBaselineMetric() {
		return baselineMetric;
	}

	public void setBaselineMetric(String baselineMetric) {
		this.baselineMetric = baselineMetric;
	}

	public String getComparisonEvent() {
		return comparisonEvent;
	}

	public void setComparisonEvent(String comparisonEvent) {
		this.comparisonEvent = comparisonEvent;
	}

	public String getComparisonMetric() {
		return comparisonMetric;
	}

	public void setComparisonMetric(String comparisonMetric) {
		this.comparisonMetric = comparisonMetric;
	}

	public Double getIntercept() {
		return intercept;
	}

	public void setIntercept(Double intercept) {
		this.intercept = intercept;
	}

	public Double getSlope() {
		return slope;
	}

	public void setSlope(Double slope) {
		this.slope = slope;
	}

	public Double getCorrelation() {
		return correlation;
	}

	public void setCorrelation(Double correlation) {
		this.correlation = correlation;
	}

	public int getBaselineType() {
		return baselineType;
	}

	public void setBaselineType(int type) {
		this.baselineType = type;
	}

	public int getComparisonType() {
		return comparisonType;
	}

	public void setComparisonType(int comparisonType) {
		this.comparisonType = comparisonType;
	}

	public void assertFacts() {
		if (RuleHarness.getInstance() != null);
			assertFacts(RuleHarness.getInstance());
	}
	
	public void assertFacts(RuleHarness ruleHarness) {
		// do nothing
	}
	
	public void assertFact(String event, String metric, int type, String event2, String metric2, int type2, double correlation, double slope, double intercept) {
		//String key = event + ":" + metric + ":" + event2 + ":" + metric2;
		FactData data = new FactData(event, metric, type, event2, metric2, type2, correlation, slope, intercept);
		RuleHarness.assertObject(data);
	}

	public void removeFact(String factName) {
		FactHandle handle = assertedFacts.get(factName);
		if (handle == null) {
			System.err.println("HANDLE NOT FOUND for " + factName + ", " + NAME);
		} else {
			RuleHarness.retractObject(handle);
		}
	}

	public class FactData {
		private String event;
		private String metric;
		private String event2;
		private String metric2;
		private Integer thread;
		private int type;
		private int type2;
		private double correlation;
		private double slope;
		private double intercept;

		public FactData(String event, String metric, int type, String event2, String metric2, int type2, double correlation, double slope, double intercept) {
			this.event = event;
			this.metric = metric;
			this.type = type;
			this.event2 = event2;
			this.metric2 = metric2;
			this.type2 = type2;
			this.correlation = correlation;
			this.slope = slope;
			this.intercept = intercept;
		}

		/**
		 * @return the event
		 */
		public String getEvent() {
			return event;
		}
		/**
		 * @param event the event to set
		 */
		public void setEvent(String event) {
			this.event = event;
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
		 * @return the thread
		 */
		public Integer getThread() {
			return thread;
		}
		/**
		 * @param thread the thread to set
		 */
		public void setThread(Integer thread) {
			this.thread = thread;
		}
		/**
		 * @return the type
		 */
		public int getType() {
			return this.type;
		}
		/**
		 * @param type the type to set
		 */
		public void setType(int type) {
			this.type = type;
		}
		/**
		 * @return the value
		 */
		public double getCorrelation() {
			return correlation;
		}
		/**
		 * @return the value
		 */
		public double getSlope() {
			return slope;
		}
		/**
		 * @return the value
		 */
		public double getIntercept() {
			return intercept;
		}
		/**
		 * @return the event2
		 */
		public String getEvent2() {
			return event2;
		}

		/**
		 * @param event2 the event2 to set
		 */
		public void setEvent2(String event2) {
			this.event2 = event2;
		}

		/**
		 * @return the metric2
		 */
		public String getMetric2() {
			return metric2;
		}

		/**
		 * @param metric2 the metric2 to set
		 */
		public void setMetric2(String metric2) {
			this.metric2 = metric2;
		}

		/**
		 * @return the type2
		 */
		public int getType2() {
			return type2;
		}

		/**
		 * @param type2 the type2 to set
		 */
		public void setType2(int type2) {
			this.type2 = type2;
		}
	}

}
