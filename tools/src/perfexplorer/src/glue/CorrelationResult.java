/**
 * 
 */
package glue;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.drools.FactHandle;

import edu.uoregon.tau.perfdmf.Trial;

import rules.FactWrapper;
import rules.RuleHarness;
import rules.SelfAsserting;

/**
 * @author khuck
 *
 */
public class CorrelationResult extends DefaultResult implements SelfAsserting {

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
	public final static String NAME = glue.CorrelationResult.class.getName();
	private RuleHarness ruleHarness = null;
	
	public static final int CORRELATION = 0;
	public static final int SLOPE = 1;
	public static final int INTERCEPT = 2;

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
/*		this.ruleHarness = ruleHarness;
		for (String event : this.getEvents()) {
			for (String metric : this.getMetrics()) {
//				for (Integer type : CorrelationResult.getTypes()) {
				Integer type = CorrelationResult.CORRELATION;
					for (Integer thread : this.getThreads()) {
						double value = this.getDataPoint(thread, event, metric, type.intValue());
						String key = event + ":" + CorrelationResult.typeToString(thread) + ":" + metric + ":" + AbstractResult.typeToString(type);
						FactData data = new FactData(event, metric, type.intValue(), thread, value);
//						FactHandle handle = ruleHarness.assertObject(new FactWrapper(key, NAME, data));
						RuleHarness.assertObject(data);
//						assertedFacts.put(key, handle);
					}
//				}
			}
		}
*/	}
	
	public void assertFact(String event, String metric, int type, String event2, String metric2, int type2, double value) {
		String key = event + ":" + metric + ":" + event2 + ":" + metric2;
		FactData data = new FactData(event, metric, type, event2, metric2, type2, value);
//		FactHandle handle = ruleHarness.assertObject(new FactWrapper(key, NAME, data));
		RuleHarness.assertObject(data);
//		assertedFacts.put(key, handle);
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
		private double value;

		public FactData(String event, String metric, int type, Integer thread, double value) {
			this.event = event;
			this.metric = metric;
			this.thread = thread;
			this.type = type;
			this.value = value;
		}
		
		public FactData(String event, String metric, int type, String event2, String metric2, int type2, double value) {
			this.event = event;
			this.metric = metric;
			this.type = type;
			this.event2 = event2;
			this.metric2 = metric2;
			this.type2 = type2;
			this.value = value;
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
		public double getValue() {
			return value;
		}
		/**
		 * @param value the value to set
		 */
		public void setValue(double value) {
			this.value = value;
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
