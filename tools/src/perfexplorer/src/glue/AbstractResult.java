/**
 * 
 */
package glue;

import java.io.Serializable;
import java.util.ListIterator;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.Map;
import java.util.HashMap;
import java.util.TreeSet;

import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;


/**
 * This class is used as an abstract implementation of the PerformanceResult
 * interface.  This class has all the member data fields for the plethora
 * of anticipated subclasses.
 * 
 * <P>CVS $Id: AbstractResult.java,v 1.7 2008/07/29 23:40:18 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0
 */
public abstract class AbstractResult implements PerformanceResult, Serializable {
	protected Set<Integer> threads = new TreeSet<Integer>();
	protected Set<String> events = new TreeSet<String>();
	protected Set<String> metrics = new TreeSet<String>();
	protected Set<String> userEvents = new TreeSet<String>();
	
	protected Map<Integer, Map<String, Map<String, Double>>> inclusiveData = 
			new HashMap<Integer, Map<String, Map<String, Double>>>();

	protected Map<Integer, Map<String, Map<String, Double>>> exclusiveData = 
		new HashMap<Integer, Map<String, Map<String, Double>>>();

	protected Map<Integer, Map<String, Double>> callData = 
		new HashMap<Integer, Map<String, Double>>();

	protected Map<Integer, Map<String, Double>> subroutineData = 
		new HashMap<Integer, Map<String, Double>>();

	protected Map<Integer, Map<String, Double[]>> usereventData = 
		new HashMap<Integer, Map<String, Double[]>>();

	private String mainEvent = null;
	private double mainInclusive = 0.0;
	private String mainMetric = null;
	
	public static final int INCLUSIVE = 0;
	public static final int EXCLUSIVE = 1;
	public static final int CALLS = 2;
	public static final int SUBROUTINES = 3;
	public static final int USEREVENT_NUMEVENTS = 4;
	public static final int USEREVENT_MAX = 5;
	public static final int USEREVENT_MIN = 6;
	public static final int USEREVENT_MEAN = 7;
	public static final int USEREVENT_SUMSQR = 8;
	private static List<Integer> types = null;
	
	protected Trial trial = null;
	protected Integer trialID = null;
	
	public static List<Integer> getTypes() {
		if (types == null) {
			types = new ArrayList<Integer>();
			types.add(AbstractResult.INCLUSIVE);
			types.add(AbstractResult.EXCLUSIVE);
			types.add(AbstractResult.CALLS);
			types.add(AbstractResult.SUBROUTINES);
			types.add(AbstractResult.USEREVENT_NUMEVENTS);
			types.add(AbstractResult.USEREVENT_MAX);
			types.add(AbstractResult.USEREVENT_MIN);
			types.add(AbstractResult.USEREVENT_MEAN);
			types.add(AbstractResult.USEREVENT_SUMSQR);
		}
		return types;
	}
	
	public static String typeToString(int type) {
		switch (type) {
		case INCLUSIVE:
			return "INCLUSIVE";
		case EXCLUSIVE:
			return "EXCLUSIVE";
		case CALLS:
			return "CALLS";
		case SUBROUTINES:
			return "SUBROUTINES";
		case USEREVENT_NUMEVENTS:
			return "USEREVENT_NUMEVENTS";
		case USEREVENT_MAX:
			return "USEREVENT_MAX";
		case USEREVENT_MIN:
			return "USEREVENT_MIN";
		case USEREVENT_MEAN:
			return "USEREVENT_MEAN";
		case USEREVENT_SUMSQR:
			return "USEREVENT_SUMSQR";
		}
		return "";
	}
	
	/**
	 * Private constructor, can't be instantiated
	 *
	 */
	protected AbstractResult() {
	}
	
	/**
	 * Constructor which takes a trial to have a back pointer to the database
	 * 
	 */
	public AbstractResult(Trial trial) {
		this.trial = trial;
	}
	
	/**
	 * copy constructor
	 *
	 */
	public AbstractResult(PerformanceResult input) {
		for (Integer thread : input.getThreads()) {
			for (String event : input.getEvents()) {
				for (String metric : input.getMetrics()) {
					putExclusive(thread, event, metric, 
							input.getExclusive(thread, event, metric));
					putInclusive(thread, event, metric, 
							input.getInclusive(thread, event, metric));
				}
				putCalls(thread, event, input.getCalls(thread, event));
				putSubroutines(thread, event, input.getSubroutines(thread, event));
			}
			for (String event : input.getUserEvents()) {
				putDataPoint(thread, event, null, USEREVENT_NUMEVENTS, 
						getDataPoint(thread, event, null, USEREVENT_NUMEVENTS));
				putDataPoint(thread, event, null, USEREVENT_MAX, 
						getDataPoint(thread, event, null, USEREVENT_MAX));
				putDataPoint(thread, event, null, USEREVENT_MIN, 
						getDataPoint(thread, event, null, USEREVENT_MIN));
				putDataPoint(thread, event, null, USEREVENT_MEAN, 
						getDataPoint(thread, event, null, USEREVENT_MEAN));
				putDataPoint(thread, event, null, USEREVENT_SUMSQR, 
						getDataPoint(thread, event, null, USEREVENT_SUMSQR));
			}
		}
	}

	public void putInclusive(Integer thread, String event, String metric, double value) {
		if (!threads.contains(thread)) {
			threads.add(thread);
		}
		if (!events.contains(event)) {
			events.add(event);
		}
		if (!metrics.contains(metric)) {
			metrics.add(metric);
		}

		if (!inclusiveData.containsKey(thread)) {
			inclusiveData.put(thread, new HashMap<String, Map<String, Double>>());
		}
		if (!inclusiveData.get(thread).containsKey(event)) {
			inclusiveData.get(thread).put(event, new HashMap<String, Double>());
		}
		inclusiveData.get(thread).get(event).put(metric, value);
		if (value > mainInclusive && !event.contains(" => ") && !event.startsWith(".TAU ") && (mainMetric == null || mainMetric.equals(metric))) {
//			if (this instanceof MeanResult && mainEvent != null) {
//				System.out.println("Old: " + mainEvent + ":" + metric + " " + mainInclusive);
//			}
			mainInclusive = value;
			mainEvent = event;
			mainMetric = metric;
//			if (this instanceof MeanResult) {
//				System.out.println("New: " + mainEvent + ":" + metric + " " + mainInclusive);
//			}
//		} else if (this instanceof MeanResult && event.equals("main")) {
//			System.out.println("MAIN: " + event + ":" + metric + " " + value);
		}
			

	}
	
	public void putExclusive(Integer thread, String event, String metric, double value) {
		if (!threads.contains(thread)) {
			threads.add(thread);
		}
		if (!events.contains(event)) {
			events.add(event);
		}
		if (!metrics.contains(metric)) {
			metrics.add(metric);
		}

		if (!exclusiveData.containsKey(thread)) {
			exclusiveData.put(thread, new HashMap<String, Map<String, Double>>());
		}
		if (!exclusiveData.get(thread).containsKey(event)) {
			exclusiveData.get(thread).put(event, new HashMap<String, Double>());
		}
		exclusiveData.get(thread).get(event).put(metric, value);
	}

	public void putCalls(Integer thread, String event, double value) {
		if (!threads.contains(thread)) {
			threads.add(thread);
		}
		if (!events.contains(event)) {
			events.add(event);
		}

		if (!callData.containsKey(thread)) {
			callData.put(thread, new HashMap<String, Double>());
		}
		callData.get(thread).put(event, value);
	}

	public void putSubroutines(Integer thread, String event, double value) {
		if (!threads.contains(thread)) {
			threads.add(thread);
		}
		if (!events.contains(event)) {
			events.add(event);
		}

		if (!subroutineData.containsKey(thread)) {
			subroutineData.put(thread, new HashMap<String, Double>());
		}
		subroutineData.get(thread).put(event, value);
	}

	public void putUsereventNumevents(Integer thread, String event, double value) {
		if (!threads.contains(thread)) {
			threads.add(thread);
		}
		if (!userEvents.contains(event)) {
			userEvents.add(event);
		}

		if (!usereventData.containsKey(thread)) {
			usereventData.put(thread, new HashMap<String, Double[]>());
		}
		Double[] data = usereventData.get(thread).get(event);
		if (data == null) {
			data = new Double[5];
		}
		data[USEREVENT_NUMEVENTS - USEREVENT_NUMEVENTS] = value;
		usereventData.get(thread).put(event, data);
	}

	public void putUsereventMax(Integer thread, String event, double value) {
		if (!threads.contains(thread)) {
			threads.add(thread);
		}
		if (!userEvents.contains(event)) {
			userEvents.add(event);
		}

		if (!usereventData.containsKey(thread)) {
			usereventData.put(thread, new HashMap<String, Double[]>());
		}
		Double[] data = usereventData.get(thread).get(event);
		if (data == null) {
			data = new Double[5];
		}
		data[USEREVENT_MAX - USEREVENT_NUMEVENTS] = value;
		usereventData.get(thread).put(event, data);
	}

	public void putUsereventMin(Integer thread, String event, double value) {
		if (!threads.contains(thread)) {
			threads.add(thread);
		}
		if (!userEvents.contains(event)) {
			userEvents.add(event);
		}

		if (!usereventData.containsKey(thread)) {
			usereventData.put(thread, new HashMap<String, Double[]>());
		}
		Double[] data = usereventData.get(thread).get(event);
		if (data == null) {
			data = new Double[5];
		}
		data[USEREVENT_MIN - USEREVENT_NUMEVENTS] = value;
		usereventData.get(thread).put(event, data);
	}

	public void putUsereventMean(Integer thread, String event, double value) {
		if (!threads.contains(thread)) {
			threads.add(thread);
		}
		if (!userEvents.contains(event)) {
			userEvents.add(event);
		}

		if (!usereventData.containsKey(thread)) {
			usereventData.put(thread, new HashMap<String, Double[]>());
		}
		Double[] data = usereventData.get(thread).get(event);
		if (data == null) {
			data = new Double[5];
		}
		data[USEREVENT_MEAN - USEREVENT_NUMEVENTS] = value;
		usereventData.get(thread).put(event, data);
	}

	public void putUsereventSumsqr(Integer thread, String event, double value) {
		if (!threads.contains(thread)) {
			threads.add(thread);
		}
		if (!userEvents.contains(event)) {
			userEvents.add(event);
		}

		if (!usereventData.containsKey(thread)) {
			usereventData.put(thread, new HashMap<String, Double[]>());
		}
		Double[] data = usereventData.get(thread).get(event);
		if (data == null) {
			data = new Double[5];
		}
		data[USEREVENT_SUMSQR - USEREVENT_NUMEVENTS] = value;
		usereventData.get(thread).put(event, data);
	}

	public double getInclusive(Integer thread, String event, String metric) {
		Double value = null;
		try {
			value = inclusiveData.get(thread).get(event).get(metric);
			if (value != null) {
				return (value.doubleValue());
			}
		} catch (NullPointerException e) {
			return 0.0;
		}
		return 0.0;
	}
	
	public double getExclusive(Integer thread, String event, String metric) {
		Double value = null;
		try {
			value = exclusiveData.get(thread).get(event).get(metric);
			if (value != null) {
				return (value.doubleValue());
			}
		} catch (NullPointerException e) {
			return 0.0;
		}
		return 0.0;
	}

	public double getCalls(Integer thread, String event) {
		Double value = null;
		try {
			value = callData.get(thread).get(event);
			if (value != null) {
				return (value.doubleValue());
			}
		} catch (NullPointerException e) {
			return 0.0;
		}
		return 0.0;
	}

	public double getSubroutines(Integer thread, String event) {
		Double value = null;
		try {
			value = callData.get(thread).get(event);
			if (value != null) {
				return (value.doubleValue());
			}
		} catch (NullPointerException e) {
			return 0.0;
		}
		return 0.0;
	}

	public double getUsereventNumevents(Integer thread, String event) {
		Double[] value = null;
		try {
			value = usereventData.get(thread).get(event);
			if (value != null) {
				return (value[USEREVENT_NUMEVENTS - USEREVENT_NUMEVENTS].doubleValue());
			}
		} catch (NullPointerException e) {
			return 0.0;
		}
		return 0.0;
	}

	public double getUsereventMax(Integer thread, String event) {
		Double[] value = null;
		try {
			value = usereventData.get(thread).get(event);
			if (value != null) {
				return (value[USEREVENT_MAX - USEREVENT_NUMEVENTS].doubleValue());
			}
		} catch (NullPointerException e) {
			return 0.0;
		}
		return 0.0;
	}

	public double getUsereventMin(Integer thread, String event) {
		Double[] value = null;
		try {
			value = usereventData.get(thread).get(event);
			if (value != null) {
				return (value[USEREVENT_MIN - USEREVENT_NUMEVENTS].doubleValue());
			}
		} catch (NullPointerException e) {
			return 0.0;
		}
		return 0.0;
	}

	public double getUsereventMean(Integer thread, String event) {
		Double[] value = null;
		try {
			value = usereventData.get(thread).get(event);
			if (value != null) {
				return (value[USEREVENT_MEAN - USEREVENT_NUMEVENTS].doubleValue());
			}
		} catch (NullPointerException e) {
			return 0.0;
		}
		return 0.0;
	}

	public double getUsereventSumsqr(Integer thread, String event) {
		Double[] value = null;
		try {
			value = usereventData.get(thread).get(event);
			if (value != null) {
				return (value[USEREVENT_SUMSQR - USEREVENT_NUMEVENTS].doubleValue());
			}
		} catch (NullPointerException e) {
			return 0.0;
		}
		return 0.0;
	}

	public Set<String> getEvents() {
		return events;
	}

	public Set<String> getMetrics() {
		return metrics;
	}

	public Set<Integer> getThreads() {
		return threads;
	}
	
	public String getMainEvent() {
		return mainEvent;
	}
	
	public Set<String> getUserEvents() {
		return userEvents;
	}
	
	/**
	 * @return the originalThreads
	 */
	public Integer getOriginalThreads() {
		return this.threads.size();
	}

	/**
	 * @param originalThreads the originalThreads to set
	 */
	public void setOriginalThreads(Integer originalThreads) {
		// meaningless.  See TrialMeanResult or TrialTotalResult for details.
	}
	
	public double getDataPoint(Integer thread, String event, String metric, int type) {
		switch (type) {
		case INCLUSIVE:
			return getInclusive(thread, event, metric);
		case EXCLUSIVE:
			return getExclusive(thread, event, metric);
		case CALLS:
			return getCalls(thread, event);
		case SUBROUTINES:
			return getSubroutines(thread, event);
		case USEREVENT_NUMEVENTS:
			return getUsereventNumevents(thread, event);
		case USEREVENT_MAX:
			return getUsereventMax(thread, event);
		case USEREVENT_MIN:
			return getUsereventMin(thread, event);
		case USEREVENT_MEAN:
			return getUsereventMean(thread, event);
		case USEREVENT_SUMSQR:
			return getUsereventSumsqr(thread, event);
		}		
		return 0.0;
	}
	
	public void putDataPoint(Integer thread, String event, String metric, int type, double value) {
		switch (type) {
		case INCLUSIVE:
			this.putInclusive(thread, event, metric, value);
		case EXCLUSIVE:
			this.putExclusive(thread, event, metric, value);
		case CALLS:
			this.putCalls(thread, event, value);
		case SUBROUTINES:
			this.putSubroutines(thread, event, value);
		case USEREVENT_NUMEVENTS:
			this.putUsereventNumevents(thread, event, value);
		case USEREVENT_MAX:
			this.putUsereventMax(thread, event, value);
		case USEREVENT_MIN:
			this.putUsereventMin(thread, event, value);
		case USEREVENT_MEAN:
			this.putUsereventMean(thread, event, value);
		case USEREVENT_SUMSQR:
			this.putUsereventSumsqr(thread, event, value);
		}
	}
	
	public String toString() {
		return this.getClass().getName();
	}
	
	public String getTimeMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("TIME") && !metric.startsWith("("))
    			return metric;
    	}
		// if time not found, use something similar
    	for (String metric : metrics) {
    		if (metric.toUpperCase().equals("PAPI_TOT_CYC"))
    			return metric;
    		else if (metric.toUpperCase().equals("CPU_CYCLES"))
    			return metric;
    	}
    	return null;
	}
	
	public String getFPMetric() {
    	for (String metric : metrics) {
    		// get either FP_OPS or FP_INS
    		if (metric.toUpperCase().contains("PAPI_FP_") && !metric.startsWith("("))
    			return metric;
    	}
    	return null;
	}

	public String getL1AccessMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L1_TCA") && !metric.startsWith("("))
    			return metric;
    	}
    	return null;
	}

	public String getL2AccessMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L2_TCA") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L1_TCM") && !metric.startsWith("("))
    			return metric;
    	}
    	return null;
	}

	public String getL3AccessMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L3_TCA") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L2_TCM") && !metric.startsWith("("))
    			return metric;
    	}
    	return null;
	}

	public String getL1MissMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L1_TCM") && !metric.startsWith("("))
    			return metric;
    	}
    	return null;
	}

	public String getL2MissMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L2_TCM") && !metric.startsWith("("))
    			return metric;
    	}
    	return null;
	}

	public String getL3MissMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L3_TCM") && !metric.startsWith("("))
    			return metric;
    	}
    	return null;
	}

	public String getTotalInstructionMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_TOT_INS") && !metric.startsWith("("))
    			return metric;
    	}
    	return null;
	}

	/* (non-Javadoc)
	 * @see glue.PerformanceResult#getSortedByValue(java.lang.String, int)
	 */
	public Map<String, Double> getSortedByValue(String metric, int type, boolean ascending) {
		Map<String, Double> sorted = new HashMap<String, Double>();
		for (String event : this.getEvents()) {
			sorted.put(event, this.getDataPoint(0, event, metric, type));
		}
		sorted = Utilities.sortHashMapByValues(sorted, ascending);
		return sorted;
	}

	public Integer getTrialID() {
		return trialID;
	}

	public void setTrialID(Integer trialID) {
		this.trialID = trialID;
	}

	public Trial getTrial() {
		return trial;
	}

	public void setTrial(Trial trial) {
		this.trial = trial;
	}
	
}
