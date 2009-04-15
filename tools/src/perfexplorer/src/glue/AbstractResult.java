/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.io.Serializable;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.Map;
import java.util.HashMap;
import java.util.TreeSet;

import edu.uoregon.tau.perfdmf.Trial;


/**
 * This class is used as an abstract implementation of the PerformanceResult
 * interface.  This class has all the member data fields for the plethora
 * of anticipated subclasses.
 * 
 * <P>CVS $Id: AbstractResult.java,v 1.17 2009/04/15 00:17:11 khuck Exp $</P>
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
	
	protected String mainEvent = null;
	protected double mainInclusive = 0.0;
	protected String mainMetric = null;
	protected Trial trial = null;
	protected Integer trialID = null;
	protected Map<Integer, String> eventMap = new HashMap<Integer, String>();
	protected String name = null;
	protected boolean ignoreWarnings = false;
	
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
		this.copyFields(input);
	}

	/**
	 * sort-of copy constructor
	 *
	 */
	public AbstractResult(PerformanceResult input, boolean notFullCopy) {
		this.copyFields(input);
	}
	
	private void copyFields(PerformanceResult input) {
		this.trial = input.getTrial();
		this.eventMap = input.getEventMap();
		this.mainEvent = input.getMainEvent();
		this.trial = input.getTrial();
		this.trialID = input.getTrialID();
		this.eventMap = input.getEventMap();
		this.name = input.getName();
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

		if (thread == 0) {
			//if (value > mainInclusive && !event.contains(" => ") && !event.startsWith(".TAU ") && (mainMetric == null || mainMetric.equals(metric))) {
			if (value > mainInclusive && !event.contains(" => ") && (mainMetric == null || mainMetric.equals(metric))) {
				mainInclusive = value;
				mainEvent = event;
				mainMetric = metric;
			}
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
			if (!ignoreWarnings)
				System.err.println("*** Warning - null inclusive value for thread: " + thread + ", event: " + event + ", metric: " + metric + " ***");
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
			if (!ignoreWarnings)
				System.err.println("*** Warning - null exclusive value for thread: " + thread + ", event: " + event + ", metric: " + metric + " ***");
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
			if (!ignoreWarnings)
				System.err.println("*** Warning - null calls value for thread: " + thread + ", event: " + event + " ***");
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
			if (!ignoreWarnings)
				System.err.println("*** Warning - null subroutine value for thread: " + thread + ", event: " + event + " ***");
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
			if (!ignoreWarnings)
				System.err.println("*** Warning - null userevent value for thread: " + thread + ", event: " + event + " ***");
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
			if (!ignoreWarnings)
				System.err.println("*** Warning - null userevent max value for thread: " + thread + ", event: " + event + " ***");
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
			if (!ignoreWarnings)
				System.err.println("*** Warning - null userevent min value for thread: " + thread + ", event: " + event + " ***");
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
			if (!ignoreWarnings)
				System.err.println("*** Warning - null userevent mean value for thread: " + thread + ", event: " + event + " ***");
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
			if (!ignoreWarnings)
				System.err.println("*** Warning - null userevent sumsqr value for thread: " + thread + ", event: " + event + " ***");
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

	public Set<String> getUserEvents(Integer thread) {
		Set<String> ues = null;
		try {
			ues = usereventData.get(thread).keySet();
		} catch (NullPointerException e) {
			if (!ignoreWarnings)
				System.err.println("*** Warning - null userevent set for thread: " + thread + " ***");
		}
		return ues;
	}
		
	/**
	 * @return the originalThreads
	 */
	public Integer getOriginalThreads() {
		if (this.trial != null) {
			// get the node, context, thread count for the trial
            int nodes = Integer.parseInt(this.trial.getField("node_count"));
            int contexts = Integer.parseInt(this.trial.getField("contexts_per_node"));
            int threads = Integer.parseInt(this.trial.getField("threads_per_context"));
			return (nodes*contexts*threads);
		}
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
	
	public String getTimeMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("TIME") && !metric.startsWith("("))
    			return metric;
    	}
    	if (!ignoreWarnings)
    		System.err.println("*** Warning - no time metric found in Trial ***");
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
    	if (!ignoreWarnings)
    		System.err.println("*** Warning - no floating point metric found in Trial ***");
    	return null;
	}

	public String getL1AccessMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L1_TCA") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L1_DCA") && !metric.startsWith("("))
    			return metric;
    	}
    	if (!ignoreWarnings)
    		System.err.println("*** Warning - no L1 access metric found in Trial ***");
    	return null;
	}

	public String getL2AccessMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L2_TCA") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L1_TCM") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L2_DCA") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L1_DCM") && !metric.startsWith("("))
    			return metric;
    	}
    	if (!ignoreWarnings)
    		System.err.println("*** Warning - no L2 access metric found in Trial ***");
    	return null;
	}

	public String getL3AccessMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L3_TCA") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L2_TCM") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L3_DCA") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L2_DCM") && !metric.startsWith("("))
    			return metric;
    	}
    	if (!ignoreWarnings)
    		System.err.println("*** Warning - no L3 access metric found in Trial ***");
    	return null;
	}

	public String getL1MissMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L1_TCM") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L1_DCM") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L2_TCA") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L2_DCA") && !metric.startsWith("("))
    			return metric;
    	}
    	if (!ignoreWarnings)
    		System.err.println("*** Warning - no L1 miss metric found in Trial ***");
    	return null;
	}

	public String getL2MissMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L2_TCM") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L2_DCM") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L3_TCA") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L3_DCA") && !metric.startsWith("("))
    			return metric;
    	}
    	if (!ignoreWarnings)
    		System.err.println("*** Warning - no L2 miss metric found in Trial ***");
    	return null;
	}

	public String getL3MissMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_L3_TCM") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_L3_DCM") && !metric.startsWith("("))
    			return metric;
    	}
    	if (!ignoreWarnings)
    		System.err.println("*** Warning - no L3 miss metric found in Trial ***");
    	return null;
	}

	public String getTLBMissMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_TLB_CM") && !metric.startsWith("("))
    			return metric;
    		if (metric.toUpperCase().contains("PAPI_TLB_DM") && !metric.startsWith("("))
    			return metric;
    	}
    	if (!ignoreWarnings)
    		System.err.println("*** Warning - no TLB miss metric found in Trial ***");
    	return null;
	}

	public String getTotalInstructionMetric() {
    	for (String metric : metrics) {
    		if (metric.toUpperCase().contains("PAPI_TOT_INS") && !metric.startsWith("("))
    			return metric;
    	}
    	if (!ignoreWarnings)
    		System.err.println("*** Warning - no total instruction metric found in Trial ***");
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

	/**
	 * @return the name
	 */
	public String getName() {
		return name;
	}

	/**
	 * @param name the name to set
	 */
	public void setName(String name) {
		this.name = name;
	}
	
	public String toString() {
		StringBuilder buf = new StringBuilder();
		
		for (Integer thread : this.getThreads()) {
			for (String event : this.getEvents()) {
				for (String metric : this.getMetrics()) {
					buf.append(thread + " : " + event + " : " + metric + " : " + this.getExclusive(thread, event, metric) + "\n");
				}
			}
		}
		return buf.toString();
	}

	/**
	 * @return the eventMap
	 */
	public Map<Integer, String> getEventMap() {
		return eventMap;
	}

	/**
	 * @param eventMap the eventMap to set
	 */
	public void setEventMap(Map<Integer, String> eventMap) {
		this.eventMap = eventMap;
	}

	/**
	 * @return the mainInclusive
	 */
	public double getMainInclusive() {
		return mainInclusive;
	}

	/**
	 * @param mainInclusive the mainInclusive to set
	 */
	public void setMainInclusive(double mainInclusive) {
		this.mainInclusive = mainInclusive;
	}
	
	public void updateEventMap() {
		// remove from the event map the events which are no longer here
		Set<Integer> keys = this.eventMap.keySet();
		Map<Integer,String> newMap = new HashMap<Integer,String>();
		for (Integer key : keys) {
			String event = this.eventMap.get(key);
			if (this.events.contains(event)) {
				newMap.put(key,event);
			}
		}
		this.eventMap = newMap;
	}

	public void setIgnoreWarnings(boolean ignore) {
		this.ignoreWarnings = ignore;
	}
}
