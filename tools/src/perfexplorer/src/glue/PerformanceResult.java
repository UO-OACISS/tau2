/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.util.Map;
import java.util.Set;

import edu.uoregon.tau.perfdmf.Trial;

/**
 * <p>
 * This interface is defined as the methods all performance results
 * should support.  All operations should be refered to through
 * this interface, whenever possible.
 * </p>
 * 
 * <P>CVS $Id: PerformanceResult.java,v 1.10 2009/04/09 00:23:51 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 2.0
 * @since   2.0
 */
public interface PerformanceResult {

	/**
	 * This method will return the name of the event which has the highest
	 * inclusive time value in the trial.
	 * 
	 * @return the name of the main event
	 */
	public String getMainEvent();

	/**
	 * This method will return a Set of Integers, which represent the
	 * identifiers of the threads of execution in the trial.
	 * 
	 * @return the set of thread identifiers
	 */
	public Set<Integer> getThreads();

	/**
	 * This method will return a Set of Strings, which represent the
	 * names of the events in the trial.
	 * 
	 * @return the set of event names
	 */
	public Set<String> getEvents();

	/**
	 * This method will return a Set of Strings, which represent the 
	 * names of the metrics in the trial.
	 * 
	 * @return the set of metric names
	 */
	public Set<String> getMetrics();

	/**
	 * This method will return a Set of Strings, which represent the
	 * names of the userevents in the trial.
	 * 
	 * @return the set of userevent names
	 */
	public Set<String> getUserEvents();

	/**
	 * This method will return the inclusive value stored in the trial for
	 * the selected thread, event, metric combination.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param metric The metric of interest
	 * @return the inclusive value
	 */
	public double getInclusive(Integer thread, String event, String metric);

	/**
	 * This method will return the exclusive value stored in the trial for
	 * the selected thread, event, metric combination.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param metric The metric of interest
	 * @return the exclusive value
	 */
	public double getExclusive(Integer thread, String event, String metric);

	/**
	 * This method will return the number of times that the specified event
	 * was called on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @return the number of calls
	 */
	public double getCalls(Integer thread, String event);

	/**
	 * This method will return the number of subroutines that the specified event
	 * had on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @return the number of subroutines
	 */
	public double getSubroutines(Integer thread, String event);
	
	/**
	 * This method will return the number of times that a specified user event
	 * happened on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @return the number of events
	 */
	public double getUsereventNumevents(Integer thread, String event);

	/**
	 * This method will return the maximum value for the specified user event
	 * which was observed on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @return the maximum value
	 */
	public double getUsereventMax(Integer thread, String event);

	/**
	 * This method will return the minimum value for the specified user event
	 * which was observed on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @return the minimum value
	 */
	public double getUsereventMin(Integer thread, String event);

	/**
	 * This method will return the mean value for the specified user event
	 * which was observed on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @return the mean value
	 */
	public double getUsereventMean(Integer thread, String event);

	/**
	 * This method will return the sum of squared values for the specified user
	 * event which was observed on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @return the sum of squared values
	 */
	public double getUsereventSumsqr(Integer thread, String event);

	/**
	 * This method will save the specified value as the inclusive value for the
	 * specified thread, event, metric combination.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param metric The metric of interest
	 * @param value The value measured on that thread, event, metric combination
	 */
	public void putInclusive(Integer thread, String event, String metric, double value);
	
	/**
	 * This method will save the specified value as the exclusive value for the
	 * specified thread, event, metric combination.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param metric The metric of interest
	 * @param value The value measured on that thread, event, metric combination
	 */
	public void putExclusive(Integer thread, String event, String metric, double value);
	
	/**
	 * This method will save the specified value as the number of calls for the
	 * specified event on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param value The value measured on that thread, event combination
	 */
	public void putCalls(Integer thread, String event, double value);
	
	/**
	 * This method will save the specified value as the number of subroutines for the
	 * specified event on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param value The value measured on that thread, event combination
	 */
	public void putSubroutines(Integer thread, String event, double value);
	
	/**
	 * This method will save the number of times that a specified user event
	 * happened on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param value The number of events
	 */
	public void putUsereventNumevents(Integer thread, String event, double value);
	
	/**
	 * This method will save the maximum value for a specified user event which
	 * was observed on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param value The maximum value
	 */
	public void putUsereventMax(Integer thread, String event, double value);

	/**
	 * This method will save the minimum value for a specified user event which
	 * was observed on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param value The minimum value
	 */
	public void putUsereventMin(Integer thread, String event, double value);

	/**
	 * This method will save the mean value for a specified user event which
	 * was observed on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param value The mean value
	 */
	public void putUsereventMean(Integer thread, String event, double value);

	/**
	 * This method will save the sum of squared values for the specified user
	 * event which was observed on the specified thread of execution.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param value The sum of squared values
	 */
	public void putUsereventSumsqr(Integer thread, String event, double value);
	
	/**
	 * This method will return the number of threads in the trial from which this data
	 * was derived.
	 * 
	 * @return the number of threads in the original trial
	 */
	public Integer getOriginalThreads();
	
	/**
	 * This method will return the value stored in the trial for the specified thread,
	 * event, metric, type combination.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param metric The metric of interest
	 * @param type The type of data to return
	 * @return the value The value of that type measured on that thread, event,
	 * metric combination
	 * @see #getExclusive
	 * @see #getInclusive
	 * @see #getCalls
	 * @see #getSubroutines
	 * @see AbstractResult#INCLUSIVE
	 * @see AbstractResult#EXCLUSIVE
	 * @see AbstractResult#CALLS
	 * @see AbstractResult#SUBROUTINES
	 * @see AbstractResult#USEREVENT_NUMEVENTS
	 * @see AbstractResult#USEREVENT_MAX
	 * @see AbstractResult#USEREVENT_MIN
	 * @see AbstractResult#USEREVENT_MEAN
	 * @see AbstractResult#USEREVENT_SUMSQR
	 */
	public double getDataPoint(Integer thread, String event, String metric, int type);

	/**
	 * This method will store the specified value in the trial for the specified thread,
	 * event, metric, type combination.
	 * 
	 * @param thread The thread of interest
	 * @param event The event of interest
	 * @param metric The metric of interest
	 * @param type The type of data to return 
	 * @param value The value The value of that type measured on that thread,
	 * event, metric combination
	 * @see #putExclusive
	 * @see #putInclusive
	 * @see #putCalls
	 * @see #putSubroutines
	 * @see AbstractResult#INCLUSIVE
	 * @see AbstractResult#EXCLUSIVE
	 * @see AbstractResult#CALLS
	 * @see AbstractResult#SUBROUTINES
	 * @see AbstractResult#USEREVENT_NUMEVENTS
	 * @see AbstractResult#USEREVENT_MAX
	 * @see AbstractResult#USEREVENT_MIN
	 * @see AbstractResult#USEREVENT_MEAN
	 * @see AbstractResult#USEREVENT_SUMSQR
	 */
	public void putDataPoint(Integer thread, String event, String metric, int type, double value);
	
	/** 
	 * This method will return a string representation of this PerformanceResult.
	 * @return a printable string
	 */
	public String toString();

	/**
	 * This method will return the metric which represents the time metric in the trial.
	 * @return the metric name
	 */
	public String getTimeMetric();

	/**
	 * This method will return a Map of values, sorted by the values.  The keys to the map
	 * are the event strings in the trial.
	 * 
	 * @param metric The metric of interest
	 * @param type The type of data
	 * @param ascending Either ascending (true) or descending (false) order
	 * @return the Map of values
	 * @see AbstractResult#INCLUSIVE
	 * @see AbstractResult#EXCLUSIVE
	 * @see AbstractResult#CALLS
	 * @see AbstractResult#SUBROUTINES
	 * @see AbstractResult#USEREVENT_NUMEVENTS
	 * @see AbstractResult#USEREVENT_MAX
	 * @see AbstractResult#USEREVENT_MIN
	 * @see AbstractResult#USEREVENT_MEAN
	 * @see AbstractResult#USEREVENT_SUMSQR
	 */
	public Map<String, Double> getSortedByValue(String metric, int type, boolean ascending);

	/**
	 * This method returns the metric name which represents floating point operations.
	 * 
	 * @return the metric name
	 */
	public String getFPMetric();

	/**
	 * This method returns the metric name which represents L1 cache accesses.
	 * 
	 * @return the metric name
	 */
	public String getL1AccessMetric();

	/**
	 * This method returns the metric name which represents L2 cache accesses.
	 * 
	 * @return the metric name
	 */
	public String getL2AccessMetric();

	/**
	 * This method returns the metric name which represents L3 cache accesses.
	 * 
	 * @return the metric name
	 */
	public String getL3AccessMetric();

	/**
	 * This method returns the metric name which represents the L1 cache misses.
	 * 
	 * @return the metric name
	 */
	public String getL1MissMetric();

	/**
	 * This method returns the metric name which represents the L2 cache misses.
	 * 
	 * @return the metric name
	 */
	public String getL2MissMetric();

	/**
	 * This method returns the metric name which represents the L3 cache misses.
	 * 
	 * @return the metric name
	 */
	public String getL3MissMetric();

	/**
	 * This method returns the metric name which represents the TLB misses.
	 * 
	 * @return the metric name
	 */
	public String getTLBMissMetric();

	/**
	 * This method returns the metric name which represents the total number of instructions.
	 * 
	 * @return the metric name
	 */
	public String getTotalInstructionMetric();
	
	/**
	 * This method returns the Trial to which the performance data is related.
	 * 
	 * @return the Trial
	 */
	public Trial getTrial();

	/**
	 * This method returns the ID of the Trial to which the performance data is related.
	 * 
	 * @return the trial's ID
	 */
	public Integer getTrialID();
	
	/**
	 * Get the name for this input.
	 * @return the name of this performance result
	 */
	public String getName();
	
	/**
	 * Set the name for this input.
	 * @param name The new name for the input
	 */
	public void setName(String name);

	/**
	 * Get a Map of events in this result.
	 * @return the eventMap
	 * @see java.util.Map
	 */
	public Map<Integer, String> getEventMap();

	/**
	 * Set the Map of events in this result.
	 * @param eventMap the eventMap to set
	 * @see java.util.Map
	 */
	public void setEventMap(Map<Integer, String> eventMap);
	
	/**
	 * update the event map - remove what's missing, essentially
	 *
	 */
	public void updateEventMap();
	
	/**
	 * When values are requested from the trial, ignore warnings if the values are null
	 * @param ignore
	 */
	public void setIgnoreWarnings(boolean ignore);
}
