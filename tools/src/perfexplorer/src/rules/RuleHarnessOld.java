/**
 * @deprecated
 */
package edu.uoregon.tau.perfexplorer.rules;

import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.Reader;
import java.sql.SQLException;
import java.util.List;
import java.util.ListIterator;
import java.util.Iterator;
import java.util.Set;
import java.util.HashMap;

import org.drools.RuleBase;
import org.drools.RuleBaseFactory;
import org.drools.WorkingMemory;
import org.drools.compiler.PackageBuilder;
import org.drools.rule.Package;


import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.IntervalEvent;
import edu.uoregon.tau.perfdmf.IntervalLocationProfile;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfexplorer.client.ScriptFacade;
import edu.uoregon.tau.perfexplorer.common.RMISortableIntervalEvent;

/**
 * @deprecated
 */
public class RuleHarnessOld {

    private static ScriptFacade facade = new ScriptFacade();
    public static WorkingMemory workingMemory = null;

    public static final void processRules(Trial baseline, Trial comparison) {
        try {
        	
        	//load up the rulebase
            RuleBase ruleBase = readRule();
            workingMemory = ruleBase.newWorkingMemory();
            workingMemory.assertObject(new RelativeTrial(baseline, RelativeTrial.BASELINE));
            workingMemory.assertObject(new RelativeTrial(comparison, RelativeTrial.COMPARISON));
            System.out.println(" done.");
            workingMemory.fireAllRules();   

        } catch (Throwable t) {
            t.printStackTrace();
        }
    }

    /**
     * Please note that this is the "low level" rule assembly API.
     */
	private static RuleBase readRule() throws Exception {
		//read in the source
		Reader source = new InputStreamReader( RuleHarnessOld.class.getResourceAsStream( "/rules/PerfExplorerOld.drl" ) );
		//Reader source = new FileReader( "PerfExplorer.drl" );
		
		PackageBuilder builder = new PackageBuilder();

		//this wil parse and compile in one step
		//NOTE: There are 2 methods here, the one argument one is for normal DRL.
		System.out.print("Reading rules...");
		builder.addPackageFromDrl( source );
		System.out.println(" done.");

		//get the compiled package (which is serializable)
		Package pkg = builder.getPackage();
		
		//add the package to a rulebase (deploy the rule package).
		RuleBase ruleBase = RuleBaseFactory.newRuleBase();
		ruleBase.addPackage( pkg );
		return ruleBase;
	}
	
	/**
	 * This is a support class for the rules.  It is a wrapper
	 * around a Trial object, and identifies the trial as either
	 * the baseline or the comparison trial.  It also includes some
	 * convenience methods for getting trial data.
	 * 
	 * @author khuck
	 * @deprecated
	 *
	 */
	public static class RelativeTrial {
		// types
		public static final int BASELINE = 0; 
		public static final int COMPARISON = 1;
		
		// member variables
		private final int type;
		private Trial trial;
		private List<String> metrics;
		private HashMap<IntervalEvent, IntervalLocationProfile> events;
		private HashMap<String, IntervalEvent> eventNames;
		private IntervalEvent main;
		private int timeIndex;
		
		@SuppressWarnings("unchecked")
		public RelativeTrial (Trial trial, int type) {
			this.trial = trial;
			this.type = type;
			this.metrics = this.trial.getMetrics();
			this.timeIndex = findMetricIndex("Time");
			this.events = buildEventMap();
		}
		
		private HashMap<IntervalEvent, IntervalLocationProfile> buildEventMap () {
			HashMap<IntervalEvent, IntervalLocationProfile> eventMap = new HashMap<IntervalEvent, IntervalLocationProfile>();
			this.eventNames = new HashMap<String, IntervalEvent>();
			ListIterator<RMISortableIntervalEvent> events = facade.getEventList(this.trial, 0);
			// if we don't have a time metric, just use the first metric available
			int timeIndex = this.timeIndex==-1?0:this.timeIndex;
			double inclusive = 0.0;
			while (events.hasNext()) {
				IntervalEvent event = (IntervalEvent)events.next();
				this.eventNames.put(event.getName(), event);
				try {
					IntervalLocationProfile ilp =event.getMeanSummary();
					if (ilp.getInclusive(timeIndex) > inclusive) {
						inclusive = ilp.getInclusive(timeIndex);
						this.main = event;
					}
					eventMap.put(event, ilp);
				} catch (SQLException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			return eventMap;
		}

		public int findMetricIndex(String findme) {
		    int i = 0;
		    if (findme.equalsIgnoreCase("TIME")) {
		        // look for the usual "WALL_CLOCK_TIME" from PAPI/TAU
		        for (ListIterator iter = metrics.listIterator(); iter.hasNext(); ) {
		        	Metric metric = (Metric)iter.next();
		        	String name = metric.getName();
		            if (name.indexOf("WALL_CLOCK_TIME") > -1) {
		                return i;
		            }
		            i += 1;
		        }
		        i = 0;
		        // look for the usual "GET_TIME_OF_DAY" from PAPI/TAU
		        for (ListIterator iter = metrics.listIterator(); iter.hasNext(); ) {
		        	Metric metric = (Metric)iter.next();
		        	String name = metric.getName();
		            if (name.indexOf("GET_TIME_OF_DAY") > -1) {
		                return i;
		            }
		            i += 1;
		        }
		        i = 0;
		    }
		    // otherwise, just look for a match
	        for (ListIterator iter = metrics.listIterator(); iter.hasNext(); ) {
	        	Metric metric = (Metric)iter.next();
	        	String name = metric.getName();
	            if (name.equalsIgnoreCase(findme)) {
	                return i;
	            }
	            i += 1;
	        }
		    return -1;
		}
		
		public Trial getTrial() {
			return this.trial;
		}
		
		public int getType() {
			return this.type;
		}
		
		public IntervalEvent getMain() {
			return this.main;
		}
		
		public int getTimeIndex() {
			return this.timeIndex;
		}
		
		public Iterator<IntervalEvent> getEventIterator() {
			return events.keySet().iterator();
		}
		
		public IntervalEvent getEvent(String name) {
			return this.eventNames.get(name);
		}
	
		public List getMetrics () {
			return metrics;
		}
	}
	
	/**
	 * @deprecated
	 */
	public static class Difference {
		public static final int SAME = 0;
		public static final int SLOWER = 2;
		public static final int FASTER = 1;
		public static final double million = 1000000;
	
		private final int type;
		private final double baseline;
		private final double comparison;
		private final double difference;
		private final double percent;
		private final String eventName;
		private final String metricName;
		
		public Difference (int type, double baseline, double comparison, String eventName, String metricName) {
			this.type = type;
			this.baseline = baseline;
			this.comparison = comparison;
			double difference = this.baseline - this.comparison;
			if (difference > 0) {
				this.difference = difference;
				this.percent = difference / this.comparison;
			} else {
				this.difference = -1.0 * difference;				
				this.percent = this.difference / this.baseline;
			}
			this.eventName = eventName;
			this.metricName = metricName;
		}
		
		public int getType () {
			return this.type;
		}
		
		public double getBaseline() {
			return this.baseline;
		}
		
		public double getComparison() {
			return this.comparison;
		}
		
		public String toString() {
			StringBuilder buf = new StringBuilder();
			// need to format this better
			if (metricName != null) {
				buf.append(metricName + ", ");
			} else if (eventName != null) {
				buf.append(eventName + ", ");
			} else {
				buf.append("Event, Baseline, Comparison, Difference, Percent,\n");
				buf.append("main, ");
			}
			buf.append((baseline / this.million) + ", " + (comparison / this.million) + ", "
					+ (this.type == this.FASTER ? "-" : "") + 
					+ (difference / this.million) + ", " 
					//+ (this.type == this.FASTER ? "-" : "") + 
					+ (percent * 100.0) + "%");
			return buf.toString();
		}

		public String getEventName() {
			return eventName;
		}

		public String getMetricName() {
			return metricName;
		}
		
	}
	
    /**
     * @deprecated
     */
	public static class Helper {
		private final String name;
		private final Object object;
		private final Class<? extends Object> objectClass;
		
		public Helper (String name, Object object) {
			this.name = name;
			this.object = object;
			this.objectClass = object.getClass();
		}
		
		public String getName () {
			return this.name;
		}

		public Object getObject() {
			return this.object;
		}

		public Class<? extends Object> getObjectClass() {
			return this.objectClass;
		}
	}
}
