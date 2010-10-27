/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.io.File;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.UserEvent;
import edu.uoregon.tau.perfdmf.UserEventProfile;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * @author khuck
 * 
 */
public class DataSourceResult extends AbstractResult {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1446658362802536887L;
	public static final int PPK = DataSource.PPK;
	public static final int TAUPROFILE = DataSource.TAUPROFILE;
	public static final int DYNAPROF = DataSource.DYNAPROF;
	public static final int MPIP = DataSource.MPIP;
	public static final int HPM = DataSource.HPM;
	public static final int GPROF = DataSource.GPROF;
	public static final int PSRUN = DataSource.PSRUN;
	public static final int PPROF = DataSource.PPROF;
	public static final int CUBE = DataSource.CUBE;
	public static final int HPCTOOLKIT = DataSource.HPCTOOLKIT;
	public static final int SNAP = DataSource.SNAP;
	public static final int OMPP = DataSource.OMPP;
	public static final int PERIXML = DataSource.PERIXML;
	public static final int GPTL = DataSource.GPTL;
	public static final int PARAVER = DataSource.PARAVER;
	public static final int IPM = DataSource.IPM;
	public static final int GOOGLE = DataSource.GOOGLE;
	public static final int GYRO = DataSource.GYRO;
	public static final int GAMESS = DataSource.GAMESS;

	private DataSource source = null;
	private List<Thread> threadList = null;
	private Map<String, Integer> metricMap = new HashMap<String, Integer>();

	protected Map<Integer, String> eventMap = new HashMap<Integer, String>();

	public DataSourceResult(int fileType, String[] sourceFiles, boolean fixNames) {
		File[] files = new File[sourceFiles.length];
		for (int i = 0; i < sourceFiles.length; i++) {
			files[i] = new File(sourceFiles[i]);
			this.name = files[i].getName();
		}
		source = UtilFncs.initializeDataSource(files, fileType, fixNames);
		try {
			long start = System.currentTimeMillis();

			this.trial = new Trial();
			this.trial.setDataSource(source);
			this.dataSource = source;

			source.load();

			// set the meta data from the datasource
			this.trial.setMetaData(source.getMetaData());
			this.trial.setUncommonMetaData(source.getUncommonMetaData());
			this.trial.setName(this.name);
//			this.trial.setField("node_count", Integer.toString(source.getNumberOfNodes()));
//			int nodes = this.source.getNumberOfNodes();
//			int contexts = 0;
//			int threads = 0;
//			for (int n = 0; n < nodes; n++) {
//				contexts = Math.max(contexts, this.source.getNumberOfContexts(n));
//				for (int c = 0; c < contexts; c++) {
//					threads = Math.max(threads, this.source.getNumberOfThreads(n, c));
//				}
//			}
//			this.trial.setField("contexts_per_node", Integer.toString(contexts));
//			this.trial.setField("threads_per_context", Integer.toString(threads));

			long elapsedTimeMillis = System.currentTimeMillis() - start;
			float elapsedTimeSec = elapsedTimeMillis / 1000F;
			System.out.println("Total time to read data from disk: "
					+ elapsedTimeSec + " seconds");
		} catch (Exception e) {
			System.err.println(e.getMessage());
			e.printStackTrace(System.err);
			return;
		}

		// get the threads
		threadList = source.getThreads();
		for (int i = 0; i < threadList.size(); i++) {
			threads.add(new Integer(i));
		}

		// get the metrics
		for (int m = 0; m < source.getNumberOfMetrics(); m++) {
			String metric = source.getMetric(m).getName();
			metricMap.put(metric, m);
			metrics.add(metric);
		}

		// get the functions
		Iterator<Function> functions = source.getFunctions();
		while (functions.hasNext()) {
			Function function = functions.next();
			String name = function.getName();
			events.add(name);
		}

		Iterator<UserEvent> userEvents = source.getUserEvents();
		while (userEvents.hasNext()) {
			UserEvent userEvent = userEvents.next();
			String name = userEvent.getName();
			this.userEvents.add(name);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getCalls(java.lang
	 * .Integer, java.lang.String)
	 */
	public double getCalls(Integer thread, String event) {
		Thread t = threadList.get(thread);
		Function f = source.getFunction(event);
		try {
			FunctionProfile fp = t.getFunctionProfile(f);
			return fp.getNumCalls();
		} catch (NullPointerException e) {
			if (!ignoreWarnings)
				System.err
						.println("*** Warning - null numCalls value for thread: "
								+ thread + ", event: " + event + " ***");
			return 0.0;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getExclusive(java
	 * .lang.Integer, java.lang.String, java.lang.String)
	 */
	public double getExclusive(Integer thread, String event, String metric) {
		Thread t = threadList.get(thread);
		Function f = source.getFunction(event);
		try {
			FunctionProfile fp = t.getFunctionProfile(f);
			return fp.getExclusive(metricMap.get(metric));
		} catch (NullPointerException e) {
			if (!ignoreWarnings)
				System.err
						.println("*** Warning - null exclusive value for thread: "
								+ thread
								+ ", event: "
								+ event
								+ ", metric: "
								+ metric + " ***");
			return 0.0;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getInclusive(java
	 * .lang.Integer, java.lang.String, java.lang.String)
	 */
	public double getInclusive(Integer thread, String event, String metric) {
		Thread t = threadList.get(thread);
		Function f = source.getFunction(event);
		try {
			FunctionProfile fp = t.getFunctionProfile(f);
			return fp.getInclusive(metricMap.get(metric));
		} catch (NullPointerException e) {
			if (!ignoreWarnings)
				System.err
						.println("*** Warning - null inclusive value for thread: "
								+ thread
								+ ", event: "
								+ event
								+ ", metric: "
								+ metric + " ***");
			return 0.0;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getMainEvent()
	 */
	public String getMainEvent() {
		if (mainEvent == null) {
			for (Thread thread : threadList) {
				int m = 0;
				mainMetric = source.getMetric(m).getName();
				Iterator<Function> functions = source.getFunctions();
				while (functions.hasNext()) {
					Function function = functions.next();
					String name = function.getName();
					FunctionProfile fp = thread.getFunctionProfile(function);
					if (fp != null && mainInclusive < fp.getInclusive(m)) {
						mainInclusive = fp.getInclusive(m);
						mainEvent = name;
					}
				}
			}
		}
		return mainEvent;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getUserEvents(Integer
	 * )
	 */
	public Set<String> getUserEvents(Integer thread) {
		Set<String> ues = new TreeSet<String>();
		Thread t = threadList.get(thread.intValue());
		Iterator<UserEventProfile> iter = t.getUserEventProfiles();
		while (iter.hasNext()) {
			UserEventProfile uep = iter.next();
			ues.add(uep.getUserEvent().getName());
		}
		return ues;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getOriginalThreads()
	 */
	public Integer getOriginalThreads() {
		return threadList.size();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getSubroutines(java
	 * .lang.Integer, java.lang.String)
	 */
	public double getSubroutines(Integer thread, String event) {
		Thread t = threadList.get(thread);
		Function f = source.getFunction(event);
		try {
			FunctionProfile fp = t.getFunctionProfile(f);
			return fp.getNumSubr();
		} catch (NullPointerException e) {
			if (!ignoreWarnings)
				System.err
						.println("*** Warning - null subroutines value for thread: "
								+ thread + ", event: " + event + " ***");
			return 0.0;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getTrial()
	 */
	public Trial getTrial() {
		return null;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getTrialID()
	 */
	public Integer getTrialID() {
		return new Integer(0);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getUsereventMax(java
	 * .lang.Integer, java.lang.String)
	 */
	public double getUsereventMax(Integer thread, String event) {
		Thread t = threadList.get(thread);
		UserEvent ue = source.getUserEvent(event);
		UserEventProfile uep = t.getUserEventProfile(ue);
		if (uep != null) {
			return uep.getMaxValue();
		}
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getUsereventMean(
	 * java.lang.Integer, java.lang.String)
	 */
	public double getUsereventMean(Integer thread, String event) {
		Thread t = threadList.get(thread);
		UserEvent ue = source.getUserEvent(event);
		UserEventProfile uep = t.getUserEventProfile(ue);
		if (uep != null) {
			return uep.getMeanValue();
		}
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getUsereventMin(java
	 * .lang.Integer, java.lang.String)
	 */
	public double getUsereventMin(Integer thread, String event) {
		Thread t = threadList.get(thread);
		UserEvent ue = source.getUserEvent(event);
		UserEventProfile uep = t.getUserEventProfile(ue);
		if (uep != null) {
			return uep.getMinValue();
		}
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getUsereventNumevents
	 * (java.lang.Integer, java.lang.String)
	 */
	public double getUsereventNumevents(Integer thread, String event) {
		Thread t = threadList.get(thread);
		UserEvent ue = source.getUserEvent(event);
		UserEventProfile uep = t.getUserEventProfile(ue);
		if (uep != null) {
			return uep.getNumSamples();
		}
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#getUsereventSumsqr
	 * (java.lang.Integer, java.lang.String)
	 */
	public double getUsereventSumsqr(Integer thread, String event) {
		Thread t = threadList.get(thread);
		UserEvent ue = source.getUserEvent(event);
		UserEventProfile uep = t.getUserEventProfile(ue);
		if (uep != null) {
			return uep.getSumSquared();
		}
		return 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#putCalls(java.lang
	 * .Integer, java.lang.String, double)
	 */
	public void putCalls(Integer thread, String event, double value) {
		System.err
				.println("*** putCalls not implemented for DataSourceResult ***");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#putDataPoint(java
	 * .lang.Integer, java.lang.String, java.lang.String, int, double)
	 */
	public void putDataPoint(Integer thread, String event, String metric,
			int type, double value) {
		System.err
				.println("*** putDataPoint not implemented for DataSourceResult ***");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#putExclusive(java
	 * .lang.Integer, java.lang.String, java.lang.String, double)
	 */
	public void putExclusive(Integer thread, String event, String metric,
			double value) {
		System.err
				.println("*** putExclusive not implemented for DataSourceResult ***");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#putInclusive(java
	 * .lang.Integer, java.lang.String, java.lang.String, double)
	 */
	public void putInclusive(Integer thread, String event, String metric,
			double value) {
		System.err
				.println("*** putInclusive not implemented for DataSourceResult ***");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#putSubroutines(java
	 * .lang.Integer, java.lang.String, double)
	 */
	public void putSubroutines(Integer thread, String event, double value) {
		System.err
				.println("*** putSubroutines not implemented for DataSourceResult ***");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#putUsereventMax(java
	 * .lang.Integer, java.lang.String, double)
	 */
	public void putUsereventMax(Integer thread, String event, double value) {
		System.err
				.println("*** putUsereventMax not implemented for DataSourceResult ***");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#putUsereventMean(
	 * java.lang.Integer, java.lang.String, double)
	 */
	public void putUsereventMean(Integer thread, String event, double value) {
		System.err
				.println("*** putUsereventMean not implemented for DataSourceResult ***");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#putUsereventMin(java
	 * .lang.Integer, java.lang.String, double)
	 */
	public void putUsereventMin(Integer thread, String event, double value) {
		System.err
				.println("*** putUsereventMin not implemented for DataSourceResult ***");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#putUsereventNumevents
	 * (java.lang.Integer, java.lang.String, double)
	 */
	public void putUsereventNumevents(Integer thread, String event, double value) {
		System.err
				.println("*** putUsereventNumevents not implemented for DataSourceResult ***");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.uoregon.tau.perfexplorer.glue.PerformanceResult#putUsereventSumsqr
	 * (java.lang.Integer, java.lang.String, double)
	 */
	public void putUsereventSumsqr(Integer thread, String event, double value) {
		System.err
				.println("*** putUsereventsSumsqr not implemented for DataSourceResult ***");
	}

}
