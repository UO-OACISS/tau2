 package edu.uoregon.tau.perfdmf;

import java.io.*;
import java.util.*;

import edu.uoregon.tau.common.TrackerInputStream;

public class GPTLDataSource extends DataSource {

	private int linenumber = 0;
	private int currentProcess = 0;
	private int currentThread = 0;
	private File file = null;
	private GlobalData globalData = null;
    
    private volatile long totalBytes = 0;
    private volatile TrackerInputStream tracker;

	
    public GPTLDataSource(File file) {
        super();
        this.file = file;
    }

    public void cancelLoad() {
        return;
    }

    public int getProgress() {
        if (totalBytes != 0) {
            return (int) ((float) tracker.byteCount() / (float) totalBytes * 100);
        }
        return 0;
    }

    public void load() throws FileNotFoundException, IOException {
        //Record time.
        long time = System.currentTimeMillis();

        Node node = null;
        Context context = null;
        Thread thread = null;
        int nodeID = -1;


		//System.out.println("Processing " + file + ", please wait ......");
		FileInputStream fileIn = new FileInputStream(file);
		tracker = new TrackerInputStream(fileIn);
		InputStreamReader inReader = new InputStreamReader(tracker);
		BufferedReader br = new BufferedReader(inReader);
		
		totalBytes = file.length();

		// process the global section, and do what's necessary
		globalData = processGlobalSection(br);
		//System.out.println("Num Tasks: " + globalData.numTasks);

		// process the process/thread data
		ThreadData data = processThreadData(br);
		while ( data != null ) {

			//clearMetrics();

			// the data is loaded, so create the node/context/thread id
			// for this data set
          	node = this.addNode(data.processid);
           	context = node.addContext(0);
           	thread = context.addThread(data.threadid);

			// cycle through the events
			for (int i = 0 ; i < data.eventData.size() ; i++ ) {
				EventData eventData = data.eventData.get(i);
				createFunction(thread, eventData, false);
				// for the first function, don't create callpath, just flat
				if (i > 0) {
					createFunction(thread, eventData, true);
				}
			}
        	this.generateDerivedData();
			// get next thread
			data = processThreadData(br);

		} // while process/thread

		setGroupNamesPresent(true);

        time = (System.currentTimeMillis()) - time;
        //System.out.println("Done processing data!");
        //System.out.println("Time to process (in milliseconds): " + time);
        fileIn.close();
    }


	private void createFunction(Thread thread, EventData eventData, boolean doCallpath) {
        Function function = null;
        FunctionProfile functionProfile = null;

		// for this function, create a function

		if (doCallpath) {
			function = this.addFunction(eventData.callpathName);
			function.addGroup(this.addGroup("TAU_CALLPATH"));
		} else {
			function = this.addFunction(eventData.name);
			function.addGroup(this.addGroup("TAU_DEFAULT"));
		}

		// create a function profile for this process/thread
		functionProfile = new FunctionProfile(function, 3+(2*globalData.metrics.size()));
		// add it to the current thread
		thread.addFunctionProfile(functionProfile);

		// set the values for the each metric
		functionProfile.setNumCalls(eventData.calls);
		functionProfile.setNumSubr(eventData.children.size());

		Measurements inclusive = eventData.inclusive;
		Measurements exclusive = eventData.getExclusive();

		Metric m = addMetric("WALL_CLOCK_TIME", thread);
		functionProfile.setInclusive(m.getID(), inclusive.wallclock * 1000000);
		functionProfile.setExclusive(m.getID(), exclusive.wallclock * 1000000);

		m = addMetric("WALL_CLOCK_TIME max", thread);
		functionProfile.setInclusive(m.getID(), inclusive.wallclockMax * 1000000);
		// this is somewhat meaningless as exclusive...
		// so use the inclusive value
		functionProfile.setExclusive(m.getID(), inclusive.wallclockMax * 1000000);

		m = this.addMetric("WALL_CLOCK_TIME min", thread);
		functionProfile.setInclusive(m.getID(), inclusive.wallclockMin * 1000000);
		// this is somewhat meaningless as exclusive...
		// so use the inclusive value
		functionProfile.setExclusive(m.getID(), inclusive.wallclockMin * 1000000);

//		this data shouldn't be archived -
//		it's the measurement overhead for the GPTL timers.
/*
		m = this.addMetric("UTR Overhead", thread);
		functionProfile.setInclusive(m.getID(), inclusive.utrOverhead);
		functionProfile.setExclusive(m.getID(), exclusive.utrOverhead);

		m = this.addMetric("OH (cyc)", thread);
		functionProfile.setInclusive(m.getID(), inclusive.ohCycles);
		functionProfile.setExclusive(m.getID(), exclusive.ohCycles);
*/

		for (int j = 0 ; j < globalData.metrics.size() ; j++ ){
			String metric = globalData.metrics.get(j);
			m = this.addMetric(metric, thread);
			functionProfile.setInclusive(m.getID(), inclusive.papi[j]);
			functionProfile.setExclusive(m.getID(), exclusive.papi[j]);

			m = this.addMetric(metric + " e6/sec", thread);
			functionProfile.setInclusive(m.getID(), inclusive.papiE6OverSeconds[j]);
			functionProfile.setExclusive(m.getID(), exclusive.papiE6OverSeconds[j]);
		}

	}

	private GlobalData processGlobalSection(BufferedReader br) {
		GlobalData data = new GlobalData();
		String inputString = null;
		String tmp = null;
		try {
			while((inputString = br.readLine()) != null){
				// ***** GLOBAL STATISTICS (  128 MPI TASKS) *****
				if (inputString.trim().startsWith("***** GLOBAL STATISTICS ")) {
        			StringTokenizer st = new StringTokenizer(inputString, " \t\n\r");
					// stars
        			tmp = st.nextToken();
					// GLOBAL
        			tmp = st.nextToken();
					// STATISTICS
        			tmp = st.nextToken();
					// (
        			tmp = st.nextToken();
					// number of tasks
        			tmp = st.nextToken();
					data.numTasks = Integer.parseInt(tmp);
				}
				// name count wallmax (proc thrd) wallmin (proc thrd) ...
				else if (inputString.trim().startsWith("name")) {
        			StringTokenizer st = new StringTokenizer(inputString, " \t\n\r()");
					// name
        			tmp = st.nextToken();
					// count
        			tmp = st.nextToken();
					// wallmax
					tmp = st.nextToken();
					// proc
       				tmp = st.nextToken();
					// thread
       				tmp = st.nextToken();
					// wallmin
       				tmp = st.nextToken();
					// proc
       				tmp = st.nextToken();
					// thread
       				tmp = st.nextToken();
					// loop through the metrics, and get their names
					while (st.hasMoreTokens()) {
						// METRICmax
						tmp = st.nextToken();
						data.metrics.add(tmp.replaceAll("max",""));
						// proc
        				tmp = st.nextToken();
						// thread
        				tmp = st.nextToken();
						// METRICmin
        				tmp = st.nextToken();
						// proc
        				tmp = st.nextToken();
						// thread
        				tmp = st.nextToken();
					}
				}
				else if (inputString.trim().length() == 0) {
					// if we already have the metrics, then break out.
					if (data.metrics.size() > 0)
						break;
				}
				// anything else
				else {
					// ignore this line
				}
			}
		} catch (IOException e) {
			System.out.println(e.getMessage());
			e.printStackTrace();
		}
		return data;
	}

	private ThreadData processThreadData(BufferedReader br) {
		ThreadData data = null;
		String inputString = null;
		String tmp = null;
		try {
			boolean inData = false;
			boolean previousLineBlank = false;
			Stack<EventData> eventStack = new Stack<EventData>();
			while((inputString = br.readLine()) != null){
				// ************ PROCESS     0 (    0) ************
				if (inputString.trim().startsWith("************ PROCESS")) {
					previousLineBlank = false;
					data = new ThreadData();
        			StringTokenizer st = new StringTokenizer(inputString, " \t\n\r()");
					// stars
        			tmp = st.nextToken();
					// PROCESS
        			tmp = st.nextToken();
					// process ID
        			tmp = st.nextToken();
					// parse the process ID
					data.processid = Integer.parseInt(tmp,10);
					// process ID again?
        			//tmp = st.nextToken();
					//data.threadid = Integer.parseInt(tmp);
				}
				// Stats for thread 0:
				else if (inputString.trim().startsWith("Stats for thread 0:")) {
					previousLineBlank = false;
					inData = true;
        			StringTokenizer st = new StringTokenizer(inputString, " \t\n\r:");
					// Stats
        			tmp = st.nextToken();
					// for
        			tmp = st.nextToken();
					// thread
        			tmp = st.nextToken();
					// thread ID
        			tmp = st.nextToken();
					data.threadid = Integer.parseInt(tmp);
				}
				// Called Recurse Wallclock max min % of TOTAL UTR Overhead TOT_CYC e6/sec FP_OPS e6/sec FP_INS e6/sec OH (cyc)...
				else if (inputString.trim().startsWith("Called")) {
					previousLineBlank = false;
					// handle the header
					inData = true;
				}
				// Overhead sum           =     0.012 wallclock seconds
				else if (inputString.trim().startsWith("Overhead sum")) {
					previousLineBlank = false;
					// end of the data section
					inData = false;
				}
				// Total calls           = 63733
				else if (inputString.trim().startsWith("Total calls")) {
					previousLineBlank = false;
					// end of the data section
					inData = false;
				}
				// Total recursive calls = 0
				else if (inputString.trim().startsWith("Total recursive calls")) {
					previousLineBlank = false;
					// end of the data section
					inData = false;
				}
				// thread 0 had some hash collisions:
				else if (inputString.trim().startsWith("thread")) {
					previousLineBlank = false;
					// end of the data section
					inData = false;
				}
				// hashtable[0][900] had 2 entries: TOTAL                 PHASE3
				else if (inputString.trim().startsWith("hashtable")) {
					previousLineBlank = false;
					// end of the data section
					inData = false;
				}
				else if (inputString.trim().length() == 0) {
					// if we already have the metrics, then break out.
					if (previousLineBlank && !inData)
						break;
					previousLineBlank = true;
				}
				else if (!inData) { 
					previousLineBlank = false;
					// ignore this line
				}
				else if (inData) { 
					previousLineBlank = false;
					// this is a data line!
					EventData eventData = processEventLine(inputString);
					data.eventData.add(eventData);
					if (eventStack.empty()) {
						// this is the new top level event
						eventStack.push(eventData);
					} else {
						// peek at the top, and get the depth for the event
						EventData parent = eventStack.peek();
						// if the just read event is deeper than the current parent,
						// add it to the stack
						if (eventData.depth > parent.depth) {
							eventStack.push(eventData);
							parent.children.add(eventData);
							eventData.callpathName = parent.callpathName + " => " + eventData.callpathName;
						}
						// if the just read event is at the same depth, pop the
						// current parent and replace it with the just read event
						else if (eventData.depth == parent.depth) {
							eventStack.pop();
							parent = eventStack.peek();
							parent.children.add(eventData);
							eventData.callpathName = parent.callpathName + " => " + eventData.callpathName;
							eventStack.push(eventData);
						}
						// if the just read event is at a shallower depth, pop
						// until we get to something which is at a higher level.
						else if (eventData.depth < parent.depth) {
							while (eventData.depth <= parent.depth) {
								eventStack.pop();
								parent = eventStack.peek();
							}
							parent.children.add(eventData);
							eventData.callpathName = parent.callpathName + " => " + eventData.callpathName;
							eventStack.push(eventData);
						}
					}
				}
			}
		} catch (IOException e) {
			System.out.println(e.getMessage());
			e.printStackTrace();
		}
		return data;
	}

	private EventData processEventLine(String inputString) {
		EventData data = new EventData();
		// process the inclusive values for this event
        StringTokenizer st = new StringTokenizer(inputString, " \t\n\r()");
		// name
        data.name = st.nextToken();
		// we may have parsed an asterix... check for it!
		if (data.name.equals("*")) {
        	data.name = st.nextToken();
			// get the depth of this event
			data.depth = inputString.length() - inputString.replaceFirst("\\*","").trim().length();
		} else {
			// get the depth of this event
			data.depth = inputString.length() - inputString.trim().length();
		}
		data.callpathName = data.name;
		// num calls
        data.calls = Integer.parseInt(st.nextToken());
		// recurse - what do do with this?
        st.nextToken();
		// Wallclock - the output is a total, so divide by the number of calls
		data.inclusive.wallclock = Double.parseDouble(st.nextToken());
		// WallclockMax
		data.inclusive.wallclockMax = Double.parseDouble(st.nextToken());
		// WallclockMin
		data.inclusive.wallclockMin = Double.parseDouble(st.nextToken());
		// % of total - ignore
        st.nextToken();
		// UTR Overhead
		data.inclusive.utrOverhead = Double.parseDouble(st.nextToken());
		// there are N PAPI events, and two measurements for each
		data.inclusive.papi = new double[globalData.metrics.size()];
		data.inclusive.papiE6OverSeconds = new double[globalData.metrics.size()];
		for (int i = 0 ; i < globalData.metrics.size() ; i++) {
			// PAPI - the output is a total, so divide by the number of calls
			data.inclusive.papi[i] = Double.parseDouble(st.nextToken());
			// e6/sec
			data.inclusive.papiE6OverSeconds[i] = Double.parseDouble(st.nextToken());
		}
		// OH (cyc)
		data.inclusive.ohCycles = Double.parseDouble(st.nextToken());
		// TODO: how to deal with an event that happens at two call sites?
		return data;
	}

	private class GlobalData {
		public int numTasks = 0;
		public List<String> metrics = new ArrayList<String>();
	}

	private class ThreadData {
		public List<EventData> eventData = new ArrayList<EventData>();	
		public int processid = 0;
		public int threadid = 0;
	}

	private class EventData {
		public String name;
		public String callpathName;
		public int depth;
		public int calls;
		private List<EventData> children = new ArrayList<EventData>();
		public Measurements inclusive = new Measurements();
		public Measurements getExclusive() {
			Measurements exclusive = (Measurements)inclusive.clone();
			//System.out.println("Getting exclusive for " + callpathName);
			for (int i = 0 ; i < children.size() ; i++) {
				EventData child = children.get(i);
				//System.out.println("	Child:  " + child.name);
				exclusive.wallclock -= child.inclusive.wallclock;
				exclusive.wallclockMax -= child.inclusive.wallclockMax;
				exclusive.wallclockMin -= child.inclusive.wallclockMin;
				exclusive.utrOverhead -= child.inclusive.utrOverhead;
				exclusive.ohCycles -= child.inclusive.ohCycles;
				for (int j = 0 ; j < globalData.metrics.size() ; j++ ){
					exclusive.papi[j] -= child.inclusive.papi[j];
					//exclusive.papiE6OverSeconds[j] -= child.inclusive.papiE6OverSeconds[j];
				}
			}
			exclusive.wallclock = exclusive.wallclock < 0.0 ? 0.0 : exclusive.wallclock;
			exclusive.wallclockMax = exclusive.wallclockMax < 0.0 ? 0.0 : exclusive.wallclockMax;
			exclusive.wallclockMin = exclusive.wallclockMin < 0.0 ? 0.0 : exclusive.wallclockMin;
			exclusive.utrOverhead = exclusive.utrOverhead < 0.0 ? 0.0 : exclusive.utrOverhead;
			exclusive.ohCycles = exclusive.ohCycles < 0.0 ? 0.0 : exclusive.ohCycles;
			for (int j = 0 ; j < globalData.metrics.size() ; j++ ){
				exclusive.papi[j] = exclusive.papi[j] < 0.0 ? 0.0 : exclusive.papi[j];
				//exclusive.papiE6OverSeconds[j] = exclusive.papiE6OverSeconds[j] < 0.0 ? 0.0 : exclusive.papiE6OverSeconds[j];
				// compute the E6 / seconds value for exclusive
				exclusive.papiE6OverSeconds[j] = (exclusive.papi[j] / 1000000) / exclusive.wallclock;
			}
			return exclusive;
		}
		public boolean hasChildren() {
			if (children.size() > 0) return true;
			return false;
		}
	}

	private class Measurements implements Cloneable {
		public double wallclock ;
		public double wallclockMax ;
		public double wallclockMin ;
		public double utrOverhead ;
		public double[] papi ;
		public double[] papiE6OverSeconds ;
		public double ohCycles ;
		public Object clone () {
			Measurements cloned = new Measurements();
			cloned.wallclock = this.wallclock;
			cloned.wallclockMax = this.wallclockMax;
			cloned.wallclockMin = this.wallclockMin;
			cloned.utrOverhead = this.utrOverhead;
			cloned.ohCycles = this.ohCycles;
			cloned.papi = new double[globalData.metrics.size()];
			cloned.papiE6OverSeconds = new double[globalData.metrics.size()];
			for (int j = 0 ; j < globalData.metrics.size() ; j++ ){
				cloned.papi[j] = this.papi[j];
				cloned.papiE6OverSeconds[j] = this.papiE6OverSeconds[j];
			}
			return cloned;
		}
	}

}