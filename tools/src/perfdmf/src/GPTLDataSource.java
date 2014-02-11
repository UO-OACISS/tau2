 package edu.uoregon.tau.perfdmf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Stack;
import java.util.StringTokenizer;

import edu.uoregon.tau.common.TrackerInputStream;

public class GPTLDataSource extends DataSource {

	//private int linenumber = 0;
	//private int currentProcess = 0;
	//private int currentThread = 0;
	private File file = null;
	private GlobalData globalData = null;
    
    private volatile int totalFiles = 0;
    private volatile int filesRead = 0;

    private volatile long totalBytes = 0;
    private volatile TrackerInputStream tracker;
    private List<File[]> dirs; // list of directories
    private boolean oneFile = false;
    private File globalFile = null; // global stats file
    private boolean isPAPI = false; // do we have PAPI data?
    private List<String> dataColumns = new ArrayList<String>(); // the data columns in the file
    private int currentProcess = 0;
	
    private File fileToMonitor;

    public GPTLDataSource(File file) {
        super();
        this.file = file;
        this.oneFile = true;
    }

    @SuppressWarnings("unchecked")
	public GPTLDataSource(List<?> dirs) {
        super();

        if (dirs.size() > 0) {
            if (dirs.get(0) instanceof File[]) {
            	this.dirs=(List<File[]>)dirs;
                File[] files = (File[])dirs.get(0);
                if (files.length > 0) {
                    fileToMonitor = files[0];
                }
            } else {
                this.dirs = new ArrayList<File[]>();
                File[] files = new File[1];
                files[0] = (File) dirs.get(0);
                this.dirs.add(files);
                fileToMonitor = files[0];
                //System.out.println(files[0].toString());
            }
        }
    }

    public void cancelLoad() {
        return;
    }

    public int getProgress() {
    	if (oneFile) {
	        if (totalBytes != 0) {
	            return (int) ((float) tracker.byteCount() / (float) totalBytes * 100);
	        }
    	} else {
            if (totalFiles != 0)
                return (int) ((float) filesRead / (float) totalFiles * 100);
    	}
        return 0;
    }

    public void load() throws FileNotFoundException, IOException {
    	
    	if (oneFile) {
    		loadOneFile();
    	} else {
    		loadMultipleFiles();
    	}
    }
    
    public void loadOneFile() throws FileNotFoundException, IOException {
    	
        //Record time.
        long time = System.currentTimeMillis();

        Node node = null;
        Context context = null;
        Thread thread = null;
        //int nodeID = -1;
        
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
		List<ThreadData> dataList = processThreadData(br);
		for (ThreadData data : dataList) {

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
		} // while process/thread
    	this.generateDerivedData();

		setGroupNamesPresent(true);

        time = (System.currentTimeMillis()) - time;
        //System.out.println("Done processing data!");
        //System.out.println("Time to process (in milliseconds): " + time);
        fileIn.close();
    }

    public void loadMultipleFiles() throws FileNotFoundException, IOException {
    	
        //Record time.
        long time = System.currentTimeMillis();

        // first count the files (for progressbar)
        for (Iterator<File[]> e = dirs.iterator(); e.hasNext();) {
            File[] files = e.next();
            for (int i = 0; i < files.length; i++) {
                totalFiles++;
                // and get the _stats file
                if (files[i].getName().endsWith("_stats")) {
                	globalFile = files[i];
                }
            }
        }

        Node node = null;
        Context context = null;
        Thread thread = null;
        //int nodeID = -1;
        
        // Process the global file
        file = globalFile;
        
		//System.out.println("Processing " + file + ", please wait ......");
		FileInputStream fileIn = new FileInputStream(file);
		tracker = new TrackerInputStream(fileIn);
		InputStreamReader inReader = new InputStreamReader(tracker);
		BufferedReader br = new BufferedReader(inReader);
		
		totalBytes = file.length();

		// process the global section, and do what's necessary
		globalData = processGlobalSection(br);
		//System.out.println("Num Tasks: " + globalData.numTasks);
        fileIn.close();

        // iterate through the vector of File arrays (each directory)
        for (Iterator<File[]> e = dirs.iterator(); e.hasNext();) {
            File[] files = e.next();

            for (int i = 0; i < files.length; i++) {
                file = files[i];
                if (file.getName().endsWith("_stats")) {
                	continue; // we already handled this file
                }
                
        		//System.out.println("Processing " + file + ", please wait ......");
        		//System.out.print(".");
        		fileIn = new FileInputStream(file);
        		tracker = new TrackerInputStream(fileIn);
        		inReader = new InputStreamReader(tracker);
        		br = new BufferedReader(inReader);
        		
        		totalBytes = file.length();
        		// process the process/thread data
        		List<ThreadData> dataList = processThreadData(br);
        		for (ThreadData data : dataList) {

        			//clearMetrics();

        			// the data is loaded, so create the node/context/thread id
        			// for this data set
        			node = this.addNode(data.processid);
        			context = node.addContext(0);
        			thread = context.addThread(data.threadid);

        			// cycle through the events
        			for (int j = 0 ; j < data.eventData.size() ; j++ ) {
        				EventData eventData = data.eventData.get(j);
        				createFunction(thread, eventData, false);
        				if (!eventData.callpathName.equals(eventData.name))
        					createFunction(thread, eventData, true);
        			}
        		} // for data in dataList

        		fileIn.close();
            }
    		setGroupNamesPresent(true);

    		time = (System.currentTimeMillis()) - time;
    		//System.out.println("Done processing data!");
    		//System.out.println("Time to process (in milliseconds): " + time);
        }
		this.generateDerivedData();
    }


	private void createFunction(Thread thread, EventData eventData, boolean doCallpath) {
        Function function = null;
        FunctionProfile functionProfile = null;

		// for this function, create a function

		if (doCallpath) {
			function = this.addFunction(eventData.callpathName);
			function.addGroup(this.addGroup("TAU_DEFAULT"));
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
        			while (st.hasMoreTokens()) {
        				tmp = st.nextToken();
        				if (tmp.equalsIgnoreCase("name")) {/* do nothing */}
        				else if (tmp.equalsIgnoreCase("processes")) {/* do nothing */}
        				else if (tmp.equalsIgnoreCase("threads")) {/* do nothing */}
        				else if (tmp.equalsIgnoreCase("count")) {/* do nothing */}
        				else if (tmp.equalsIgnoreCase("walltotal")) {/* do nothing */}
        				else if (tmp.equalsIgnoreCase("wallmax")) {/* do nothing */}
        				else if (tmp.equalsIgnoreCase("wallmin")) {/* do nothing */}
        				else if (tmp.equalsIgnoreCase("proc")) {/* do nothing */}
        				else if (tmp.equalsIgnoreCase("thrd")) {/* do nothing */}
        				else if (tmp.endsWith("max")) {
        					data.metrics.add(tmp.replaceAll("max", ""));
        				} else if (tmp.endsWith("min")) { /* do nothing */ }
        			}
/*        			if (data.metrics.size() == 0) {
        				data.metrics.add("WALLTIME");
        			}
*/				}
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

	private List<ThreadData> processThreadData(BufferedReader br) {
		List<ThreadData> dataList = new ArrayList<ThreadData>();
		ThreadData data = null;
		String inputString = null;
		String tmp = null;
		boolean inFooter = false;
		try {
			boolean inData = false;
			boolean previousLineBlank = false;
			Stack<EventData> eventStack = new Stack<EventData>();
			while((inputString = br.readLine()) != null){
				//System.out.println(inputString);
				// ************ PROCESS     0 (    0) ************
				if (inputString.trim().startsWith("************ PROCESS")) {
					previousLineBlank = false;
        			StringTokenizer st = new StringTokenizer(inputString, " \t\n\r()");
					// stars
        			tmp = st.nextToken();
					// PROCESS
        			tmp = st.nextToken();
					// process ID
        			tmp = st.nextToken();
					// parse the process ID
        			currentProcess = Integer.parseInt(tmp,10);
					// process ID again?
        			//tmp = st.nextToken();
					//data.threadid = Integer.parseInt(tmp);
				}
				// Stats for thread 0:
				else if (inputString.trim().startsWith("Stats for thread ")) {
					previousLineBlank = false;
					data = new ThreadData();
					data.processid = currentProcess;
					this.dataColumns.clear();
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
				// On Called Recurse Wallclock max min UTR Overhead
				else if (inputString.trim().startsWith("On")) {
					previousLineBlank = false;
					// handle the header
					inData = true;
					isPAPI = false;
					parseDataColumns(inputString);
				}
				// Called Recurse Wallclock max min % of TOTAL UTR Overhead TOT_CYC e6/sec FP_OPS e6/sec FP_INS e6/sec OH (cyc)...
				else if (inputString.trim().startsWith("Called")) {
					previousLineBlank = false;
					// handle the header
					inData = true;
					isPAPI = true;
					parseDataColumns(inputString);
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
					// clear the event stack for the next thread
					while (!eventStack.isEmpty()) {
						eventStack.pop();
					}
					// save the thread data in the list
					dataList.add(data);
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
				} else if (!inData) { 
					previousLineBlank = false;
					// ignore this line
				} else if (inData) { 
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
							parent.children.add(eventData);
							eventData.callpathName = parent.callpathName + " => " + eventData.callpathName;
						}
						// if the just read event is at the same depth, pop the
						// current parent and replace it with the just read event
						else if (eventData.depth == parent.depth) {
							eventStack.pop();
							if (eventStack.empty()) {
								parent = null;
							} else {
								parent = eventStack.peek();
								parent.children.add(eventData);
								eventData.callpathName = parent.callpathName + " => " + eventData.callpathName;
							}
						}
						// if the just read event is at a shallower depth, pop
						// until we get to something which is at a higher level.
						else if (eventData.depth < parent.depth) {
							while (eventData.depth <= parent.depth) {
								eventStack.pop();
								if (eventStack.empty()) {
									parent = null;
									break;
								} else {
									parent = eventStack.peek();
								}
							}
							if (parent != null) {
								parent.children.add(eventData);
								eventData.callpathName = parent.callpathName + " => " + eventData.callpathName;
							}
						}
						// this is the new top level event
						eventStack.push(eventData);
						//System.out.println(eventData.callpathName);
					}
				}
			}
		} catch (IOException e) {
			System.out.println(e.getMessage());
			e.printStackTrace();
		}
		return dataList;
	}

	private void parseDataColumns(String inputString) {
		StringTokenizer st = new StringTokenizer(inputString, " \t\n\r");
		while (st.hasMoreTokens()) {
			String tmp = st.nextToken();
			this.dataColumns.add(tmp);
		}
	}

	public static String rtrim(String s) {
        int i = s.length()-1;
        while (i >= 0 && Character.isWhitespace(s.charAt(i))) {
            i--;
        }
        return s.substring(0,i+1);
    }
	
	private EventData processEventLine(String inputString) {
		EventData data = new EventData();
		inputString = rtrim(inputString);
		// process the inclusive values for this event
		String upToNCharacters = inputString.substring(0,60);
		// name
		data.name = upToNCharacters.trim();

		// we may have parsed an asterix... check for it!
		if (data.name.startsWith("*")) {
        	data.name = data.name.replaceFirst("\\*","").trim();
			// get the depth of this event
			data.depth = inputString.length() - inputString.replaceFirst("\\*","").trim().length();
		} else {
			// get the depth of this event
			data.depth = inputString.length() - inputString.trim().length();
		}
		data.callpathName = data.name;
		
		String theRest = inputString.substring(60,inputString.length());
		
        StringTokenizer st = new StringTokenizer(theRest, " \t\n\r()");
        // keep track of where we are in the data columns
		int skip = 0;
        for (String column : this.dataColumns) {
        	if (skip > 0) {
        		skip--;
        		continue;
        	}
        	if (column.equalsIgnoreCase("on")) {
        		// do nothing
                st.nextToken();
        	} else if (column.equalsIgnoreCase("called")) {
        		// num calls
                data.calls = Integer.parseInt(st.nextToken());
        	} else if (column.equalsIgnoreCase("recurse")) {
        		// recurse - what do do with this?
                st.nextToken();
        	} else if (column.equalsIgnoreCase("wallclock")) {
        		// Wallclock - the output is a total, so divide by the number of calls
        		data.inclusive.wallclock = Double.parseDouble(st.nextToken());
        	} else if (column.equalsIgnoreCase("max")) {
        		// WallclockMax
        		data.inclusive.wallclockMax = Double.parseDouble(st.nextToken());
        	} else if (column.equalsIgnoreCase("min")) {
        		// WallclockMin
        		data.inclusive.wallclockMin = Double.parseDouble(st.nextToken());
        	} else if (column.equalsIgnoreCase("%")) {
        		// % of total - ignore
                st.nextToken();
                skip = 2; // skip the next two column tokens, "of" and "total"
        	} else if (column.equalsIgnoreCase("utr")) {
        		// UTR Overhead
        		data.inclusive.utrOverhead = Double.parseDouble(st.nextToken());
        		skip = 1; // skip the next column token, "Overhead"
        	} else if (column.equalsIgnoreCase("OH")) {
        		// OH (cyc)
        		data.inclusive.ohCycles = Double.parseDouble(st.nextToken());
        		skip = 1;
        	} else {
        		// there are N PAPI events, and two measurements for each
        		data.inclusive.papi = new double[globalData.metrics.size()];
        		data.inclusive.papiE6OverSeconds = new double[globalData.metrics.size()];
        		for (int i = 0 ; i < globalData.metrics.size() ; i++) {
        			// PAPI - the output is a total, so divide by the number of calls
        			data.inclusive.papi[i] = Double.parseDouble(st.nextToken());
        			// e6/sec
        			data.inclusive.papiE6OverSeconds[i] = Double.parseDouble(st.nextToken());
        			skip += 2;
        		}
        	}
        }
		// TODO: how to deal with an event that happens at two call sites?
		return data;
	}

	private class GlobalData {
		@SuppressWarnings("unused")
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
		@SuppressWarnings("unused")
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