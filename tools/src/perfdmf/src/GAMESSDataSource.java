/**
 * 
 */
package edu.uoregon.tau.perfdmf;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;
import java.util.Vector;



/**
 * @author khuck
 *
 */
public class GAMESSDataSource extends DataSource {
    private int metric = 0;
    private Function function = null;
    private FunctionProfile functionProfile = null;
    private Node node = null;
    private Context context = null;
    private Thread thread = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private List v = null;
    boolean initialized = false;
    private Hashtable nodeHash = new Hashtable();
    private int threadCounter = 0;
	private File file = null;
	private int nodeCount = 0;
	private int coreCount = 0;
	private String moleculeName;
	private String basisSet;
	private String runType;
	private String scfType;
	private String accuracy;
	private String inputString;
	private StringBuffer phaseName;
	private double elapsedTime;
	private List events = new ArrayList();
	private double cpuTime;
	private double totalCpuTime = 0;
	private double totalWallClockTime = 0;
	private double currentWallClockTime = 0;
	private FunctionProfile fp;

    public GAMESSDataSource(File file) {
        super();
        this.setMetrics(new Vector());
		this.file = file;
    }

	/**
	 * 
	 */
	public GAMESSDataSource() {
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
     * @see edu.uoregon.tau.perfdmf.DataSource#cancelLoad()
     */
    public void cancelLoad() {
            return;
    }

    /* (non-Javadoc)
     * @see edu.uoregon.tau.perfdmf.DataSource#getProgress()
     */
    public int getProgress() {
            return 0;
    }

    /* (non-Javadoc)
	 * @see edu.uoregon.tau.perfdmf.DataSource#load()
	 */
	public void load() throws FileNotFoundException, IOException,
			DataSourceException, SQLException {
        try {
            long time = System.currentTimeMillis();

			nodeID = 0;
    		contextID = 0;
			threadID = 0;

			// parse metadata from filename
			parseFileName(this.file.getName());
			
            // parse the next file			
			FileInputStream fileIn = new FileInputStream(file);
			InputStreamReader inReader = new InputStreamReader(fileIn);
			BufferedReader br = new BufferedReader(inReader);

			while((inputString = br.readLine()) != null){
				inputString = inputString.trim();
				if (inputString.startsWith("FINAL RHF ENERGY IS")) {
					parseAccuracy();
				} else if (inputString.startsWith("....")) {
					if (!inputString.equals("...... PI ENERGY ANALYSIS ......") &&
						!inputString.equals("...... END OF PI ENERGY ANALYSIS ......"))
						parsePhaseName();
				} else if (inputString.startsWith("ON NODE")) {
					parseStepTime(true);
				} else if (inputString.startsWith("STEP CPU TIME =")) {
					parseStepTime(false);
				} else if (inputString.startsWith("TOTAL WALL CLOCK TIME")) {
					parseTime();
					createEvent();
				}
			}
	    	MyEvent mainEvent = new MyEvent("MAIN", this.totalCpuTime, this.totalWallClockTime, this.events.size());

			for (nodeID = 0 ; nodeID < this.nodeCount ; nodeID++) {
				for (threadID = 0 ; threadID < this.coreCount ; threadID++) {
					this.initializeThread();
					this.createFunction(this.thread, mainEvent, false);
					Iterator iter = this.events.iterator();
					while (iter.hasNext()) {
						MyEvent tmp = (MyEvent)iter.next();
						this.createFunction(this.thread, tmp, false);
						this.createFunction(this.thread, tmp, true);
					}
					this.setMetadata();
				}
			}
			
            time = (System.currentTimeMillis()) - time;
            //System.out.println("Time to process file (in milliseconds): " + time);
            this.generateDerivedData();
    		this.aggregateMetaData();
    		this.buildXMLMetaData();
			this.setGroupNamesPresent(true);
    	} catch (Exception e) {
            if (e instanceof DataSourceException) {
                throw (DataSourceException)e;
            } else {
                throw new DataSourceException(e);
            }
        }
	}

    private void createEvent() {
    	MyEvent event = new MyEvent(this.phaseName.toString(), this.cpuTime, this.currentWallClockTime, 0);
    	this.events.add(event);
	}

	private void parseTime() {
    	StringTokenizer st = new StringTokenizer(inputString, " ");
    	String dummy = st.nextToken(); // TOTAL
    	dummy = st.nextToken(); // WALL
    	dummy = st.nextToken(); // CLOCK
    	dummy = st.nextToken(); // TIME=
    	double tmp = Double.parseDouble(st.nextToken());
    	this.currentWallClockTime = tmp - this.totalWallClockTime;
       	this.totalWallClockTime = tmp;
	}

	private void parseStepTime(boolean mpi) {
    	StringTokenizer st = new StringTokenizer(inputString, " ");
    	String dummy = null;
    	if (mpi) {
	    	dummy = st.nextToken(); // ON
	    	dummy = st.nextToken(); // NODE
	    	dummy = st.nextToken(); // 0
    	}
    	dummy = st.nextToken(); // STEP
    	dummy = st.nextToken(); // CPU
    	dummy = st.nextToken(); // TIME or TIME=
    	if (!mpi) {
    		dummy = st.nextToken(); // =
    	}
    	this.cpuTime = Double.parseDouble(st.nextToken());
    	dummy = st.nextToken(); // TOTAL
    	dummy = st.nextToken(); // CPU
    	dummy = st.nextToken(); // TIME or TIME=
    	if (!mpi) {
    		dummy = st.nextToken(); // =
    	}
    	this.totalCpuTime = Double.parseDouble(st.nextToken());
	}

	private void parsePhaseName() {
    	StringTokenizer st = new StringTokenizer(inputString, " ");
    	String dummy = st.nextToken(); // "....."
    	String key = st.nextToken(); // "DONE" or "END"
    	this.phaseName = new StringBuffer();
    	if (key.equals("DONE")) {
    		// do nothing
    	} else { // key.equals("END")
    		String tmp = st.nextToken(); // "OF"
    	}
		while (st.hasMoreTokens()) {
			String tmp = st.nextToken();
			if (!tmp.startsWith("..."))  {
				this.phaseName.append(tmp);
				this.phaseName.append(" ");
			}
		}
	
	}

	private void parseAccuracy() {
    	StringTokenizer st = new StringTokenizer(inputString, " ");
    	String dummy = st.nextToken(); // FINAL
    	dummy = st.nextToken(); // RHF
    	dummy = st.nextToken(); // ENERGY
    	dummy = st.nextToken(); // IS
    	this.accuracy = st.nextToken();
	}

	public void initializeThread() {
    	// make sure we start at zero for all counters
    	nodeID = (nodeID == -1) ? 0 : nodeID;
    	contextID = (contextID == -1) ? 0 : contextID;
    	threadID = (threadID == -1) ? 0 : threadID;

        //Get the node,context,thread.
        node = this.getNode(nodeID);
        if (node == null)
            node = this.addNode(nodeID);
        context = node.getContext(contextID);
        if (context == null)
            context = node.addContext(contextID);
        thread = context.getThread(threadID);
        if (thread == null) {
            thread = context.addThread(threadID);
        }
        
        setMetadata();
    }

    private void setMetadata() {
   		this.getThread().getMetaData().put("molecule name", this.moleculeName);
   		this.getThread().getMetaData().put("basis set", this.basisSet);
   		this.getThread().getMetaData().put("run type", this.runType);
   		this.getThread().getMetaData().put("scf type", this.scfType);
   		this.getThread().getMetaData().put("node count", Integer.toString(this.nodeCount));
   		this.getThread().getMetaData().put("core count", Integer.toString(this.coreCount));
   		this.getThread().getMetaData().put("accuracy", this.accuracy);
	}

	public Thread getThread() {
        return thread;
    }

    private void parseFileName(String fileName) {
    	try {
	    	StringTokenizer st = new StringTokenizer(fileName, ".");
	   		moleculeName = st.nextToken();
			basisSet = st.nextToken();
			runType = st.nextToken();
			String token = st.nextToken();
	   		StringTokenizer st2 = new StringTokenizer(token, "-");
			scfType = st2.nextToken();
			token = st2.nextToken();
	   		StringTokenizer st3 = new StringTokenizer(token, "x");
	   		String tmp = st3.nextToken();
	   		this.nodeCount = Integer.parseInt(tmp);
	   		tmp = st3.nextToken();
	   		this.coreCount = Integer.parseInt(tmp);
    	} catch (Exception e) {
    		// filename wasn't what we expected.
    	}
    	return;
    }
    
	/** THe order that these are done is very important. */
	private void createFunction(Thread thread, MyEvent event, boolean callPath) {
		if (callPath) {
			this.function = addFunction("MAIN => " + event.name);
			this.function.addGroup(addGroup("TAU_CALLPATH"));
		} else {
			this.function = addFunction(event.name);
		}
		this.function.addGroup(addGroup("TAU_DEFAULT"));
		this.fp = new FunctionProfile (function, 3);
		thread.addFunctionProfile(this.fp);
		fp.setNumCalls(1);
		fp.setNumSubr(event.subroutines);
		String metric = "CPU TIME";
		Metric m = addMetric(metric, thread);
		if (event.name.equals("MAIN")) {
			fp.setInclusive(m.getID(), event.cpu);
			fp.setExclusive(m.getID(), 0.0);
		} else {
			fp.setInclusive(m.getID(), event.cpu);
			fp.setExclusive(m.getID(), event.cpu);
		}
		String metric2 = "Time";
		Metric m2 = addMetric(metric2, thread);
		if (event.name.equals("MAIN")) {
			fp.setInclusive(m2.getID(), event.wall);
			fp.setExclusive(m2.getID(), 0.0);
		} else {
			fp.setInclusive(m2.getID(), event.wall);
			fp.setExclusive(m2.getID(), event.wall);
		}
		String metric3 = "CPU UTILIZATION";
		Metric m3 = addMetric(metric3, thread);
		if (event.name.equals("MAIN")) {
			fp.setInclusive(m3.getID(), event.cpu/event.wall);
			fp.setExclusive(m3.getID(), 1.0);
		} else {
			fp.setInclusive(m3.getID(), event.cpu/event.wall);
			fp.setExclusive(m3.getID(), event.cpu/event.wall);
		}
	}
    
    private class MyEvent {
    	public String name = new String();
    	public double cpu = 0.0;
    	public double wall = 0.0;
    	public int subroutines = 0;
    	public MyEvent(String name, double cpu, double wall, int subroutines) {
    		this.name = name;
    		this.cpu = cpu * 1000000;
    		this.wall = wall * 1000000;
    		if (this.cpu > this.wall)
    			this.wall = this.cpu;
    	}
    }

}
