/**
 * 
 */
package edu.uoregon.tau.perfexplorer.glue;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.sql.SQLException;
import java.util.Iterator;
import java.util.List;

import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.*;

/**
 * @author khuck
 *
 */
public class DataSourceResult extends AbstractResult {
	
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
    public static final int GYRO = DataSource.GYRO;
    public static final int GAMESS = DataSource.GAMESS;

	/**
	 * 
	 */
	public DataSourceResult() {
		super();
	}

	/**
	 * @param input
	 */
	public DataSourceResult(PerformanceResult input) {
		super(input);
	}

	@SuppressWarnings("unchecked")
	public DataSourceResult(int fileType, String[] sourceFiles, boolean fixNames) {
        File[] files = new File[sourceFiles.length];
        for (int i = 0; i < sourceFiles.length; i++) {
            files[i] = new File(sourceFiles[i]);
    		this.name = files[i].getName();
        }
        DataSource source = UtilFncs.initializeDataSource(files, fileType, fixNames);
        try{
        	source.load();
        } catch (Exception e) {
        	System.err.println(e.getMessage());
        	e.printStackTrace(System.err);
        	return;
        }
		List<Thread> threads = source.getThreads();
		int tid = 0;  // thread ID
		for (Thread thread : threads) {
			for (int m = 0 ; m < source.getNumberOfMetrics() ; m++) {
				String metric = source.getMetric(m).getName();
				Iterator<Function> functions = source.getFunctions();
				while (functions.hasNext()) {
					Function function = functions.next();
					String name = function.getName();
					FunctionProfile fp = thread.getFunctionProfile(function);
					if (fp != null) {
						this.putExclusive(tid, name, metric, fp.getExclusive(m));
						this.putInclusive(tid, name, metric, fp.getInclusive(m));
						this.putCalls(tid, name, fp.getNumCalls());
						this.putSubroutines(tid, name, fp.getNumSubr());
					}
				}
			}
			Iterator<UserEvent> userEvents = source.getUserEvents();
			while (userEvents.hasNext()) {
				UserEvent userEvent = userEvents.next();
				String name = userEvent.getName();
				UserEventProfile uep = thread.getUserEventProfile(userEvent);
				if (uep != null) {
					this.putUsereventMax(tid, name, uep.getMaxValue());
					this.putUsereventMean(tid, name, uep.getMeanValue());
					this.putUsereventMin(tid, name, uep.getMinValue());
					this.putUsereventNumevents(tid, name, uep.getNumSamples());
					this.putUsereventSumsqr(tid, name, uep.getSumSquared());
				}
			}
			tid++;
		}
	}
}
