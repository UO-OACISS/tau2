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
		int threadID = 0;
		for (Thread thread : threads) {
			for (int m = 0 ; m < source.getNumberOfMetrics() ; m++) {
				String metric = source.getMetric(m).getName();
				Iterator<Function> functions = source.getFunctions();
				while (functions.hasNext()) {
					Function function = functions.next();
					FunctionProfile fp = thread.getFunctionProfile(function);
					this.putExclusive(threadID, function.getName(), metric, fp.getExclusive(m));
					this.putInclusive(threadID, function.getName(), metric, fp.getInclusive(m));
					this.putCalls(threadID, function.getName(), fp.getNumCalls());
					this.putSubroutines(threadID, function.getName(), fp.getNumSubr());
				}
			}
			threadID++;
		}
	}
}
