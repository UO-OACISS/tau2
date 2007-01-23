package server;

import common.EngineType;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import java.util.Timer;


/**
 * This is the main server thread which processes long-executing analysis 
 * requests.  It is created by the PerfExplorerServer object, and 
 * checks the queue every 1 seconds to see if there are any new requests.
 *
 * <P>CVS $Id: TimerThread.java,v 1.6 2007/01/23 18:46:29 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 * @see     PerfExplorerServer
 */
public class TimerThread extends Timer implements Runnable {

	/**
	 *  reference to the server which created this thread
	 */
	private PerfExplorerServer server = null;
	private DatabaseAPI workerSession = null;

	/**
	 * Constructor.  Expects a reference to a PerfExplorerServer.
	 * @param server
	 */
	TimerThread (PerfExplorerServer server, DatabaseAPI workerSession) {
		this.server = server;
		this.workerSession = workerSession;
	}

	/**
	 * run method.  When the thread wakes up, this method is executed.
	 * This method creates an AnalysisTask object, and schedules it to
	 * execute after a delay of 1 second.  After the task is completed,
	 * it is repeated every 1 seconds.  If there is no work to be done,
	 * the analysisTask returns immediately.
	 */
	public void run() {
		AnalysisTask analysisTask = new
		AnalysisTask(this.server, this.workerSession);
		schedule(analysisTask, 1000, 1000);
	}
}
