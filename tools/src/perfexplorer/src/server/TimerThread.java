package server;

import java.util.Timer;


/**
 * This is the main server thread which processes long-executing analysis 
 * requests.  It is created by the PerfExplorerServer object, and 
 * checks the queue every 1 seconds to see if there are any new requests.
 *
 * <P>CVS $Id: TimerThread.java,v 1.2 2005/09/27 19:46:32 khuck Exp $</P>
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
	private int analysisEngine = AnalysisTaskWrapper.WEKA_ENGINE;

	/**
	 * Constructor.  Expects a reference to a PerfExplorerServer.
	 * @param server
	 */
	public TimerThread (PerfExplorerServer server, int analysisEngine) {
		this.server = server;
		this.analysisEngine = analysisEngine;
	}

	/**
	 * run method.  When the thread wakes up, this method is executed.
	 * This method creates an AnalysisTask object, and schedules it to
	 * execute after a delay of 1 second.  After the task is completed,
	 * it is repeated every 1 seconds.  If there is no work to be done,
	 * the analysisTask returns immediately.
	 */
	public void run() {
		// use this line for the new processing
		AnalysisTaskWrapper analysisTask = new
		AnalysisTaskWrapper(this.server, this.analysisEngine);
		// use this line for the old processing
		//AnalysisTask analysisTask = new AnalysisTask(this.server);

		// 
		schedule(analysisTask, 1000, 1000);
	}
}
