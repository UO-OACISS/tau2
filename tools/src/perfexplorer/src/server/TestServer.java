package server;

import java.util.ListIterator;
import java.util.Vector;
//import java.lang.Thread;
import edu.uoregon.tau.perfdmf.*;
import common.*;

/**
 * This class exists as a unit test of the PerfExplorerServer class.
 *
 * <P>CVS $Id: TestServer.java,v 1.4 2005/11/02 22:16:53 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class TestServer {

	public static void main (String[] args) {
		System.out.println ("LIBRARY PATH: " + System.getProperty ("java.library.path"));
		try {
			//int engine = AnalysisTaskWrapper.RPROJECT_ENGINE;
			int engine = AnalysisTaskWrapper.WEKA_ENGINE;
			//int engine = AnalysisTaskWrapper.OCTAVE_ENGINE;
			PerfExplorerServer server = PerfExplorerServer.getServer(args[0], engine);
			System.out.println(server.sayHello());
			Object[] objects = new Object[4];
			ListIterator apps = server.getApplicationList().listIterator();
			Application app = null;
			while (apps.hasNext()) {
				app = (Application)apps.next();
				if (app.getID() == 2) {
					objects[0] = app;
					break;
				}
			}
			ListIterator exps =
			server.getExperimentList(app.getID()).listIterator();
			Experiment exp = null;
			while (exps.hasNext()) {
				exp = (Experiment)exps.next();
				if (exp.getID() == 2) {
					objects[1] = exp;
					break;
				}
			}
			ListIterator trials =
			server.getTrialList(exp.getID()).listIterator();
			Trial trial = null;
			while (trials.hasNext()) {
				trial = (Trial)trials.next();
				if (trial.getID() == 3) {
					objects[2] = trial;
					break;
				}
			}
			Vector metrics = trial.getMetrics();
			for (int i = 0 ; i < metrics.size() ; i++) {
				Metric metric = (Metric)metrics.elementAt(i);
				if (metric.getID() == 3) {
					objects[3] = metric;
					break;
				}
			}
			RMIPerfExplorerModel model = new RMIPerfExplorerModel();
			//model.setClusterMethod(RMIPerfExplorerModel.CORRELATION_ANALYSIS);
			model.setDimensionReduction(RMIPerfExplorerModel.OVER_X_PERCENT);
			model.setNumberOfClusters("2");
			model.setXPercent("2");
			model.setCurrentSelection(objects);
			String status = server.requestAnalysis(model, true);
			System.out.println(status);
			if (status.equals("Request already exists"))
				System.out.println(server.requestAnalysis(model, true));
			//java.lang.Thread.sleep(300000);
			//System.exit(0);
		} catch (Exception e) {
			System.err.println("TestServer exception: " +
							   e.getMessage());
			e.printStackTrace();
		}
	}
}

