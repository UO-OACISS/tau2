import java.util.*;
import edu.uoregon.tau.dms.dss.*;

public class TestDelete {

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {

		// Create a PerfDMFSession object
		DataSession session = new PerfDMFSession();
		session.initialize(args[0]);

		// select the application
		Application myApp = session.setApplication(1);
		if (myApp != null)
			System.out.println("Got application: " + myApp.getName() + ", version " + myApp.getVersion());

		// select an experiment
		Experiment myExp = session.setExperiment(1);
		if (myExp != null)
			System.out.println("Got experiment: " + myExp.getID());

		// Get the list of trials
		ListIterator trials;
		trials = session.getTrialList();
		Trial myTrial = null;

		while (trials.hasNext()) {
			myTrial = (Trial)trials.next();
			session.setTrial(myTrial);
			break;
		}

		// delete the application
		session.deleteApplication(myApp.getID());

		// delete the experiment
		// session.deleteExperiment(myExp.getID());

		// delete the trial
		// session.deleteTrial(myTrial.getID());

		// disconnect and exit.
		session.terminate();
		System.out.println ("Exiting.");
		return;
    }
}

