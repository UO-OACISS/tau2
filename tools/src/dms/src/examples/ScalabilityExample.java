import java.util.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.analysis.*;

public class ScalabilityExample {

    public ScalabilityExample() {
		super();
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {

		// Create a PerfDMFSession object
		PerfDMFSession session = new PerfDMFSession();
		session.initialize(args[0]);

		// select an experiment
		Experiment myExp = session.setExperiment(5);
		if (myExp != null)
			System.out.println("Got experiment: " + myExp.getID());

		// Get the list of trials
		DataSessionIterator trials;
		trials = (DataSessionIterator)session.getTrialList();

		// Get The scalability output
		Scalability test = new Scalability(session);
		System.out.println("Exclusive data:");
		String function = null;
		ScalabilityResults results = test.exclusive(trials.vector(), function);
		System.out.println(results.toString(false));
		System.out.println("Inclusive data:");
		results = test.inclusive(trials.vector(), function);
		System.out.println(results.toString(false));

		// disconnect and exit.
		session.terminate();
		System.out.println ("Exiting.");
		return;
    }
}

