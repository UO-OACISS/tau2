import java.util.*;
import edu.uoregon.tau.dms.dss.*;

public class TestSave {

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {

		// Create a PerfDMFSession object
		DataSession session = new PerfDMFSession();
		session.initialize(args[0]);

		// select the application
		Application myApp = session.setApplication(1);
		if (myApp != null)
			System.out.println("Got application: " + myApp.getName() + ", version " + myApp.getVersion());

		myApp.setName("changed");
		session.saveApplication(myApp);

		// select an experiment
		Experiment myExp = session.setExperiment(1);
		if (myExp != null)
			System.out.println("Got experiment: " + myExp.getID());

		myExp.setName("changed");
		session.saveExperiment(myExp);

		// disconnect and exit.
		session.terminate();
		System.out.println ("Exiting.");
		return;
    }
}

