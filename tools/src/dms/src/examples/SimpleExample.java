
import java.util.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.analysis.*;

public class SimpleExample {

    /*** Beginning of main program. ***/
    public static void main(java.lang.String[] args) {
	try {

	    if (args.length < 1) {
		System.out.println ("Usage: SimpleExample perfdmf.cfg");
		System.exit(-1);
	    }

	    DatabaseAPI dbAPI = new DatabaseAPI();

	    // prompt for password if not given
	    dbAPI.initialize(args[0],true);

	    System.out.println ("API loaded...");

	    // Get the list of applications
	    Iterator applications;
	    applications = dbAPI.getApplicationList().iterator();
	    Application app = null;

	    // loop through all the applications, and print out some info
	    while (applications.hasNext()) {
		app = (Application) applications.next();
		System.out.println ("Application ID = " + app.getID() + ", name = " + app.getName());
	    }
	    
	    // the following code shows how to select applications -
	    // use one of the following methods.  You don't have to 
	    // do all of them, just one.
	    //

	    // select an application
	    dbAPI.setApplication(app);
	    // select an application, another way
	    dbAPI.setApplication(app.getID());
	    // select an application, yet another way
	    dbAPI.setApplication(app.getName(), null);

	    // Get the list of experiments
	    Iterator experiments;
	    experiments = dbAPI.getExperimentList().iterator();
	    Experiment exp = null;

	    while(experiments.hasNext()) {
		exp = (Experiment) experiments.next();
		System.out.println ("Experiment ID = " + exp.getID() + ", appid = " + exp.getApplicationID());
	    }

	    // select an experiment
	    dbAPI.setExperiment(exp);
	    // select an experiment, another way
	    dbAPI.setExperiment(exp.getID());

	    // Get the list of trials
	    Iterator trials;
	    trials = dbAPI.getTrialList().iterator();
	    Trial trial = null;
	    Vector tmpTrials = new Vector();

	    while(trials.hasNext()) {
		trial = (Trial) trials.next();
		tmpTrials.addElement(trial);
		System.out.println ("Trial ID = " + trial.getID() + ", Experiment ID = " + trial.getExperimentID() + ", appid = " + trial.getApplicationID());
	    }
	    
	    // select a trial
	    dbAPI.setTrial(trial.getID());
	    // select a trial, another way
	    dbAPI.setTrial(trial.getID());

	    // get the metric count
	    int metricCount = trial.getMetricCount();
	    System.out.println ("Metric count: " + metricCount);

	    // Get the list of functions
	    Iterator functions;
	    functions = dbAPI.getIntervalEvents().iterator();
	    IntervalEvent function = null;

	    while (functions.hasNext()) {
		function = (IntervalEvent) functions.next();
		System.out.println ("IntervalEvent Name = " + function.getName());
	    }

	    // select a function
	    dbAPI.setIntervalEvent(function.getID());

	    if (function != null) {
		// select a function, another way
		dbAPI.setIntervalEvent(function.getID());
	    }

	    // Get the list of user events
	    Iterator userEvents;
	    userEvents = dbAPI.getAtomicEvents().iterator();
	    AtomicEvent userEvent = null;

	    while (userEvents.hasNext()) {
		userEvent = (AtomicEvent) userEvents.next();
		System.out.println ("AtomicEvent Name = " + userEvent.getName());
		AtomicLocationProfile means = userEvent.getMeanSummary();
		AtomicLocationProfile totals = userEvent.getTotalSummary();
		if (means.getMeanValue() != 0.0) {
		    System.out.print ("AtomicEvent Mean Value: Average = " + means.getMeanValue());
		    System.out.println (", Total = " + totals.getMeanValue());
		} else {
		    System.out.print ("AtomicEvent Sample Count: Average = " + means.getSampleCount());
		    System.out.println (", Total = " + totals.getSampleCount());
		}
	    }
	    
	    // select a userEvent
	    dbAPI.setAtomicEvent(userEvent.getID());
	    // select a userEvent, another way
	    if (userEvent != null)
		dbAPI.setAtomicEvent(userEvent.getID());

	    dbAPI.terminate();
	    System.out.println ("Exiting.");

	} catch (Exception e) {
	    e.printStackTrace();
	}
	return;
    }
}

