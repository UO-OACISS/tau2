package examples;

import java.util.*;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.analysis.*;

public class TestPerfDMFSession {

    public TestPerfDMFSession() {
		super();
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
		try {

		// Create a PerfDMFSession object
		DataSession session = new PerfDMFSession();
		session.initialize(args[0]);
		System.out.println ("API loaded...");

		// Get the list of applications
		ListIterator applications;
		applications = session.getApplicationList();
		Application app = null;

		// loop through all the applications, and print out some info
        while(applications.hasNext())
		{
			app = (Application) applications.next();
			System.out.println ("Application ID = " + app.getID() + ", name = " + app.getName() + ", version = " + app.getVersion() + ", description = " + app.getDescription());
		}

		// the following code shows how to select applications -
		// use one of the following methods.  You don't have to 
		// do all of them, just one.
		//

		// select an application
		session.setApplication(app);
		// select an application, another way
		session.setApplication(app.getID());
		// select an application, yet another way
		session.setApplication(app.getName(), null);

		// Get the list of experiments
		ListIterator experiments;
		experiments = session.getExperimentList();
		Experiment exp = null;

        while(experiments.hasNext())
		{
			exp = (Experiment) experiments.next();
			System.out.println ("Experiment ID = " + exp.getID() + ", appid = " + exp.getApplicationID());
		}

		// select an experiment
		session.setExperiment(exp);
		// select an experiment, another way
		session.setExperiment(exp.getID());

		// Get the list of trials
		ListIterator trials;
		trials = session.getTrialList();
		Trial trial = null;
		Vector tmpTrials = new Vector();

        while(trials.hasNext())
		{
			trial = (Trial) trials.next();
			tmpTrials.addElement(trial);
			System.out.println ("Trial ID = " + trial.getID() + ", Experiment ID = " + trial.getExperimentID() + ", appid = " + trial.getApplicationID());
		}

		// select a trial
		session.setTrial(trial);
		// select a trial, another way
		session.setTrial(trial.getID());

		// get the metric count
		int metricCount = trial.getMetricCount();
		System.out.println ("Metric count: " + metricCount);

		// Get the list of functions
		ListIterator functions;
		functions = session.getIntervalEvents();
		IntervalEvent function = null;

		while (functions.hasNext())
		{
			function = (IntervalEvent) functions.next();
			System.out.println ("IntervalEvent Name = " + function.getName());
		}

		// select a function
		session.setIntervalEvent(function);

		if (function != null) {
			// select a function, another way
			session.setIntervalEvent(function.getID());
		}

		// Get the list of user events
		ListIterator userEvents;
		userEvents = session.getAtomicEvents();
		AtomicEvent userEvent = null;

		while (userEvents.hasNext())
		{
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
		session.setAtomicEvent(userEvent);
		// select a userEvent, another way
		if (userEvent != null)
			session.setAtomicEvent(userEvent.getID());

		Vector nodes = new Vector();
		Integer node = new Integer(0);
		nodes.addElement(node);
		node = new Integer(1);
		nodes.addElement(node);
		session.setNode(nodes);
		session.setContext(0);
		session.setThread(0);

		// Get the data
		session.getIntervalEventData();

		// Get the data
		session.getAtomicEventData();

		// test out the analysis!
		trial = session.setTrial(1);
		Vector metrics = session.getMetrics();
		Metric metric = (Metric)(metrics.elementAt(0));
		Distance distance = new Distance((PerfDMFSession)session, trial, metric);
		double[][] matrix = distance.getEuclidianDistance();
		System.out.println("Euclidian distance:");
		for (int i = 0 ; i < distance.getThreadCount(); i++ ) {
			System.out.print("thread " + i + ": ");
			for (int j = 0 ; j < distance.getEventCount(); j++ ) {
				if (j > 0) System.out.print(", ");
				System.out.print(matrix[i][j]);
			}
			System.out.println("");
		}
		matrix = distance.getManhattanDistance();
		System.out.println("Manhattan distance:");
		for (int i = 0 ; i < distance.getThreadCount(); i++ ) {
			System.out.print("thread " + i + ": ");
			for (int j = 0 ; j < distance.getEventCount(); j++ ) {
				if (j > 0) System.out.print(", ");
				System.out.print(matrix[i][j]);
			}
			System.out.println("");
		}
		// disconnect and exit.
		session.terminate();
		System.out.println ("Exiting.");

		} catch (Exception e) {
			e.printStackTrace();
		}
		return;
    }
}

