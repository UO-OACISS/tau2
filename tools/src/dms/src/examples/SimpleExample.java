import java.util.*;
import edu.uoregon.tau.dms.dss.*;

public class SimpleExample {

    public SimpleExample() {
		super();
    }

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

		//session.setNode(0);
		//session.setContext(0);
		//session.setThread(0);
		//session.setMetric("time");
		session.setIntervalEvent(37);
		IntervalEvent nullFun = null;
		session.setIntervalEvent(nullFun);
		session.setIntervalEvent(39);

		ListIterator myIterator;
		// Get the data
		System.out.println("Getting function data...");
		myIterator = session.getIntervalEventData();
		System.out.println(" ...done.");
		IntervalLocationProfile functionDataObject;
		IntervalEvent function = null;
		String name, group;
		int functionIndexID, trial, node, context, thread;
		double inclusivePercentage, inclusive, exclusive, exclusivePercentage, inclusivePerCall;

		System.out.println ("Inclusive, Exclusive, Inc. Percent, Ex. Percent, Inc. Per. call:");
		System.out.println ("Trial, Node, Context, Thread, Name, Group, Values:");
		while (myIterator.hasNext()) {
			functionDataObject = (IntervalLocationProfile)(myIterator.next());
			functionIndexID = functionDataObject.getIntervalEventID();
			System.out.println("function ID:" + functionIndexID);
			function = session.getIntervalEvent(functionIndexID);
			name = function.getName();
			group = function.getGroup();
			trial = function.getTrialID();
			node = functionDataObject.getNode();
			context = functionDataObject.getContext();
			thread = functionDataObject.getThread();
			inclusivePercentage = functionDataObject.getInclusivePercentage();
			exclusivePercentage = functionDataObject.getExclusivePercentage();
			inclusive = functionDataObject.getInclusive();
			exclusive = functionDataObject.getExclusive();
			inclusivePerCall = functionDataObject.getInclusivePerCall();
			System.out.println (trial + ", " + node + ", " + context + ", " + thread + ", " + name + ", " + group + " = [" + inclusive + ", " + exclusive + ", " + inclusivePercentage + ", " + exclusivePercentage + ", " + inclusivePerCall +"]");
		}

		// get the user events
		System.out.println("Getting userEvent data...");
		myIterator = session.getAtomicEventData();
		System.out.println(" ...done.");
		AtomicLocationProfile userEventDataObject;
		AtomicEvent userEvent = null;
		int id;
		double sampleCount, minimumValue, maximumValue, meanValue, standardDeviation;

		System.out.println ("UserEVents:");
		System.out.println ("Trial, Node, Context, Thread, Name, Group, Count, Min, Max, Mean, StdDev:");
		while (myIterator.hasNext()) {
			userEventDataObject = (AtomicLocationProfile)(myIterator.next());
			id = userEventDataObject.getAtomicEventID();
			userEvent = session.getAtomicEvent(id);
			name = userEvent.getName();
			group = userEvent.getGroup();
			trial = userEvent.getTrialID();
			node = userEventDataObject.getNode();
			context = userEventDataObject.getContext();
			thread = userEventDataObject.getThread();
			sampleCount = userEventDataObject.getSampleCount();
			minimumValue = userEventDataObject.getMinimumValue();
			maximumValue = userEventDataObject.getMaximumValue();
			meanValue = userEventDataObject.getMeanValue();
			standardDeviation = userEventDataObject.getStandardDeviation();
			System.out.println (trial + ", " + node + ", " + context + ", " + thread + ", " + name + ", " + group + " = [" + sampleCount + ", " + minimumValue + ", " + maximumValue + ", " + meanValue + ", " + standardDeviation +"]");
		}

		// disconnect and exit.
		session.terminate();
		System.out.println ("Exiting.");
		return;
    }
}

