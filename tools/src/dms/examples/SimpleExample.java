import java.io.*;
import java.util.*;
import java.net.*;
import java.sql.*;
import dms.dss.*;

public class SimpleExample {

    public SimpleExample() {
		super();
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {

		// Create a PerfDBSession object
		DataSession session = new PerfDBSession();
		session.initialize(args[0]);

		// select the application
		Application myApp = session.setApplication("example", null);
		if (myApp != null)
			System.out.println("Got application: " + myApp.getName() + ", version " + myApp.getVersion());

		// select an experiment
		Experiment myExp = session.setExperiment(1);
		if (myExp != null)
			System.out.println("Got experiment: " + myApp.getID());

		// Get the list of trials
		ListIterator trials;
		trials = session.getTrialList();
		Trial myTrial = null;

		while (trials.hasNext()) {
			myTrial = (Trial)trials.next();
			session.setTrial(myTrial);
			break;
		}

		session.setNode(0);
		session.setContext(0);
		session.setThread(0);
		session.setMetric("time");
		session.setFunction(39);

		ListIterator myIterator;
		// Get the data
		System.out.println("Getting function data...");
		myIterator = session.getFunctionData();
		System.out.println(" ...done.");
		FunctionDataObject functionDataObject;
		Function function = null;
		String name, group;
		int functionIndexID, trial, node, context, thread;
		double inclusivePercentage, inclusive, exclusive, exclusivePercentage, inclusivePerCall;

		System.out.println ("Inclusive, Exclusive, Inc. Percent, Ex. Percent, Inc. Per. call:");
		System.out.println ("Trial, Node, Context, Thread, Name, Group, Values:");
		while (myIterator.hasNext()) {
			functionDataObject = (FunctionDataObject)(myIterator.next());
			functionIndexID = functionDataObject.getFunctionIndexID();
			function = session.getFunction(functionIndexID);
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

		// disconnect and exit.
		session.terminate();
		System.out.println ("Exiting.");
		return;
    }
}

