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
		DataSession session = new PerfDBSession(args[0]);
		session.open();

		// select the application
		Application myApp = session.setApplication("example", null);
		System.out.println("Got application: " + myApp.getName() + ", version " + myApp.getVersion());

		// select an experiment
		Experiment myExp = session.setExperiment(1);
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

		ListIterator myIterator;
		// Get the data
		System.out.print("Getting function data...");
		myIterator = session.getFunctionData();
		System.out.println(" done.");
		FunctionDataObject functionDataObject;
		Function function;
		String name, group;
		int functionIndexID, trial, node, context, thread;
		double inclusivePercentage;

		System.out.println ("Inclusive Percentages:");
		System.out.println ("Trial, Node, Context, Thread, Name, Group, Value:");
		while (myIterator.hasNext()) {
			functionDataObject = (FunctionDataObject)(myIterator.next());
			functionIndexID = functionDataObject.getFunctionIndexID();
			function = session.getFunction(functionIndexID);
			name = function.getName();
			group = function.getGroup();
			trial = function.getTrialID();
			node = functionDataObject.getNodeID();
			context = functionDataObject.getContextID();
			thread = functionDataObject.getThreadID();
			inclusivePercentage = functionDataObject.getInclusivePercentage();
			System.out.println (trial + ", " + node + ", " + context + ", " + thread + ", " + name + ", " + group + " = " + inclusivePercentage);

		}

		// disconnect and exit.
		session.close();
		System.out.println ("Exiting.");
		return;
    }
}

