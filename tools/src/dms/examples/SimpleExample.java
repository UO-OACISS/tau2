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
		Trial trial = null;

		while (trials.hasNext()) {
			trial = (Trial)trials.next();
			if (trial.getMetric().compareTo("time") == 0 ){
				session.setTrial(trial);
				break;
			}
		}

		ListIterator myIterator;
		// Get the data
		myIterator = session.getFunctionData();
		FunctionDataObject fundo;
		String name;
		int t, n, c, h;
		double inclusivePercentage;

		System.out.println ("Inclusive Percentages:");
		System.out.println ("Trial, Node, Context, Thread, Name, Value:");
		while (myIterator.hasNext()) {
			fundo = (FunctionDataObject)(myIterator.next());
			name = session.getFunction(fundo.getFunctionID()).getName();
			t = session.getFunction(fundo.getFunctionID()).getTrialID();
			n = fundo.getNodeID();
			c = fundo.getContextID();
			h = fundo.getThreadID();
			inclusivePercentage = fundo.getInclusivePercentage();
			System.out.println (t + ", " + n + ", " + c + ", " + t + ", " + name + " = " + fundo.getInclusivePercentage());

		}

		// disconnect and exit.
		session.close();
		System.out.println ("Exiting.");
		return;
    }
}

