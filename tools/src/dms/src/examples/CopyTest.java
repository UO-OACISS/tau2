package examples;

import java.io.*;
import java.util.*;
import java.net.*;
import java.sql.*;
import dms.dss.*;

public class CopyTest {

    public CopyTest() {
		super();
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {

		// Create a PerfDBSession object
		DataSession session = new PerfDBSession();
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

		// select the first trial
		while (trials.hasNext()) {
			myTrial = (Trial)trials.next();
			session.setTrial(myTrial);
			break;
		}

		// get the metrics
		Vector metrics = myTrial.getMetrics();
		session.setMetric(metrics);

		// get the list of functions
		ListIterator functions;
		functions = session.getFunctions();
		Function function = null;

		// get the function details
		while (functions.hasNext()) {
			function = (Function)functions.next();
			session.getFunctionDetail(function);
		}

		ListIterator myIterator;
		// Get all the data
		System.out.println("Getting function data...");
		myIterator = session.getFunctionData();
		System.out.println(" ...done.");

		// get the user events
		System.out.println("Getting userEvent data...");
		myIterator = session.getUserEventData();
		System.out.println(" ...done.");

		// save the trial!
		System.out.println("Saving the trial data...");
		session.saveTrial();
		System.out.println(" ...done.");
		



		// disconnect and exit.
		session.terminate();
		System.out.println ("Exiting.");
		return;
    }
}

