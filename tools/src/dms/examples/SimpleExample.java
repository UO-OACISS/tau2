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
		session.setApplication("example");

		// select an experiment
		session.setExperiment(1);

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

		while (myIterator.hasNext()) {
			fundo = (FunctionDataObject)(myIterator.next());
			System.out.println ("inclusive percentage = " + fundo.getInclusivePercentage());

		}

		// disconnect and exit.
		session.close();
		System.out.println ("Exiting.");
		return;
    }
}

