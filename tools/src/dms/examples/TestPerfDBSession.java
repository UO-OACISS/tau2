import java.io.*;
import java.util.*;
import java.net.*;
import java.sql.*;
import dms.dss.*;

public class TestPerfDBSession {

    public TestPerfDBSession() {
		super();
    }

    /*** Beginning of main program. ***/

    public static void main(java.lang.String[] args) {
		// Create a PerfDBSession object
		DataSession session = new PerfDBSession(args[0]);
		session.open();
		System.out.println ("API loaded...");

		// Get the list of applications
		Vector applications;
		applications = session.getAppList();
		Application app = null;

        for(Enumeration en = applications.elements(); en.hasMoreElements() ;)
		{
			app = (Application) en.nextElement();
			System.out.println ("Application ID = " + app.getID() + ", name = " + app.getName() + ", version = " + app.getVersion() + ", description = " + app.getDescription());

			// select an application
			session.setApplication(app);
			// select an application, another way
			session.setApplication(app.getID());
			// select an application, yet another way
			session.setApplication(app.getName());
		}

		// Get the list of experiments
		Vector experiments;
		experiments = session.getExpList();
		Experiment exp = null;

        for(Enumeration en = experiments.elements(); en.hasMoreElements() ;)
		{
			exp = (Experiment) en.nextElement();
			System.out.println ("Experiment ID = " + exp.getID() + ", appid = " + exp.getApplicationID());

			// select an experiment
			session.setExperiment(exp);
			// select an experiment, another way
			session.setExperiment(exp.getID());
		}

		// Get the number of nodes, without selecting a trial
		int numNodes = session.getNumberOfNodes();
		System.out.println ("NumNodes = " + numNodes);

		// Get the number of contexts, without selecting a trial
		int numContexts = session.getNumberOfContexts();
		System.out.println ("NumContexts = " + numContexts);

		// Get the number of threads, without selecting a trial
		int numThreads = session.getNumberOfThreads();
		System.out.println ("NumThreads = " + numThreads);

		// Get the list of trials
		Vector trials;
		trials = session.getTrialList();
		Trial trial = null;

        for(Enumeration en = trials.elements(); en.hasMoreElements() ;)
		{
			trial = (Trial) en.nextElement();
			System.out.println ("Trial ID = " + trial.getID() + ", Experiment ID = " + trial.getExperimentID() + ", appid = " + trial.getApplicationID());

			// select a trial
			session.setTrial(trial);
			// select a trial, another way
			session.setTrial(trial.getID());
		}

		// Get the number of nodes
		numNodes = session.getNumberOfNodes();
		System.out.println ("NumNodes = " + numNodes);

		// Get the number of contexts
		numContexts = session.getNumberOfContexts();
		System.out.println ("NumContexts = " + numContexts);

		// Get the number of threads
		numThreads = session.getNumberOfThreads();
		System.out.println ("NumThreads = " + numThreads);

		// Get the list of functions
		Vector functions;
		functions = session.getFunctions();
		Function function = null;

        for(Enumeration en = functions.elements(); en.hasMoreElements() ;)
		{
			function = (Function) en.nextElement();
			// System.out.println ("Function Index ID = " + function.getIndexID() + ", Function ID = " + function.getFunctionID());
			System.out.println ("Function Name = " + function.getName());
			// System.out.println ("Trial ID = " + function.getTrialID() + ", Experiment ID = " + function.getExperimentID() + ", appid = " + function.getApplicationID());
			// System.out.println ("Mean Summary = " + function.getMeanSummary().getInclusivePercentage() + ", " + function.getMeanSummary().getInclusive() + ", " + function.getMeanSummary().getExclusivePercentage() + ", " + function.getMeanSummary().getExclusive() + ", " + function.getMeanSummary().getNumCalls() + ", " + function.getMeanSummary().getNumSubroutines() + ", " + function.getMeanSummary().getInclusivePerCall());
			// System.out.println ("Total Summary = " + function.getTotalSummary().getInclusivePercentage() + ", " + function.getTotalSummary().getInclusive() + ", " + function.getTotalSummary().getExclusivePercentage() + ", " + function.getTotalSummary().getExclusive() + ", " + function.getTotalSummary().getNumCalls() + ", " + function.getTotalSummary().getNumSubroutines() + ", " + function.getTotalSummary().getInclusivePerCall());

			// select a function
			session.setFunction(function);
			// select a function, another way
			session.setFunction(function.getIndexID());
		}

		session.setFunction(functions);

		Vector nodes = new Vector();
		Integer node = new Integer(0);
		nodes.addElement(node);
		node = new Integer(1);
		nodes.addElement(node);
		session.setNode(nodes);
		session.setContext(0);
		session.setThread(0);

		// Get the data
		session.getFunctionData();

		// disconnect and exit.
		session.close();
		System.out.println ("Exiting.");
		return;
    }
}

