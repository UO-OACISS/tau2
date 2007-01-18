package server;

import java.util.ListIterator;
import java.util.NoSuchElementException;
import java.util.Vector;

import common.AnalysisType;
import common.PerfExplorerOutput;
import common.RMIPerfExplorerModel;
import common.TransformationType;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;

/**
 * The implementation of the PerfExplorerServer Facade.
 * This class is declared as package private.
 */
class ScriptFacadeImpl implements ScriptFacade {
    /**
	 * 
	 */
	private RMIPerfExplorerModel model = new RMIPerfExplorerModel();
    private final PerfExplorerServer server;

    ScriptFacadeImpl(PerfExplorerServer server) {
		this.server = server;
    }

    /*
     * (non-Javadoc)
     * @see server.PerfExplorerServer.Facade#doSomething()
     */
    public void doSomething() {
        PerfExplorerOutput.println("Testing Script Facade");
        return;
    }

    /*
     * (non-Javadoc)
     * @see server.PerfExplorerServer.Facade#setApplication(java.lang.String)
     */
    public void setApplication(String name) {
        // check the argument
        if (name == null)
            throw new IllegalArgumentException("Application name cannot be null.");
        if (name.equals(""))
            throw new IllegalArgumentException("Application name cannot be an empty string.");

        boolean found = false;
        for (ListIterator apps = server.getApplicationList().listIterator();                 apps.hasNext() && !found; ) {
            Application app = (Application)apps.next();
            if (app.getName().equals(name)) {
                model.setCurrentSelection(app);;
                found = true;
            }
        }
        if (!found)
            throw new NoSuchElementException("Application '" + name + "' not found.");
    }

    /*
     * (non-Javadoc)
     * @see server.PerfExplorerServer.Facade#setExperiment(java.lang.String)
     */
    public void setExperiment(String name) {
        // check the argument
        if (name == null)
            throw new IllegalArgumentException("Experiment name cannot be null.");
        if (name.equals(""))
            throw new IllegalArgumentException("Experiment name cannot be an empty string.");

        Application app = model.getApplication();
        if (app == null)
            throw new NullPointerException("Application selection is null. Please select an Application before setting the Experiment.");
        boolean found = false;
        for (ListIterator exps = server.getExperimentList(app.getID()).listIterator();
             exps.hasNext() && !found;) {
            Experiment exp = (Experiment)exps.next();
            if (exp.getName().equals(name)) {
                model.setCurrentSelection(exp);
                found = true;
            }
        }
        if (!found)
            throw new NoSuchElementException("Experiment '" + name + "' not found.");
    }

    /*
     * (non-Javadoc)
     * @see server.PerfExplorerServer.Facade#setTrial(java.lang.String)
     */
    public void setTrial(String name) {
        // check the argument
        if (name == null)
            throw new IllegalArgumentException("Trial name cannot be null.");
        if (name.equals(""))
            throw new IllegalArgumentException("Trial name cannot be an empty string.");

        Experiment exp = model.getExperiment();
        if (exp == null)
            throw new NullPointerException("Experiment selection is null.  Please select an Experiment before setting the Trial.");
        boolean found = false;
        for (ListIterator trials = server.getTrialList(exp.getID()).listIterator();
             trials.hasNext() && !found;) {
            Trial trial = (Trial)trials.next();
            if (trial.getName().equals(name)) {
                model.setCurrentSelection(trial);
                found = true;
            }
        }
        if (!found)
            throw new NoSuchElementException("Trial '" + name + "' not found.");
    }

    /*
     * (non-Javadoc)
     * @see server.PerfExplorerServer.Facade#setMetric(java.lang.String)
     */
    public void setMetric(String name) {
        // check the argument
        if (name == null)
            throw new IllegalArgumentException("Metric name cannot be null.");
        if (name.equals(""))
            throw new IllegalArgumentException("Metric name cannot be an empty string.");

        Trial trial = model.getTrial();
        if (trial == null)
            throw new NullPointerException("Trial selection is null.  Please select a Trial before setting the Metric.");
        boolean found = false;
        Vector metrics = trial.getMetrics();
        for (int i = 0, size = metrics.size(); i < size && !found ; i++) {
            Metric metric = (Metric)metrics.elementAt(i);
            if (metric.getName().equals(name)) {
                model.setCurrentSelection(metric);
                found = true;
            }
        }
        if (!found)
            throw new NoSuchElementException("Metric '" + name + "' not found.");
    }

    /*
     * (non-Javadoc)
     * @see server.PerfExplorerServer.Facade#setDimensionReduction(common.TransformationType, java.lang.String)
     */
    public void setDimensionReduction(TransformationType type, String parameter) {
        if (type == null)
            throw new IllegalArgumentException("TransformationType type cannot be null.");

        model.setDimensionReduction(type);
        if (type == TransformationType.OVER_X_PERCENT) {
            if (parameter == null)
                throw new IllegalArgumentException("Object parameter cannot be null.");
            model.setXPercent(parameter);
        }
    }

    /*
     * (non-Javadoc)
     * @see server.PerfExplorerServer.Facade#setAnalysisType(common.AnalysisType)
     */
    public void setAnalysisType(AnalysisType type) {
        model.setClusterMethod(type);
    }

    /*
     * (non-Javadoc)
     * @see server.PerfExplorerServer.Facade#requestAnalysis()
     */
    public String requestAnalysis() {
        return server.requestAnalysis(model, true);
    }

    /*
     * (non-Javadoc)
     * @see server.ScriptFacade#DoANOVA()
     */
    public void DoANOVA() {
    	PerfExplorerOutput.println("Doing ANOVA");
    	return;
    }

    /*
     * (non-Javadoc)
     * @see server.ScriptFacade#Do3DCorrelationCube()
     */
	public void Do3DCorrelationCube() {
		// TODO Auto-generated method stub
		
	}

	/*
	 * (non-Javadoc)
	 * @see server.ScriptFacade#SetMaximumNumberOfClusters(int)
	 */
	public void SetMaximumNumberOfClusters(int max) {
		// TODO Auto-generated method stub
		
	}
}