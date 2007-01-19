package client;

import java.util.ListIterator;
import java.util.NoSuchElementException;
import java.util.Vector;


import common.AnalysisType;
import common.PerfExplorerOutput;
import common.RMIPerfExplorerModel;
import common.RMIVarianceData;
import common.TransformationType;
import common.AnalysisType;
import common.TransformationType;
import common.EngineType;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * The facade interface to the application for scripting purpose. This facade
 * allows a limited and easy access from user scripts With this facade user do
 * not have to traverse the object containment hierarchy. Also because the
 * subsystems are not exposed, scripts are limited in what they can do.
 */
public class ScriptFacade {
	private final PerfExplorerConnection connection;
	private final PerfExplorerModel model;

    public ScriptFacade() {
		connection = PerfExplorerConnection.getConnection();
		model = PerfExplorerModel.getModel();
    }

	public ScriptFacade(String configFile, EngineType analysisEngine) {
		PerfExplorerOutput.setQuiet(false);
		PerfExplorerConnection.setStandalone(true);
		PerfExplorerConnection.setConfigFile(configFile);
		PerfExplorerConnection.setAnalysisEngine(analysisEngine);
		connection = PerfExplorerConnection.getConnection();
		model = PerfExplorerModel.getModel();
	}

	/**
	 * Test method for the facade class.
	 * 
	 */
    public void doSomething() {
        PerfExplorerOutput.println("Testing Script Facade");
        return;
    }

    /**
     * Exit the application.
     * 
     */
    public void exit() {
    	System.exit(0);
    }
    
	/**
	 * Set the focus on the application specified.
	 * 
	 * @param name
	 */
    public void setApplication(String name) {
        // check the argument
        if (name == null)
            throw new IllegalArgumentException("Application name cannot be null.");
        if (name.equals(""))
            throw new IllegalArgumentException("Application name cannot be an empty string.");

        boolean found = false;
        for (ListIterator apps = connection.getApplicationList(); apps.hasNext() && !found; ) {
            Application app = (Application)apps.next();
            if (app.getName().equals(name)) {
                model.setCurrentSelection(app);;
                found = true;
            }
        }
        if (!found)
            throw new NoSuchElementException("Application '" + name + "' not found.");
    }

	/**
	 * Set the focus on the experiment specified.
	 * 
	 * @param name
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
        for (ListIterator exps = connection.getExperimentList(app.getID());
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

	/**
	 * Set the focus on the trial specified.
	 * 
	 * @param name
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
        for (ListIterator trials = connection.getTrialList(exp.getID());
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

	/**
	 * Set the focus on the metric specified.
	 * 
	 * @param name
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

	/**
	 * Choose the dimension reduction method.
	 * 
	 * @param type
	 * @param parameter
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

	/**
	 * Set the analysis method.
	 * 
	 * @param type
	 */
    public void setAnalysisType(AnalysisType type) {
        model.setClusterMethod(type);
    }

	/**
	 * Request the analysis configured.
	 * 
	 * @return
	 */
    public String requestAnalysis() {
        return connection.requestAnalysis(model, true);
    }

	/**
	 * Request the ANOVA results.
	 * 
	 */
    public void doANOVA() {
    	PerfExplorerOutput.println("Doing ANOVA");
    	return;
    }

	/**
	 * Request a 3D view of correlation data
	 *
	 */
	public void do3DCorrelationCube() {
		// TODO Auto-generated method stub
		
	}

	/**
	 * Set the maximum number of clusters for cluster analysis
	 * 
	 * @param max
	 */
	public void setMaximumNumberOfClusters(int max) {
		// TODO Auto-generated method stub
		
	}
	
	/**
	 * Create boxcharts, and display them
	 *
	 */
	public void createBoxChart() {
		// TODO Auto-generated method stub
		
	}
	
	/**
	 * Create the data histograms
	 *
	 */
	public void createHistograms() {
		// TODO Auto-generated method stub
		
	}
	
	/**
	 * Create the normal probability chart
	 *
	 */
	public void createNormalProbabilityChart() {
		// TODO Auto-generated method stub
		
	}

	/**
	 * Show the summary of the profile data
	 *
	 */
	public void showDataSummary() {
		// get the data
		RMIVarianceData data = connection.requestVariationAnalysis(model);
		PerfExplorerOutput.print(UtilFncs.pad(data.getValueName(0), 25).toUpperCase() + " " );
		for (int i = 1 ; i < data.getValueCount() ; i++) {
			PerfExplorerOutput.print(UtilFncs.lpad(data.getValueName(i), 11).toUpperCase() + " " );
		}
		PerfExplorerOutput.print("\n");
		Object[][] matrix = data.getDataMatrix();
		int maxSize = 25;
		for (int i = 0 ; i < data.getEventCount() ; i++) {
			for (int j = 0 ; j < data.getValueCount() ; j++) {
				if (matrix[i][j] instanceof String) {
					String s = (String)matrix[i][j];
	      			s = (s.length() > maxSize ? s.substring(0, maxSize) : s);
					PerfExplorerOutput.print(UtilFncs.pad(s, maxSize) + " " );
				}
				else if (matrix[i][j] instanceof Double) {
					Double d = (Double)matrix[i][j];
					PerfExplorerOutput.print(UtilFncs.lpad(UtilFncs.formatDouble(d.doubleValue(), 10, true), 11) + " ");
				}
			}
			PerfExplorerOutput.print("\n");
		}
	}

}