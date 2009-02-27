package edu.uoregon.tau.perfexplorer.client;

import java.util.ListIterator;
import java.util.NoSuchElementException;
import java.util.Vector;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.IntervalEvent;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.UtilFncs;
import edu.uoregon.tau.perfexplorer.common.AnalysisType;
import edu.uoregon.tau.perfexplorer.common.EngineType;
import edu.uoregon.tau.perfexplorer.common.PerfExplorerOutput;
import edu.uoregon.tau.perfexplorer.common.RMISortableIntervalEvent;
import edu.uoregon.tau.perfexplorer.common.RMIVarianceData;
import edu.uoregon.tau.perfexplorer.common.TransformationType;

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
    public Application setApplication(String name) {
        Application app = null;
        // check the argument
        if (name == null)
            throw new IllegalArgumentException("Application name cannot be null.");
        if (name.equals(""))
            throw new IllegalArgumentException("Application name cannot be an empty string.");

        boolean found = false;
        for (ListIterator<Application> apps = connection.getApplicationList(); apps.hasNext() && !found; ) {
            app = apps.next();
            if (app.getName().equals(name)) {
                model.setCurrentSelection(app);;
                found = true;
            }
        }
        if (!found)
            throw new NoSuchElementException("Application '" + name + "' not found.");
		return app;
    }

	/**
	 * Set the focus on the experiment specified.
	 * 
	 * @param name
	 */
    public Experiment setExperiment(String name) {
		Experiment exp = null;
        // check the argument
        if (name == null)
            throw new IllegalArgumentException("Experiment name cannot be null.");
        if (name.equals(""))
            throw new IllegalArgumentException("Experiment name cannot be an empty string.");

        Application app = model.getApplication();
        if (app == null)
            throw new NullPointerException("Application selection is null. Please select an Application before setting the Experiment.");
        boolean found = false;
        for (ListIterator<Experiment> exps = connection.getExperimentList(app.getID());
             exps.hasNext() && !found;) {
            exp = exps.next();
            if (exp.getName().equals(name)) {
                model.setCurrentSelection(exp);
                found = true;
            }
        }
        if (!found)
            throw new NoSuchElementException("Experiment '" + name + "' not found.");
		return exp;
    }

	/**
	 * Set the focus on the trial specified.
	 * 
	 * @param name
	 */
    public Trial setTrial(String name) {
		Trial trial = null;
        // check the argument
        if (name == null)
            throw new IllegalArgumentException("Trial name cannot be null.");
        if (name.equals(""))
            throw new IllegalArgumentException("Trial name cannot be an empty string.");

        Experiment exp = model.getExperiment();
        if (exp == null)
            throw new NullPointerException("Experiment selection is null.  Please select an Experiment before setting the Trial.");
        boolean found = false;
        for (ListIterator<Trial> trials = connection.getTrialList(exp.getID());
             trials.hasNext() && !found;) {
            trial = trials.next();
            if (trial.getName().equals(name)) {
                model.setCurrentSelection(trial);
                found = true;
            }
        }
        if (!found)
            throw new NoSuchElementException("Trial '" + name + "' not found.");
		return trial;
    }

	/**
	 * Set the focus on the metric specified.
	 * 
	 * @param name
	 */
    @SuppressWarnings("unchecked") // for getMetrics() call
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
        Vector<Metric> metrics = trial.getMetrics();
        for (int i = 0, size = metrics.size(); i < size && !found ; i++) {
            Metric metric = metrics.elementAt(i);
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
		// print the first column heading, the event name
		PerfExplorerOutput.print(UtilFncs.pad(data.getValueName(0), 25).toUpperCase() + " " );
		// for each of the other column headings, output them padded to 11 characters
		for (int i = 1 ; i < data.getValueCount() ; i++) {
			PerfExplorerOutput.print(UtilFncs.lpad(data.getValueName(i), 11).toUpperCase() + " " );
		}
		PerfExplorerOutput.println("\n--------------------------------------------------------------------------------------------------------------------------------");

		// get the data matrix
		Object[][] matrix = data.getDataMatrix();
		// set the maximum length of the event name
		int maxSize = 25;
		// for each event...
		for (int i = 0 ; i < data.getEventCount() ; i++) {
			// first, we need to check for any dimension reduction.
			if (model.getDimensionReduction() == TransformationType.OVER_X_PERCENT) {
				// if this event is not more than X percent of the total, don't output it.
				Double d = (Double)matrix[i][2];
				if (d.doubleValue() < model.getXPercent())
					continue;
			}
			// for each data point...
			for (int j = 0 ; j < data.getValueCount() ; j++) {
				// if this is the event name, then output it, padded to no more than maxSize length
				if (matrix[i][j] instanceof String) {
					String s = (String)matrix[i][j];
	      			s = (s.length() > maxSize ? s.substring(0, maxSize) : s);
					PerfExplorerOutput.print(UtilFncs.pad(s, maxSize).toUpperCase() + " " );
				}
				// else, this is a value
				else if (matrix[i][j] instanceof Double) {
					Double d = (Double)matrix[i][j];
					PerfExplorerOutput.print(UtilFncs.lpad(UtilFncs.formatDouble(d.doubleValue(), 10, true), 11) + " ");
				}
			}
			PerfExplorerOutput.print("\n");
		}
		PerfExplorerOutput.print("\n");
	}

	public Object[] getTrialList(String criteria) {
        return connection.getTrialList(criteria).toArray();
	}

	public void doGeneralChart() {
		PerfExplorerChart.doGeneralChart();
	}

	/**
	 * Set the focus on the experiment specified.
	 * 
	 * @param name
	 */
    public void addApplication(String name) {
        // check the argument
        if (name == null)
            throw new IllegalArgumentException("Application name cannot be null.");
        if (name.equals(""))
            throw new IllegalArgumentException("Application name cannot be an empty string.");

        boolean found = false;
        for (ListIterator<Application> apps = connection.getApplicationList();
             apps.hasNext() && !found;) {
            Application app = apps.next();
            if (app.getName().equals(name)) {
                model.addSelection(app);
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
    public void addExperiment(String name) {
        // check the argument
        if (name == null)
            throw new IllegalArgumentException("Experiment name cannot be null.");
        if (name.equals(""))
            throw new IllegalArgumentException("Experiment name cannot be an empty string.");

        Application app = model.getApplication();
        if (app == null)
            throw new NullPointerException("Application selection is null. Please select an Application before setting the Experiment.");
        boolean found = false;
        for (ListIterator<Experiment> exps = connection.getExperimentList(app.getID());
             exps.hasNext() && !found;) {
            Experiment exp = exps.next();
            if (exp.getName().equals(name)) {
                model.addSelection(exp);
                found = true;
            }
        }
        if (!found)
            throw new NoSuchElementException("Experiment '" + name + "' not found.");
    }

    public void setEventName(String eventName) {
    	model.setEventName(eventName);
    }

    public void addEventName(String eventName) {
    	model.addEventName(eventName);
    }

    public void setGroupName(String groupName) {
    	model.setGroupName(groupName);
    }

    public void addGroupName(String groupName) {
    	model.addGroupName(groupName);
    }

    public void setMetricName(String metricName) {
    	model.setMetricName(metricName);
    }

    //public void addMetricName(String metricName) {
    	//model.addMetricName(metricName);
    //}
    
	public void setChartMetadataFieldName(String fieldName) {
		model.setChartMetadataFieldName(fieldName);
	}

	public void setChartMetadataFieldValue(String fieldValue) {
		model.setChartMetadataFieldValue(fieldValue);
	}

	public void setChartSeriesName(String seriesName) {
		model.setChartSeriesName(seriesName);
	}

	public void setChartXAxisName(String xAxisColumnName, String xAxisLabel) {
		model.setChartXAxisName(xAxisColumnName, xAxisLabel);
	}

	public void setChartYAxisName(String yAxisColumnName, String yAxisLabel) {
		model.setChartYAxisName(yAxisColumnName, yAxisLabel);
	}

	public void setChartTitle(String title) {
		model.setChartTitle(title);
	}

	public void setChartUnits(String units) {
		model.setChartUnits(units);
	}

	public void setChartMainEventOnly(int value) {
		if (value == 0)
			model.setMainEventOnly(false);
		else
			model.setMainEventOnly(true);
	}

	public void setChartEventNoCallPath(int value) {
		if (value == 0)
			model.setEventNoCallpath(false);
		else
			model.setEventNoCallpath(true);
	}

	public void setChartEventExclusive100(int value) {
		if (value == 0)
			model.setEventExclusive100(false);
		else
			model.setEventExclusive100(true);
	}

	public void setChartLogYAxis(int value) {
		if (value == 0)
			model.setChartLogYAxis(false);
		else
			model.setChartLogYAxis(true);
	}

	public void setChartScalability(int value) {
		if (value == 0)
			model.setChartScalability(false);
		else
			model.setChartScalability(true);
	}

	public void setChartEfficiency(int value) {
		if (value == 0)
			model.setChartEfficiency(false);
		else
			model.setChartEfficiency(true);
	}

	public void setChartConstantProblem(int value) {
		if (value == 0)
			model.setConstantProblem(false);
		else
			model.setConstantProblem(true);
	}

	public void setChartHorizontal(int value) {
		if (value == 0)
			model.setChartHorizontal(false);
		else
			model.setChartHorizontal(true);
	}

	public void resetChartDefaults() {
		//model.setCurrentSelection(model.getApplication());
		model.setGroupName(null);
		model.setMetricName(null);
		model.setEventName(null);
    	this.setDimensionReduction(TransformationType.NONE, null);
		this.setChartMetadataFieldName(null);
		this.setChartMetadataFieldValue(null);
		this.setChartSeriesName(null);
		this.setChartXAxisName(null, null);
		this.setChartYAxisName(null, null);
		this.setChartTitle(null);
		this.setChartMainEventOnly(0);
		this.setChartEventNoCallPath(0);
		this.setChartEventExclusive100(0);
		this.setChartLogYAxis(0);
		this.setChartScalability(0);
		this.setChartEfficiency(0);
		this.setChartConstantProblem(0);
		this.setChartHorizontal(0);
	}

	public ListIterator<Application> getApplicationList() {
		return connection.getApplicationList();
	}

	public ListIterator<Experiment> getExperimentList() {
		return connection.getExperimentList(model.getApplication().getID());
	}

	public ListIterator<Trial> getTrialList() {
		return connection.getTrialList(model.getExperiment().getID());
	}

	public ListIterator<RMISortableIntervalEvent> getEventList(Trial trial, int metricIndex) {
		return connection.getEventList(trial.getID(), metricIndex);
	}
	
	public void runComparisonRules(Trial baseline, Trial comparison) {
		//RuleHarnessOld.processRules(baseline, comparison);
	}

	public void setShowZero(int value) {
		if (value == 0)
			model.setShowZero(false);
		else
			model.setShowZero(true);
	}
}