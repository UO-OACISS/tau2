package common;

import edu.uoregon.tau.perfdmf.Application;
import edu.uoregon.tau.perfdmf.Experiment;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.IntervalEvent;
import java.io.Serializable;
import java.util.List;

/**
 * This RMI object defines the state of the client model when an analysis
 * request is made.
 *
 * <P>CVS $Id: RMIPerfExplorerModel.java,v 1.21 2007/01/04 21:20:03 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class RMIPerfExplorerModel implements Serializable {
	public final static int MAX_CLUSTERS = 10;
	public final static double X_PERCENT = 1.0;

	protected List multiSelections = null;
	protected SelectionType multiSelectionType = SelectionType.NO_MULTI;

	// constants for chart parameters
	protected String groupName = null;
	protected String metricName = null;
	protected String eventName = null;
	protected String totalTimesteps = null;
	protected Boolean constantProblem = null;

	// more cluster settings
	protected AnalysisType clusterMethod = null;
	protected TransformationType dimensionReduction = null;
	protected TransformationType normalization = null;
	protected int numberOfClusters = MAX_CLUSTERS;
	protected double xPercent = X_PERCENT;

	protected Object currentSelection = null;
	protected Application application = null;
	protected Experiment experiment = null;
	protected Trial trial = null;
	protected RMIView view = null;
	protected Metric metric = null;
	protected IntervalEvent event = null;
	protected int analysisID = 0;
	protected Object[] fullPath = null;

	/**
	 * Default Constructor.
	 *
	 */
	public RMIPerfExplorerModel () {
		super();
	}

	/**
	 * Constructor which copies all fields.
	 * 
	 * @param source
	 */
	public RMIPerfExplorerModel (RMIPerfExplorerModel source) {
		this.multiSelectionType = source.multiSelectionType;
		this.groupName = source.groupName;
		this.metricName = source.metricName;
		this.eventName = source.eventName;
		this.totalTimesteps = source.totalTimesteps;
		this.clusterMethod = source.clusterMethod;
		this.dimensionReduction = source.dimensionReduction;
		this.normalization = source.normalization;
		this.numberOfClusters = source.numberOfClusters;
		this.xPercent = source.xPercent;
		this.currentSelection = source.currentSelection;
		this.application = source.application;
		this.experiment = source.experiment;
		this.trial = source.trial;
		this.view = source.view;
		this.metric = source.metric;
		this.event = source.event;
		this.analysisID = source.analysisID;
		this.fullPath = source.fullPath;
	}

	/**
	 * Returns the current selection.
	 * 
	 * @return
	 */
	public Object getCurrentSelection () {
		return currentSelection;
	}

	/**
	 * Returns the currently selected application.
	 * 
	 * @return
	 */
	public Application getApplication() {
		return application;
	}

	/**
	 * Returns the currently selected experiment.
	 * 
	 * @return
	 */
	public Experiment getExperiment() {
		return experiment;
	}

	/**
	 * Returns the currently selected trial.
	 * 
	 * @return
	 */
	public Trial getTrial() {
		return trial;
	}

	/**
	 * Returns the currently selected metric.
	 * 
	 * @return
	 */
	public Metric getMetric() {
		return metric;
	}

	/**
	 * Returns the currently selected event.
	 * 
	 * @return
	 */
	public IntervalEvent getEvent() {
		return event;
	}

	/**
	 * Returns the currently selected cluster method.
	 * @return
	 */
	public AnalysisType getClusterMethod () {
		return (clusterMethod == null) ? AnalysisType.K_MEANS : clusterMethod;
	}

	/**
	 * Returns the currently selected dimension reduction method.
	 * @return
	 */
	public TransformationType getDimensionReduction () {
		return (dimensionReduction == null) ? TransformationType.NONE : dimensionReduction;
	}

	/**
	 * Returns the currently selected normalization method.
	 * 
	 * @return
	 */
	public TransformationType getNormalization () {
		return (normalization == null) ? TransformationType.NONE : normalization;
	}

    /**
     * Returns the currently selected number of clusters to find.
     *
     * @return
     */
	public int getNumberOfClusters () {
		return numberOfClusters;
	}

    /**
     * Returns the value for the "x percent" dimension reduction.
     *
     * @return
     */
	public double getXPercent () {
		return xPercent;
	}

    /**
     * Sets the current selection.
     *
     * @param currentSelection
     */
	public void setCurrentSelection (Object currentSelection) {
		groupName = null;
		metricName = null;
		eventName = null;
		totalTimesteps = null;
		multiSelections = null;
		constantProblem = null;
		multiSelectionType = SelectionType.NO_MULTI;
		if (currentSelection instanceof Application) {
			application = (Application)currentSelection;
		} else if (currentSelection instanceof Experiment) {
			experiment = (Experiment)currentSelection;
		} else if (currentSelection instanceof Trial) {
			trial = (Trial)currentSelection;
		} else if (currentSelection instanceof RMIView) {
			view = (RMIView)currentSelection;
		} else if (currentSelection instanceof Metric) {
			metric = (Metric)currentSelection;
		} else if (currentSelection instanceof IntervalEvent) {
			event = (IntervalEvent)currentSelection;
		}
		this.currentSelection = currentSelection;
	}

    /**
     * Sets the current selection based on the selection path in the
     * client navigation tree.
     *
     * @param objectPath
     */
	public void setCurrentSelection (Object[] objectPath) {
		groupName = null;
		metricName = null;
		eventName = null;
		totalTimesteps = null;
		multiSelections = null;
		constantProblem = null;
		multiSelectionType = SelectionType.NO_MULTI;
		fullPath = objectPath;
		// application = null;
		// experiment = null;
		// trial = null;
		for (int i = 0 ; i < objectPath.length ; i++) {
			if (objectPath[i] instanceof Application) {
				application = (Application)objectPath[i];
			} else if (objectPath[i] instanceof Experiment) {
				experiment = (Experiment)objectPath[i];
			} else if (objectPath[i] instanceof Trial) {
				trial = (Trial)objectPath[i];
			} else if (objectPath[i] instanceof RMIView) {
				view = (RMIView)objectPath[i];
			} else if (objectPath[i] instanceof Metric) {
				metric = (Metric)objectPath[i];
			} else if (objectPath[i] instanceof IntervalEvent) {
				event = (IntervalEvent)objectPath[i];
			}
			currentSelection = objectPath[i];
		}
	}

    /**
     * Sets the cluster method to use.
     *
     * @param clusterMethod
     */
	public void setClusterMethod (String clusterMethod) {
		this.clusterMethod = AnalysisType.fromString(clusterMethod);
	}

    /**
     * Sets the cluster method to use.
     *
     * @param clusterMethod
     */
	public void setClusterMethod (AnalysisType clusterMethod) {
		this.clusterMethod = clusterMethod;
	}

    /**
     * Sets the dimension reduction method to use.
     *
     * @param dimensionReduction
     */
	public void setDimensionReduction (TransformationType dimensionReduction) {
		this.dimensionReduction = dimensionReduction;
	}

    /**
     * Sets the normalization method to use.
     *
     * @param normalization
     */
	public void setNormalization (TransformationType normalization) {
		this.normalization = normalization;
	}

    /**
     * Sets the number of clusters to find.
     *
     * @param numberOfClusters
     */
	public void setNumberOfClusters (String numberOfClusters) {
		this.numberOfClusters = Integer.parseInt(numberOfClusters);
	}

    /**
     * Sets the minimum threshold for the "x percent" dimension reduction
     * method.
     *
     * @param xPercent
     */
	public void setXPercent (String xPercent) {
		this.xPercent = Double.parseDouble(xPercent);
	}

    /**
     * Converts the current selection to a String.
     *
     * @return
     */
	public String toString() {
		if (multiSelectionType == SelectionType.APPLICATION) {
			return "Applications";
		} else if (multiSelectionType == SelectionType.EXPERIMENT) {
			String tmpStr = (application == null) ? "" : application.getName();
			return tmpStr;
		} else if (multiSelectionType == SelectionType.TRIAL) {
			String tmpStr1 = (application == null) ? "" : application.getName();
			String tmpStr2 = experiment.getName();
			String tmpStr = tmpStr1 + ":" + tmpStr2;
			return tmpStr;
		} else if (multiSelectionType == SelectionType.METRIC) {
			String tmpStr1 = (application == null) ? "" : application.getName();
			String tmpStr2 = (experiment == null) ? "" : experiment.getName();
			String tmpStr3 = trial.getName();
			String tmpStr = tmpStr1 + ":" + tmpStr2 + ":" + tmpStr3;
			return tmpStr;
		} else if (multiSelectionType == SelectionType.EVENT) {
			String tmpStr1 = (application == null) ? "" : application.getName();
			String tmpStr2 = (experiment == null) ? "" : experiment.getName();
			String tmpStr3 = trial.getName();
			String tmpStr = tmpStr1 + ":" + tmpStr2 + ":" + tmpStr3;
			return tmpStr;
		} else if (multiSelectionType == SelectionType.VIEW) {
			return "Custom View";
		} else if (currentSelection instanceof IntervalEvent) {
			IntervalEvent event = (IntervalEvent)currentSelection;
			String tmpStr1 = (application == null) ? "" : application.getName();
			String tmpStr2 = (experiment == null) ? "" : experiment.getName();
			String tmpStr3 = (trial == null) ? "" : trial.getName();
			String tmpStr4 = (metric == null) ? "" : metric.getName();
			String tmpStr5 = event.getName();
			String tmpStr = tmpStr1 + ":" + tmpStr2 + ":" + tmpStr3 + ":" + tmpStr4 + ":" + tmpStr5;
			return tmpStr;
		} else if (currentSelection instanceof Metric) {
			Metric metric = (Metric)currentSelection;
			String tmpStr1 = (application == null) ? "" : application.getName();
			String tmpStr2 = (experiment == null) ? "" : experiment.getName();
			String tmpStr3 = (trial == null) ? "" : trial.getName();
			String tmpStr4 = metric.getName();
			String tmpStr = tmpStr1 + ":" + tmpStr2 + ":" + tmpStr3 + ":" + tmpStr4;
			return tmpStr;
		} else if (currentSelection instanceof Trial) {
			Trial trial = (Trial)currentSelection;
			String tmpStr1 = (application == null) ? "" : application.getName();
			String tmpStr2 = (experiment == null) ? "" : experiment.getName();
			String tmpStr3 = trial.getName();
			String tmpStr = tmpStr1 + ":" + tmpStr2 + ":" + tmpStr3;
			return tmpStr;
		} else if (currentSelection instanceof Experiment) {
			Experiment experiment = (Experiment)currentSelection;
			String tmpStr1 = (application == null) ? "" : application.getName();
			String tmpStr2 = experiment.getName();
			String tmpStr = tmpStr1 + ":" + tmpStr2;
			return tmpStr;
		} else if (currentSelection instanceof Application) {
			Application application = (Application)currentSelection;
			String tmpStr = application.getName();
			return tmpStr;
		} else if (currentSelection instanceof RMIView) {
			RMIView view = (RMIView)currentSelection;
			String tmpStr = view.getField("NAME");
			return tmpStr;
		}
		else 
			return new String("");
	}

    /**
     * Converts the current selection to a shortened String.
     *
     * @return
     */
	public String toShortString() {
		if (currentSelection instanceof IntervalEvent) {
			IntervalEvent event = (IntervalEvent)currentSelection;
			String tmpStr1 = (application == null) ? "" : "" + application.getID();
			String tmpStr2 = (experiment == null) ? "" : "" + experiment.getID();
			String tmpStr3 = (trial == null) ? "" : "" + trial.getID();
			String tmpStr4 = (metric == null) ? "" : "" + metric.getID();
			String tmpStr5 = "" + event.getID();
			String tmpStr = tmpStr1 + ":" + tmpStr2 + ":" + tmpStr3 + ":" + tmpStr4 + ":" + tmpStr5;
			return tmpStr;
		}
		if (currentSelection instanceof Metric) {
			Metric metric = (Metric)currentSelection;
			String tmpStr1 = (application == null) ? "" : "" + application.getID();
			String tmpStr2 = (experiment == null) ? "" : "" + experiment.getID();
			String tmpStr3 = (trial == null) ? "" : "" + trial.getID();
			String tmpStr4 = "" + metric.getID();
			String tmpStr = tmpStr1 + "." + tmpStr2 + "." + tmpStr3 + "." + tmpStr4;
			return tmpStr;
		}
		if (currentSelection instanceof Trial) {
			Trial trial = (Trial)currentSelection;
			String tmpStr1 = (application == null) ? "" : "" + application.getID();
			String tmpStr2 = (experiment == null) ? "" : "" + experiment.getID();
			String tmpStr3 = "" + trial.getID();
			String tmpStr = tmpStr1 + "." + tmpStr2 + "." + tmpStr3;
			return tmpStr;
		}
		if (currentSelection instanceof Experiment) {
			Experiment experiment = (Experiment)currentSelection;
			String tmpStr1 = (application == null) ? "" : "" + application.getID();
			String tmpStr2 = "" + experiment.getID();
			String tmpStr = tmpStr1 + "." + tmpStr2;
			return tmpStr;
		}
		if (currentSelection instanceof Application) {
			Application application = (Application)currentSelection;
			String tmpStr = "" + application.getID();
			return tmpStr;
		}
		if (currentSelection instanceof RMIView) {
			RMIView view = (RMIView)currentSelection;
			String tmpStr = view.getField("NAME");
			return tmpStr;
		}
		else 
			return new String("");

	}

    /**
     * Gets the current analysis ID.
     * When the analysis request is accepted, the database ID of the request
     * is stored as the analysis ID.
     *
     * @return
     */
	public int getAnalysisID() {
		return this.analysisID;
	}

    /**
     * Sets the current analysis ID.
     *
     * @param analysisID
     */
	public void setAnalysisID(int analysisID) {
		this.analysisID = analysisID;
	}

    /**
     * Adds the selection to the multiselection in the
     * client navigation tree, provided it is the same type as the
     * current selection, and multiselection is allowed.
     *
     * @param objects
     * @return
     */
	public boolean setMultiSelection(List objects) {
		for (int i = 0 ; i < objects.size() ; i++) {
			if (objects.get(i) instanceof Application) {
				if (multiSelectionType != SelectionType.APPLICATION &&
					multiSelectionType != SelectionType.NO_MULTI)
					return false;
				multiSelectionType = SelectionType.APPLICATION;
			} else if (objects.get(i) instanceof Experiment) {
				if (multiSelectionType != SelectionType.EXPERIMENT &&
					multiSelectionType != SelectionType.NO_MULTI)
					return false;
				multiSelectionType = SelectionType.EXPERIMENT;
			} else if (objects.get(i) instanceof Trial) {
				if (multiSelectionType != SelectionType.TRIAL &&
					multiSelectionType != SelectionType.NO_MULTI)
					return false;
				multiSelectionType = SelectionType.TRIAL;
			} else if (objects.get(i) instanceof Metric) {
				if (multiSelectionType != SelectionType.METRIC &&
					multiSelectionType != SelectionType.NO_MULTI)
					return false;
				multiSelectionType = SelectionType.METRIC;
			} else if (objects.get(i) instanceof RMIView) {
				if (multiSelectionType != SelectionType.VIEW &&
					multiSelectionType != SelectionType.NO_MULTI)
					return false;
				multiSelectionType = SelectionType.VIEW;
			} else if (objects.get(i) instanceof IntervalEvent) {
				if (multiSelectionType != SelectionType.EVENT &&
					multiSelectionType != SelectionType.NO_MULTI)
					return false;
				multiSelectionType = SelectionType.EVENT;
			}
		}

		groupName = null;
		metricName = null;
		eventName = null;
		totalTimesteps = null;
		constantProblem = null;
		multiSelections = objects;
		return true;
	}

    /**
     * Returns the current multiselection.
     *
     * @return
     */
	public List getMultiSelection() {
		return multiSelections;
	}

    /**
     * Returns the currently selected group name.
     * @return
     */
	public String getGroupName () {
		return groupName;
	}

    /**
     * Sets the selected group name.
     *
     * @param groupName
     */
	public void setGroupName (String groupName) {
		this.groupName = groupName;
	}

    /**
     * Based on the selection, return the available metrics.
     *
     * @return
     */
	public String getMetricName () {
		if (metricName != null)
			return metricName;
		if (currentSelection instanceof Metric) {
			Metric met = (Metric)currentSelection;
			return met.getName();
		}
		if (currentSelection instanceof IntervalEvent) {
			if (metric != null)
				return metric.getName();
		}
		// otherwise, just return the null String
		return metricName;
	}

	/**
	 * Gets the units for the current metric.
	 * 
	 * @return
	 */
	public String getMetricNameUnits() {
		// similar to getMetricName, but add "seconds" if metric == time
		String name = getMetricName();
		if (name != null && name.equalsIgnoreCase("time")) {
			return name + " (seconds)";
		}
		return name;
	}

	/**
     * Set the selected metric name.
     *
     * @param metricName
     */
	public void setMetricName (String metricName) {
		this.metricName = metricName;
	}

    /**
     * Get the total number of timeteps for the application
     * TODO - remove this option.
     *
     * @return
     */
	public String getTotalTimesteps () {
		return totalTimesteps;
	}

    /**
     * Set the total number of timesteps for the application
     * TODO - remove this option
     *
     * @param totalTimesteps
     */
	public void setTotalTimesteps (String totalTimesteps) {
		this.totalTimesteps = totalTimesteps;
	}

    /**
     * Get the currently selected event.
     *
     * @return
     */
	public String getEventName () {
		return eventName;
	}

    /**
     * Set the currently selected event.
     * @param eventName
     */
	public void setEventName (String eventName) {
		this.eventName = eventName;
	}
	
    /**
     * Based on the current selection path, build the SQL inner join clause
     * to select the current trials.
     *
     * @param joinApp
     * @param joinExp
     * @param dbType
     * @return
     */
	public String getViewSelectionPath (boolean joinApp, boolean joinExp, String dbType) {
		StringBuffer buf = new StringBuffer();
		if (joinExp)
			buf.append(" inner join experiment e on t.experiment = e.id ");
		if (joinApp)
			buf.append(" inner join application a on e.application = a.id ");
		buf.append(" WHERE ");
		boolean doAnd = false;
		for (int i = 0 ; i < fullPath.length ; i++) {
			if (i > 0 && doAnd) {
				buf.append (" AND ");
			}
			if (fullPath[i] instanceof RMIView) {
				RMIView view = (RMIView) fullPath[i];
				if (dbType.equalsIgnoreCase("db2"))
					buf.append(" cast (");
				if (view.getField("table_name").equalsIgnoreCase("Application")) {
					buf.append (" a.");
				} else if (view.getField("table_name").equalsIgnoreCase("Experiment")) {
					buf.append (" e.");
				} else /*if (view.getField("table_name").equalsIgnoreCase("Trial")) */ {
					buf.append (" t.");
				}
				buf.append (view.getField("column_name"));
				if (dbType.equalsIgnoreCase("db2"))
					buf.append(" as varchar(256)) ");
				buf.append (" " + view.getField("operator") + " '");
				buf.append (view.getField("value"));
				buf.append ("' ");
				doAnd = true;
			}
		}

		return buf.toString();
	}

    /**
     * Based on the current selection path, build the SQL where clause
     * to select the current trials.
     *
     * @param dbType
     * @return
     */
	public String getViewSelectionString (String dbType) {
		StringBuffer buf = new StringBuffer();
		int i = fullPath.length - 1;
		RMIView view = (RMIView) fullPath[i];
		if (//(view.getField("operator").equalsIgnoreCase("like")) || //{
			//buf.append(" '" + view.getField("value").replaceAll("%","") + "'");
		/*} else if*/ (view.getField("operator").equals("="))) {
			if (dbType.equalsIgnoreCase("db2"))
				buf.append(" cast ( ");
			if (view.getField("table_name").equalsIgnoreCase("Application")) {
				buf.append (" a.");
			} else if (view.getField("table_name").equalsIgnoreCase("Experiment")) {
				buf.append (" e.");
			} else /*if (view.getField("table_name").equalsIgnoreCase("Trial")) */ {
				buf.append (" t.");
			}
			buf.append (view.getField("column_name"));
			if (dbType.equalsIgnoreCase("db2"))
				buf.append(" as varchar(256)) ");
		}else {
			buf.append(" '" + view.getField("name") + "'");
		}
		return buf.toString();
	}

    /**
     * Based on the current selection, find the lowest level view which
     * filters these trials.
     *
     * @return
     */
	public String getViewID () {
		int i = fullPath.length - 1;
		RMIView view = (RMIView) fullPath[i];
		return view.getField("id");
	}
	
    /**
     * Set the weak/strong scaling option for scalability charts.
     * @param constantProblem
     */
	public void setConstantProblem(boolean constantProblem) {
		this.constantProblem = new Boolean(constantProblem);
	}

    /**
     * Get the weak/strong scaling option for scalability charts.
     *
     * @return
     */
	public Boolean getConstantProblem() {
		return this.constantProblem;
	}
}
