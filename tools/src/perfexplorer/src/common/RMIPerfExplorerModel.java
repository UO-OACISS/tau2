package common;

import edu.uoregon.tau.perfdmf.*;
import java.io.Serializable;
import java.util.List;

/**
 * This RMI object defines the state of the client model when an analysis
 * request is made.
 *
 * <P>CVS $Id: RMIPerfExplorerModel.java,v 1.8 2005/10/21 20:58:59 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class RMIPerfExplorerModel implements Serializable {
	// constants for cluster analysis parameters
	public final static String K_MEANS = "K Means";
	public final static String K_HARMONIC_MEANS = "K Harmonic Means";
	public final static String GEM = "Gaussian Expectation-Maximization";
	public final static String FUZZY_K_MEANS = "Fuzzy K Means";
	public final static String CORRELATION_ANALYSIS = "Correlation Analysis";
	// constants for dimension reduction
	public final static String LINEAR_PROJECTION = "Random Linear Projection (disabled)";
	public final static String OVER_X_PERCENT = "Over X Percent";
	public final static String REGRESSION = "PCA (disabled)";
	public final static String NONE = "none";
	public final static String PERCENTAGE_OF_TOTAL = "Percentage of Total";
	public final static String RANGE_OF_TOTAL = "Range of Total";
	public final static int MAX_CLUSTERS = 10;
	public final static double X_PERCENT = 1.0;

	// constants for chart selections
	public final static int NO_MULTI = 0;
	public final static int APPLICATION = 1;
	public final static int EXPERIMENT = 2;
	public final static int TRIAL = 3;
	public final static int METRIC = 4;
	public final static int VIEW = 5;
	protected List multiSelections = null;
	protected int multiSelectionType = 0;

	// constants for chart parameters
	protected String groupName = null;
	protected String metricName = null;
	protected String eventName = null;
	protected String totalTimesteps = null;
	protected Boolean constantProblem = null;

	// more cluster settings
	protected String clusterMethod = null;
	protected String dimensionReduction = null;
	protected String normalization = null;
	protected int numberOfClusters = MAX_CLUSTERS;
	protected double xPercent = X_PERCENT;

	protected Object currentSelection = null;
	protected Application application = null;
	protected Experiment experiment = null;
	protected Trial trial = null;
	protected RMIView view = null;
	protected int analysisID = 0;
	protected Object[] fullPath = null;

	public RMIPerfExplorerModel () {
		super();
	}

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
		this.analysisID = source.analysisID;
		this.fullPath = source.fullPath;
	}

	public static Object[] getClusterMethods() {
		Object[] options = {K_MEANS, K_HARMONIC_MEANS, GEM, FUZZY_K_MEANS};
		return options;
	}

	public static Object[] getDimensionReductions() {
		//Object[] options = {LINEAR_PROJECTION, REGRESSION, OVER_X_PERCENT, NONE};
		Object[] options = {OVER_X_PERCENT, NONE};
		return options;
	}

	public static Object[] getNormalizations() {
		Object[] options = {PERCENTAGE_OF_TOTAL, RANGE_OF_TOTAL, NONE};
		return options;
	}

	public Object getCurrentSelection () {
		return currentSelection;
	}

	public Application getApplication() {
		return application;
	}

	public Experiment getExperiment() {
		return experiment;
	}

	public Trial getTrial() {
		return trial;
	}

	public String getClusterMethod () {
		return (clusterMethod == null) ? K_MEANS : clusterMethod;
	}

	public String getDimensionReduction () {
		return (dimensionReduction == null) ? NONE : dimensionReduction;
	}

	public String getNormalization () {
		return (normalization == null) ? NONE : normalization;
	}

	public int getNumberOfClusters () {
		return numberOfClusters;
	}

	public double getXPercent () {
		return xPercent;
	}

	public void setCurrentSelection (Object currentSelection) {
		groupName = null;
		metricName = null;
		eventName = null;
		totalTimesteps = null;
		multiSelections = null;
		multiSelectionType = NO_MULTI;
		if (currentSelection instanceof Application) {
			application = (Application)currentSelection;
		} else if (currentSelection instanceof Experiment) {
			experiment = (Experiment)currentSelection;
		} else if (currentSelection instanceof Trial) {
			trial = (Trial)currentSelection;
		} else if (currentSelection instanceof RMIView) {
			view = (RMIView)currentSelection;
		} //else if (currentSelection instanceof Metric) {
		this.currentSelection = currentSelection;
	}

	public void setCurrentSelection (Object[] objectPath) {
		groupName = null;
		metricName = null;
		eventName = null;
		totalTimesteps = null;
		multiSelections = null;
		multiSelectionType = NO_MULTI;
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
			} //else if (objectPath[i] instanceof Metric) {
			currentSelection = objectPath[i];
		}
	}

	public void setClusterMethod (String clusterMethod) {
		this.clusterMethod = clusterMethod;
	}

	public void setDimensionReduction (String dimensionReduction) {
		this.dimensionReduction = dimensionReduction;
	}

	public void setNormalization (String normalization) {
		this.normalization = normalization;
	}

	public void setNumberOfClusters (String numberOfClusters) {
		this.numberOfClusters = Integer.parseInt(numberOfClusters);
	}

	public void setXPercent (String xPercent) {
		this.xPercent = Double.parseDouble(xPercent);
	}

	public String toString() {
		if (multiSelectionType == APPLICATION) {
			return "Applications";
		} else if (multiSelectionType == EXPERIMENT) {
			String tmpStr = (application == null) ? "" : application.getName();
			return tmpStr;
		} else if (multiSelectionType == TRIAL) {
			String tmpStr1 = (application == null) ? "" : application.getName();
			String tmpStr2 = experiment.getName();
			String tmpStr = tmpStr1 + ":" + tmpStr2;
			return tmpStr;
		} else if (multiSelectionType == METRIC) {
			String tmpStr1 = (application == null) ? "" : application.getName();
			String tmpStr2 = (experiment == null) ? "" : experiment.getName();
			String tmpStr3 = trial.getName();
			String tmpStr = tmpStr1 + ":" + tmpStr2 + ":" + tmpStr3;
			return tmpStr;
		} else if (multiSelectionType == VIEW) {
			return "Custom View";
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

	public String toShortString() {
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

	public int getAnalysisID() {
		return this.analysisID;
	}

	public void setAnalysisID(int analysisID) {
		this.analysisID = analysisID;
	}

	public boolean setMultiSelection(List objects) {
		for (int i = 0 ; i < objects.size() ; i++) {
			if (objects.get(i) instanceof Application) {
				if (multiSelectionType != APPLICATION &&
					multiSelectionType != NO_MULTI)
					return false;
				multiSelectionType = APPLICATION;
			} else if (objects.get(i) instanceof Experiment) {
				if (multiSelectionType != EXPERIMENT &&
					multiSelectionType != NO_MULTI)
					return false;
				multiSelectionType = EXPERIMENT;
			} else if (objects.get(i) instanceof Trial) {
				if (multiSelectionType != TRIAL &&
					multiSelectionType != NO_MULTI)
					return false;
				multiSelectionType = TRIAL;
			} else if (objects.get(i) instanceof Metric) {
				if (multiSelectionType != METRIC &&
					multiSelectionType != NO_MULTI)
					return false;
				multiSelectionType = METRIC;
			} else if (objects.get(i) instanceof RMIView) {
				if (multiSelectionType != VIEW &&
					multiSelectionType != NO_MULTI)
					return false;
				multiSelectionType = VIEW;
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

	public List getMultiSelection() {
		return multiSelections;
	}

	public String getGroupName () {
		return groupName;
	}

	public void setGroupName (String groupName) {
		this.groupName = groupName;
	}

	public String getMetricName () {
		return metricName;
	}

	public void setMetricName (String metricName) {
		this.metricName = metricName;
	}

	public String getTotalTimesteps () {
		return totalTimesteps;
	}

	public void setTotalTimesteps (String totalTimesteps) {
		this.totalTimesteps = totalTimesteps;
	}

	public String getEventName () {
		return eventName;
	}

	public void setEventName (String eventName) {
		this.eventName = eventName;
	}
	
	public String getViewSelectionPath (boolean joinApp, boolean joinExp) {
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
				if (view.getField("table_name").equalsIgnoreCase("Application")) {
					buf.append (" a.");
				} else if (view.getField("table_name").equalsIgnoreCase("Experiment")) {
					buf.append (" e.");
				} else /*if (view.getField("table_name").equalsIgnoreCase("Trial")) */ {
					buf.append (" t.");
				}
				buf.append (view.getField("column_name"));
				buf.append (" " + view.getField("operator") + " '");
				buf.append (view.getField("value"));
				buf.append ("' ");
				doAnd = true;
			}
		}

		return buf.toString();
	}

	public String getViewSelectionString () {
		StringBuffer buf = new StringBuffer();
		int i = fullPath.length - 1;
		RMIView view = (RMIView) fullPath[i];
		if (view.getField("table_name").equalsIgnoreCase("Application")) {
			buf.append (" a.");
		} else if (view.getField("table_name").equalsIgnoreCase("Experiment")) {
			buf.append (" e.");
		} else /*if (view.getField("table_name").equalsIgnoreCase("Trial")) */ {
			buf.append (" t.");
		}
		buf.append (view.getField("column_name"));
		return buf.toString();
	}

	public String getViewID () {
		int i = fullPath.length - 1;
		RMIView view = (RMIView) fullPath[i];
		return view.getField("id");
	}
	
	public void setConstantProblem(boolean constantProblem) {
		this.constantProblem = new Boolean(constantProblem);
	}

	public Boolean getConstantProblem() {
		return this.constantProblem;
	}
}
