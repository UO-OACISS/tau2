/*
 * ParaProfTrial.java
 * 
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description: The manner in which this
 * class behaves is slightly different from its parent. It behaves more as a
 * container for its DataSession, than a setting for it. So, in a sense the role
 * is almost reversed (but not quite). This is a result of the fact that ParaProf
 * must maintain the majority of its data itself, and as such, ParaProfTrial
 * serves as the reference through which data is accessed.
 */

package edu.uoregon.tau.paraprof;

import java.awt.EventQueue;
import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Observable;
import java.util.Observer;
import java.util.Set;
import java.util.Vector;

import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreePath;

import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.common.MetaDataMap.MetaDataKey;
import edu.uoregon.tau.paraprof.script.ParaProfScript;
import edu.uoregon.tau.paraprof.script.ParaProfTrialScript;
import edu.uoregon.tau.paraprof.util.FileMonitor;
import edu.uoregon.tau.paraprof.util.FileMonitorListener;
import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.Database;
import edu.uoregon.tau.perfdmf.DatabaseAPI;
import edu.uoregon.tau.perfdmf.FileList;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Group;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.perfdmf.UserEvent;

public class ParaProfTrial extends Observable implements ParaProfTreeNodeUserObject {

    private Function highlightedFunction = null;
    private Group highlightedGroup = null;
    private UserEvent highlightedUserEvent = null;

    private DatabaseAPI dbAPI;
    private ParaProfExperiment experiment = null;
    private ParaProfView view = null;
    private DefaultMutableTreeNode defaultMutableTreeNode = null;
    private TreePath treePath = null;
    private boolean dBTrial = false;
    private boolean upload = false;
    private boolean loading = false;

    private GlobalDataWindow fullDataWindow = null;
    private ColorChooser clrChooser = ParaProf.colorChooser;
    private PreferencesWindow preferencesWindow = ParaProf.preferencesWindow;

    private String path = null;
    private String pathReverse = null;
    private Metric defaultMetric = null;
    private int selectedSnapshot = -1; // -1 = final snapshot

    private Group selectedGroup;
    private Group groupInclude;
    private Group groupExclude;

    private Trial trial;

    private boolean functionMask[];

    private boolean monitored;
    //private Timer monitorTimer;
    
    //JRadioButtonMenuItem metricButton = null;

    private List<Observer> obs = new ArrayList<Observer>();

    private FileMonitorListener fileMonitorListener;

    private SnapshotControlWindow snapshotControlWindow;

    public ParaProfTrial() {
        trial = new Trial();
        trial.setID(-1);
        trial.setExperimentID(-1);
        trial.setApplicationID(-1);
        trial.setName("");
    }

    public ParaProfTrial(Trial trial) {
        this.trial = new Trial(trial);
        assignDefaultMetric();
    }

    public Iterator<Function> getFunctions() {
        return getDataSource().getFunctionIterator();
    }

    public Thread getMeanThread() {
        return getDataSource().getMeanData();
    }

    public Trial getTrial() {
        return trial;
    }

    public int getApplicationID() {
        return trial.getApplicationID();
    }

    public int getExperimentID() {
        return trial.getExperimentID();
    }

    public int getID() {
        return trial.getID();
    }

    public void setApplicationID(int id) {
        trial.setApplicationID(id);
    }

    public void setExperimentID(int id) {
        trial.setExperimentID(id);
    }

    public void setID(int id) {
        trial.setID(id);
    }
    
    public void rename(String newName){
    	if(dBTrial){
			DatabaseAPI databaseAPI = getDatabaseAPI();
			if (databaseAPI != null) {
				this.trial.rename(databaseAPI.getDb(), newName);
			}
		}
		trial.setName(newName);
    }

    public String getName() {
        return trial.getName();
    }

    public DataSource getDataSource() {
        return trial.getDataSource();
    }

    public void setExperiment(ParaProfExperiment experiment) {
        this.experiment = experiment;
    }

    public void setView(ParaProfView view) {
        this.view = view;
    }

    public ParaProfView getView() {
        return view;
    }

    public ParaProfExperiment getExperiment() {
        return experiment;
    }

    public void setDMTN(DefaultMutableTreeNode defaultMutableTreeNode) {
        this.defaultMutableTreeNode = defaultMutableTreeNode;
    }

    public DefaultMutableTreeNode getDMTN() {
        return defaultMutableTreeNode;
    }

    public void setTreePath(TreePath treePath) {
        this.treePath = treePath;
    }

    public TreePath getTreePath() {
        return treePath;
    }

    public void setDBTrial(boolean dBTrial) {
        this.dBTrial = dBTrial;
    }

    public boolean dBTrial() {
        return dBTrial;
    }

    public void setUpload(boolean upload) {
        this.upload = upload;
    }

    public boolean upload() {
        return upload;
    }

    public void setLoading(boolean loading) {
        this.loading = loading;
    }

    public boolean loading() {
        return loading;
    }

    public String getIDString() {
        if (experiment != null) {
            return (experiment.getIDString()) + ":" + (trial.getID());
        } else if (view != null) {
            return (view.getIDString()) + ":" + (trial.getID());
        } else {
            return ":" + (trial.getID());
        }
    }

    public ColorChooser getColorChooser() {
        return clrChooser;
    }

    public PreferencesWindow getPreferencesWindow() {
        return preferencesWindow;
    }

    //Used in many ParaProf windows for the title of the window.
    public String getTrialIdentifier(boolean reverse) {
        if (path != null) {
            if (reverse) {
                return pathReverse;
            } else {
                return path;
            }
        } else {
            return "Application " + trial.getApplicationID() + ", Experiment " + trial.getExperimentID() + ", Trial "
                    + trial.getID() + ".";
        }
    }

    //Sets both the path and the path reverse.
    public void setPaths(String path) {
        this.path = path;
        this.pathReverse = FileList.getPathReverse(path);
    }

    public String getPath() {
        return path;
    }

    public String getPathReverse() {
        return pathReverse;
    }

    public String toString() {
        if (this.loading()) {
            return trial.getName() + " (Loading...)";
        } else {
            return trial.getName();
        }
    }

    public void clearDefaultMutableTreeNode() {
        this.setDMTN(null);
    }

    public GlobalDataWindow getFullDataWindow() {
        return fullDataWindow;
    }

    public void showMainWindow() {
        if (fullDataWindow == null) {
            showSnapshotController();
        }
        fullDataWindow = new GlobalDataWindow(this, trial.getDataSource().getTopLevelPhase());
        fullDataWindow.setVisible(true);
    }

    public void showSnapshotController() {
        if (getDataSource().getWellBehavedSnapshots()) {
            snapshotControlWindow = new SnapshotControlWindow(this);
            snapshotControlWindow.setVisible(true);
        }
    }

    public void setDefaultMetric(Metric metric) {
        this.defaultMetric = metric;
    }

    public Metric getDefaultMetric() {
        return defaultMetric;
    }

    public boolean isTimeMetric() {
        return defaultMetric.isTimeMetric();
    }

    public boolean isDerivedMetric() {

        // We can't do this, HPMToolkit stuff has /'s and -'s all over the place
        //String metricName = this.getMetricName(this.getSelectedMetricID());
        //if (metricName.indexOf("*") != -1 || metricName.indexOf("/") != -1)
        //    return true;
        return defaultMetric.getDerivedMetric();
    }

    //Override this function.
    public List<Metric> getMetrics() {
        if (trial == null) {
            return null;
        }
        if (trial.getDataSource() == null) {
            return null;
        }
        return trial.getDataSource().getMetrics();
    }

    public int getMetricID(String name) {
        Metric m = getDataSource().getMetric(name);
        if (m == null) {
            return -1;
        } else {
            return m.getID();
        }
    }

    public int getNumberOfMetrics() {
        return trial.getDataSource().getNumberOfMetrics();
    }

    public ParaProfMetric getMetric(int metricID) {
        return (ParaProfMetric) trial.getDataSource().getMetric(metricID);
    }

    public Metric getRegularMetric(int metricID) {
        return trial.getDataSource().getMetric(metricID);
    }

    public String getMetricName(int metricID) {
        return trial.getDataSource().getMetricName(metricID);
    }

    public ParaProfMetric addMetric() {
        ParaProfMetric metric = new ParaProfMetric();
        trial.getDataSource().addMetric(metric);
        return metric;
    }

    public boolean groupNamesPresent() {
        return trial.getDataSource().getGroupNamesPresent();
    }

    public boolean userEventsPresent() {
        return trial.getDataSource().getUserEventsPresent();
    }

    public boolean callPathDataPresent() {
        return trial.getDataSource().getCallPathDataPresent();
    }

    //Overrides the parent getMaxNCTNumbers.
    public int[] getMaxNCTNumbers() {
        return trial.getDataSource().getMaxNCTNumbers();
    }

    public void setMeanData(int metricID) {
    	// save the old derived provided state
    	boolean tmpVal = trial.getDataSource().isDerivedProvided();
    	// set to false, so we compute new statistics for the new metric
    	trial.getDataSource().setDerivedProvided(false);
        trial.getDataSource().generateStatistics(metricID, metricID);
        // restore the old value
    	trial.getDataSource().setDerivedProvided(tmpVal);
    }

    // return a vector of only those functions that are currently "displayed" (i.e. group masks, etc)
    public List<Function> getDisplayedFunctions() {
        List<Function> displayedFunctions = new ArrayList<Function>();

        for (Iterator<Function> it = this.getDataSource().getFunctionIterator(); it.hasNext();) {
            Function function = it.next();
            if (this.displayFunction(function)) {
                displayedFunctions.add(function);
            }
        }
        return displayedFunctions;
    }

    public boolean displayFunction(Function function) {
        if (functionMask == null) {
            return true;
        } else {
            return functionMask[function.getID()];
        }
    }

    public void showGroup(Group group) {
        for (Iterator<Function> it = getDataSource().getFunctionIterator(); it.hasNext();) {
            Function function = it.next();
            if (function.isGroupMember(group)) {
                functionMask[function.getID()] = true;
            }
        }
        groupInclude = null;
        groupExclude = null;
        updateRegisteredObjects("dataEvent");
    }

    public void hideGroup(Group group) {
        for (Iterator<Function> it = getDataSource().getFunctionIterator(); it.hasNext();) {
            Function function = it.next();
            if (function.isGroupMember(group)) {
                functionMask[function.getID()] = false;
            }
        }
        groupInclude = null;
        groupExclude = null;
        updateRegisteredObjects("dataEvent");
    }

    public void showGroupOnly(Group group) {
        for (Iterator<Function> it = getDataSource().getFunctionIterator(); it.hasNext();) {
            Function function = it.next();
            if (function.isGroupMember(group)) {
                functionMask[function.getID()] = true;
            } else {
                functionMask[function.getID()] = false;
            }
        }
        groupInclude = group;
        groupExclude = null;
        updateRegisteredObjects("dataEvent");
    }

    public void showAllExcept(Group group) {
        for (Iterator<Function> it = getDataSource().getFunctionIterator(); it.hasNext();) {
            Function function = it.next();
            if (function.isGroupMember(group)) {
                functionMask[function.getID()] = false;
            } else {
                functionMask[function.getID()] = true;
            }
        }
        groupInclude = null;
        groupExclude = group;
        updateRegisteredObjects("dataEvent");
    }

    public void setFunctionMask(boolean mask[]) {
        this.functionMask = mask;
        groupInclude = null;
        groupExclude = null;
        updateRegisteredObjects("dataEvent");
    }

    public void showFunction(Function function) {
        functionMask[function.getID()] = true;
        updateRegisteredObjects("dataEvent");
    }

    public void hideFunction(Function function) {
        functionMask[function.getID()] = false;
        updateRegisteredObjects("dataEvent");
    }

    public void hideMatching(String string, boolean caseSensitive, boolean allExcept) {
        maskMatching(string, false, caseSensitive, allExcept);
    }

    public void showMatching(String string, boolean caseSensitive, boolean allExcept) {
        maskMatching(string, true, caseSensitive, allExcept);
    }

    public void maskMatching(String string, boolean value, boolean caseSensitive, boolean allExcept) {
        if (caseSensitive) {
            string = string.toUpperCase();
        }

        if (allExcept) {
            value = !value;
            for (int i = 0; i < functionMask.length; i++) {
                functionMask[i] = !value;
            }
        }

        for (Iterator<Function> it = getDataSource().getFunctionIterator(); it.hasNext();) {
            Function function = it.next();
            String name = function.getName();
            if (caseSensitive) {
                name = name.toUpperCase();
            }

            if (name.indexOf(string) != -1) {
                functionMask[function.getID()] = value;
            }
        }
        updateRegisteredObjects("dataEvent");
    }

    public void setSelectedGroup(Group group) {
        this.selectedGroup = group;
    }

    public void closeTrialWindows() {
        updateRegisteredObjects("subWindowCloseEvent");
    }

    public Group getSelectedGroup() {
        return selectedGroup;
    }

    private void assignDefaultMetric() {
        // set the default metric to be the first one
        if (getMetrics() != null && getMetrics().size() > 0) {
            setDefaultMetric((Metric) getMetrics().get(0));
        }

        if (getMetrics() == null) {
            return;
        }

        // set the default metric to the first time based metric (if it exists)
        for (Iterator<Metric> it = getMetrics().iterator(); it.hasNext();) {
            Metric metric = it.next();
            if (metric.isTimeMetric()) {
                setDefaultMetric(metric);
                break;
            }
        }

        // set the default metric to the first metric named "Time" (if it exists), higher priority than above
        for (Iterator<Metric> it = getMetrics().iterator(); it.hasNext();) {
            Metric metric = it.next();
            if (metric.getName().equalsIgnoreCase(("time"))) {
                setDefaultMetric(metric);
                break;
            }
        }
    }

    public void finishLoad() {

        // assign the metadata from the datasource to the trial
        trial.setMetaData(trial.getDataSource().getMetaData());
        trial.setUncommonMetaData(trial.getDataSource().getUncommonMetaData());

        // The dataSource has accumulated metrics.
        // Inside ParaProf, these need to be ParaProfMetrics.

        int numberOfMetrics = trial.getDataSource().getNumberOfMetrics();
        Vector<Metric> ppMetrics = new Vector<Metric>();
        for (int i = 0; i < numberOfMetrics; i++) {
            ParaProfMetric ppMetric = new ParaProfMetric();
            ppMetric.setName(trial.getDataSource().getMetricName(i));
            ppMetric.setID(i);
            ppMetric.setDbMetricID(trial.getDataSource().getMetric(i).getDbMetricID());
            ppMetric.setPpTrial(this);
            ppMetrics.add(ppMetric);
        }

        // mow set the dataSource's metrics.
        trial.getDataSource().setMetrics(ppMetrics);

        assignDefaultMetric();

        // set the mask
        functionMask = new boolean[getDataSource().getNumFunctions()];
        for (int i = 0; i < functionMask.length; i++) {
            functionMask[i] = true;
        }

        // set the colors
        clrChooser.setColors(this, -1);

        //Set this trial's loading flag to false.
        this.setLoading(false);

        Group derived = getGroup("TAU_CALLPATH_DERIVED");
        if (derived != null) {
            showAllExcept(derived);
        }

        // run any scripts
        for (int i = 0; i < ParaProf.scripts.size(); i++) {
            ParaProfScript pps = ParaProf.scripts.get(i);
            if (pps instanceof ParaProfTrialScript) {
                try {
                    ((ParaProfTrialScript) pps).trialLoaded(this);
                } catch (Exception e) {
                    new ParaProfErrorDialog("Exception while executing script: ", e);
                }
            }
        }
    }

    public DatabaseAPI getDatabaseAPI() {
        return dbAPI;
    }

    public void setDatabaseAPI(DatabaseAPI dbAPI) {
        this.dbAPI = dbAPI;
    }

    public void setHighlightedFunction(Function func) {
        this.highlightedFunction = func;
        updateRegisteredObjects("colorEvent");
    }

    public Function getHighlightedFunction() {
        return this.highlightedFunction;
    }

    public void toggleHighlightedFunction(Function function) {
        if (highlightedFunction == function) {
            highlightedFunction = null;
        } else {
            highlightedFunction = function;
        }
        updateRegisteredObjects("colorEvent");
    }

    public void setHighlightedGroup(Group group) {
        this.highlightedGroup = group;
        updateRegisteredObjects("colorEvent");
    }

    public Group getHighlightedGroup() {
        return highlightedGroup;
    }

    public void toggleHighlightedGroup(Group group) {
        if (highlightedGroup == group) {
            highlightedGroup = null;
        } else {
            highlightedGroup = group;
        }
        updateRegisteredObjects("colorEvent");
    }

    public void setHighlightedUserEvent(UserEvent userEvent) {
        this.highlightedUserEvent = userEvent;
        updateRegisteredObjects("colorEvent");
    }

    public UserEvent getHighlightedUserEvent() {
        return highlightedUserEvent;
    }

    public void toggleHighlightedUserEvent(UserEvent userEvent) {
        if (highlightedUserEvent == userEvent) {
            highlightedUserEvent = null;
        } else {
            highlightedUserEvent = userEvent;
        }
        updateRegisteredObjects("colorEvent");
    }

    public boolean getMonitored() {
        return monitored;
    }

    public void setMonitored(boolean monitored) {
        this.monitored = monitored;

        if (monitored == true) {

            FileMonitor fileMonitor = new FileMonitor(1000);

            List<File> files = trial.getDataSource().getFiles();

            for (Iterator<File> it = files.iterator(); it.hasNext();) {
                File file = it.next();
                fileMonitor.addFile(file);
            }

            fileMonitorListener = new FileMonitorListener() {

                public void fileChanged(File file) {
                    try {
                        while (ParaProfTrial.this.loading) {
                            java.lang.Thread.sleep(1000);
                        }
                        //                        System.err.println("fileChanged!");

                        EventQueue.invokeAndWait(new Runnable() {
                            public void run() {
                                try {
                                    if (getDataSource().reloadData()) {

                                        // set the colors
                                        clrChooser.setColors(ParaProfTrial.this, -1);

                                        updateRegisteredObjects("dataEvent");
                                    }
                                } catch (Exception e) {
                                    // eat it
                                }

                            }
                        });

                    } catch (Exception e) {
                        // eat it
                    }

                }
            };

            fileMonitor.addListener(fileMonitorListener);

        }
    }

    public Function getFunction(String function) {
        return getDataSource().getFunction(function);
    }

    public Thread getThread(int n, int c, int t) {
        return getDataSource().getThread(n, c, t);
    }

    public void updateRegisteredObjects(String inString) {
        //Set this object as changed.
        this.setChanged();

        //Now notify observers.
        this.notifyObservers(inString);
    }

    public void addObserver(Observer o) {
        super.addObserver(o);
        obs.add(o);
    }

    public void deleteObserver(Observer o) {
        super.deleteObserver(o);
        obs.remove(o);
    }

    public List<Observer> getObservers() {
        return obs;
    }

    public Group getGroup(String name) {
        return getDataSource().getGroup(name);
    }

    public Group getGroupInclude() {
        return groupInclude;
    }

    public Group getGroupExclude() {
        return groupExclude;
    }

    public int getSelectedSnapshot() {
        return selectedSnapshot;
    }

    public void setSelectedSnapshot(int selectedSnapshot) {
        this.selectedSnapshot = selectedSnapshot;
        updateRegisteredObjects("dataEvent");
    }

    public Database getDatabase() {
        if (experiment != null) {
        	return experiment.getDatabase();
        } else if (view != null) {
            return view.getDatabase();
        }
        return null;
    }

    public List<Thread> getThreads() {
        return getDataSource().getAllThreads();
    }

    private static final String TAU_APPLICATION_NAME="TAU_APPLICATION_NAME";
    public List<String> getThreadNames() {
        List<String> threadNames = new ArrayList<String>();

        for (Iterator<Thread> it = getDataSource().getAllThreads().iterator(); it.hasNext();) {
            Thread thread = it.next();
            
            
            

            	MetaDataMap mdm = thread.getMetaData();
            	String tan="";
            	if(ParaProf.preferences.getAppNameLabels()==1&&mdm.containsKey(TAU_APPLICATION_NAME)){
            		tan = ":"+mdm.get(TAU_APPLICATION_NAME);
            	}
            
            if (getDataSource().getExecutionType() == DataSource.EXEC_TYPE_MPI) {
                threadNames.add(Integer.toString(thread.getNodeID())+tan);
            } else if (getDataSource().getExecutionType() == DataSource.EXEC_TYPE_HYBRID) {
                threadNames.add(thread.getNodeID() + ":" + thread.getThreadID()+tan);
            } else {
                threadNames.add(thread.getNodeID() + ":" + thread.getContextID() + ":" + thread.getThreadID()+tan);
            }
        }
        return threadNames;
    }

    public void deleteMetric(Metric metric) {

        trial.getDataSource().getMetrics().remove(metric);

        if (metric == defaultMetric) {
            assignDefaultMetric();
        }
    }

    public Metric[] getMetricArray() {

        int count = 0;

        for (Iterator<Metric> it = getMetrics().iterator(); it.hasNext();) {
            Metric metric = it.next();
            if (metric == null) {
                continue;
            }
            count++;
        }

        Metric[] items = new Metric[count];

        int idx = 0;

        for (Iterator<Metric> it = getMetrics().iterator(); it.hasNext();) {
            Metric metric = it.next();
            if (metric == null) {
                continue;
            }
            items[idx++] = metric;
        }

        return items;
    }

	public Vector<String> getTopologyArray() {
		Set<MetaDataKey> keys = getDataSource().getMetaData().keySet();
		Vector<String> topos = new Vector<String>();
		boolean foundTopo=false;
		for(Iterator<MetaDataKey> it = keys.iterator(); it.hasNext();){
			String key = it.next().name;
			if(key.contains(" isTorus")||key.contains(" Period")||key.contains(" Dimension")){
				int lastSpace=key.lastIndexOf(' ');
				topos.add(key.substring(0,lastSpace));
				//topos.add(key.split(" ")[0]);
				foundTopo=true;
			}
			
		}
		
		if(!foundTopo){
			Thread base =getDataSource().getThread(0, 0, 0);
			if(base==null){
				/*
				 * If we don't have 0,0,0 get a list of all the threads. If any thread has more metadata than the first we assume that's the one with the core metadata. If any has less than the first we assume the first has the core metadata.
				 */
				List<Thread> threads = getDataSource().getThreads();
				
				Iterator<Thread> threadIt = threads.iterator();
				if(!threadIt.hasNext()){
					return topos;
				}
				Thread currentThread=threadIt.next();
				base = currentThread;
				int num=currentThread.getMetaData().keySet().size();
				while(threadIt.hasNext()){
					
					currentThread=threadIt.next();
					
					int tmp =currentThread.getMetaData().keySet().size();
					if(tmp>num){
						base=currentThread;
						break;
					}else if(tmp<num){
						
						break;
					}
				}
			}
			if(base==null){
				return topos;
			}
			boolean foundCray=false;
			keys = base.getMetaData().keySet();
			for(Iterator<MetaDataKey> it = keys.iterator(); it.hasNext();){
				String key = it.next().name;
				if((key.startsWith("Cray")&&key.contains("Nodename")))
				{
					topos.add(key.split(" ")[0]);
					foundTopo=true;
				}
				if(!foundCray&&key.startsWith("CRAY_PMI")){
					topos.add("CRAY_PMI");
					foundTopo=true;
					foundCray=true;
				}
			}
		}
		
//		if(topos.size()==0){
//			return new String[]{null};
//		}
		
//		topos.add("Sphere");
//		topos.add("Custom");
		
		//String[] a = new String[topos.size()];
		return topos;//topos.toArray(a);
	}

}