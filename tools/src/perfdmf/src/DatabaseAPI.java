package edu.uoregon.tau.perfdmf;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Vector;

import edu.uoregon.tau.perfdmf.database.ConnectionManager;
import edu.uoregon.tau.perfdmf.database.DB;

/**
 * This is the top level class for the Database API.
 * 
 * <P>
 * CVS $Id: DatabaseAPI.java,v 1.29 2009/12/14 23:07:49 amorris Exp $
 * </P>
 * 
 * @author Kevin Huck, Robert Bell
 * @version $Revision: 1.29 $
 */
public class DatabaseAPI {

    private Application application = null;
    private Experiment experiment = null;
    private Trial trial = null;
    private List<Integer> nodes = null;
    private List<Integer> contexts = null;
    private List<Integer> threads = null;
    private Vector<IntervalEvent> intervalEvents = null;
    private List<Metric> metrics = null;
    private Vector<IntervalLocationProfile> intervalEventData = null;
    private Vector<AtomicEvent> atomicEvents = null;
    private Vector<AtomicLocationProfile> atomicEventData = null;

    // from datasession
    private DB db = null;
    private ConnectionManager connector;
    private Hashtable<Integer, IntervalEvent> intervalEventHash = null;
    private Hashtable<Integer, AtomicEvent> atomicEventHash = null;
    //private String configFileName = null;

    private boolean cancelUpload = false;

    private Database database;

    public void cancelUpload() {
        this.cancelUpload = true;
    }

    public String getMetricName(int metricID) {
        if (this.metrics == null) {
            if (this.trial != null) {
                this.metrics = this.trial.getMetrics();
            }
        }

        //Try getting the metric name.
        if ((this.metrics != null) && (metricID < this.metrics.size()))
            return this.metrics.get(metricID).getName();
        else
            return null;
    }

    public void setApplication(Application application) {
        this.application = application;
    }

    public void setExperiment(Experiment experiment) {
        this.experiment = experiment;
    }

    public DatabaseAPI() {}

    public DB db() {
        return db;
    }

    public void setDB(DB db) {
        this.db = db;
    }

    // Initialization / termination routines

    public void initialize(String configFile, boolean prompt) throws SQLException {
        if (configFile.startsWith("http") || (new java.io.File(configFile).exists())) {
            initialize(new Database(configFile), prompt);
        } else {
            System.err.println("Could not find file: " + configFile);
        }
    }
    public void initialize(String configFile, boolean prompt, String dbName) throws SQLException {
        if (configFile.startsWith("http") || (new java.io.File(configFile).exists())) {
            initialize(new Database(dbName,configFile), prompt);
        } else {
            System.err.println("Could not find file: " + configFile);
        }
    }

    public void initialize(Database database, boolean prompt) throws SQLException {
        this.database = database;
        connector = new ConnectionManager(database, prompt);
        connector.connect();
        db = connector.getDB();
        Application.getMetaData(db);
        Experiment.getMetaData(db);
        Trial.getMetaData(db);
    }

    public void initialize(Database database, String password) throws SQLException {
        this.database = database;
        connector = new ConnectionManager(database, password);
        connector.connect();
        db = connector.getDB();
        Application.getMetaData(db);
        Experiment.getMetaData(db);
        Trial.getMetaData(db);
    }

    public void initialize(Database database) throws SQLException {
        this.database = database;
        connector = new ConnectionManager(database);
        connector.connect();
        db = connector.getDB();
        Application.getMetaData(db);
        Experiment.getMetaData(db);
        Trial.getMetaData(db);
    }

    public void terminate() {
        connector.dbclose();
    }

    public ConnectionManager getConnector() {
        return connector;
    }

    // returns Vector of ALL Application objects
    public List<Application> getApplicationList() throws DatabaseException {
        String whereClause = "";
        return Application.getApplicationList(db, whereClause);
    }

    // returns Vector of Experiment objects
    public List<Experiment> getExperimentList() throws DatabaseException {

        String whereClause = "";
        if (application != null)
            whereClause = "WHERE application = " + application.getID();
        return Experiment.getExperimentList(db, whereClause);

    }

    // returns Vector of Trial objects
    public List<Trial> getTrialList(boolean getMetadata) {
        StringBuffer whereClause = new StringBuffer();
        if (experiment != null) {
            whereClause.append("WHERE t.experiment = " + experiment.getID());
        } else if (application != null) {
            whereClause.append("WHERE e.application = " + application.getID());
        }
        return Trial.getTrialList(db, whereClause.toString(), getMetadata);
    }

    // set the Application for this session
    public Application setApplication(int id) {
        this.application = null;
        this.experiment = null;
        this.trial = null;
        this.intervalEventHash = null;
        this.atomicEventHash = null;
        // create a string to hit the database
        String whereClause = " WHERE id = " + id;
        Vector<Application> applications = Application.getApplicationList(db, whereClause);
        if (applications.size() == 1) {
            this.application = applications.elementAt(0);
        } // else exception?
        return this.application;
    }

    // set the Application for this session
    public Application setApplication(String name, String version) {
        this.application = null;
        this.experiment = null;
        this.trial = null;
        this.intervalEventHash = null;
        this.atomicEventHash = null;
        // create a string to hit the database
        StringBuffer whereClause = new StringBuffer();
        whereClause.append(" WHERE name = '" + name + "'");
        if (version != null) {
            whereClause.append(" AND version = " + version);
        }
        Vector<Application> applications = Application.getApplicationList(db, whereClause.toString());
        if (applications.size() == 1) {
            this.application = applications.elementAt(0);
            return this.application;
        } else {
            return null;
        }
    }

    // set the Experiment for this session
    public Experiment setExperiment(int id) throws DatabaseException {
        this.experiment = null;
        this.trial = null;
        this.intervalEventHash = null;
        this.atomicEventHash = null;
        // create a string to hit the database
        String whereClause;
        whereClause = " WHERE id = " + id;
        Vector<Experiment> experiments = Experiment.getExperimentList(db, whereClause);
        if (experiments.size() == 1) {
            this.experiment = experiments.elementAt(0);
            return this.experiment;
        } else {
            return null;
        }
    }

    // set the Trial for this session
    public Trial setTrial(int id, boolean getXMLMetadata) {
        return setTrial(id, true, getXMLMetadata);
    }

    private Trial setTrial(int id, boolean clearHashes, boolean getXMLMetadata) {
        this.trial = null;
        this.metrics = null;
        if (clearHashes) {
            this.intervalEventHash = null;
            this.atomicEventHash = null;
        }
        // create a string to hit the database
        String whereClause;
        whereClause = " WHERE t.id = " + id;
        Vector<Trial> trials = Trial.getTrialList(db, whereClause, getXMLMetadata);
        if (trials.size() == 1) {
            this.trial = trials.elementAt(0);
        } //else exception?

        return this.trial;
    }

    // set the Trial for this session
    public Trial setTrial(String trialName, boolean getXMLMetadata) {
        return setTrial(trialName, true, getXMLMetadata);
    }

    private Trial setTrial(String trialName, boolean clearHashes, boolean getXMLMetadata) {
        this.trial = null;
        this.metrics = null;
        if (clearHashes) {
            this.intervalEventHash = null;
            this.atomicEventHash = null;
        }
        // create a string to hit the database
        String whereClause;
        whereClause = " WHERE t.name = '" + trialName + "'";
        Vector<Trial> trials = Trial.getTrialList(db, whereClause, getXMLMetadata);
        if (trials.size() == 1) {
            this.trial = trials.elementAt(0);
        } //else exception?

        return this.trial;
    }

    // returns a List of IntervalEvents
    public List<IntervalEvent> getIntervalEvents() {
        String whereClause = new String();
        if (trial != null) {
            whereClause = " WHERE trial = " + trial.getID();
        } else if (experiment != null) {
            whereClause = " WHERE experiment = " + experiment.getID();
        } else if (application != null) {
            whereClause = " WHERE application = " + application.getID();
        }

        intervalEvents = IntervalEvent.getIntervalEvents(this, db, whereClause);

        if (intervalEventHash == null) {
            intervalEventHash = new Hashtable<Integer, IntervalEvent>();
        }
        IntervalEvent fun;
        for (Enumeration<IntervalEvent> en = intervalEvents.elements(); en.hasMoreElements();) {
            fun = en.nextElement();
            intervalEventHash.put(new Integer(fun.getID()), fun);
        }
        return intervalEvents;
    }

    // gets the mean & total data for a intervalEvent
    public void getIntervalEventDetail(IntervalEvent intervalEvent) throws SQLException {
        StringBuffer buf = new StringBuffer();
        buf.append(" WHERE ms.interval_event = " + intervalEvent.getID());
        if (metrics != null && metrics.size() > 0) {
            buf.append(" AND ms.metric in (");
            Metric metric;
            for (Iterator<Metric> en = metrics.iterator(); en.hasNext();) {
                metric = en.next();
                buf.append(metric.getID());
                if (en.hasNext()) {
                    buf.append(", ");
                } else {
                    buf.append(") ");
                }
            }
        }
        IntervalLocationProfile.getIntervalEventDetail(db, intervalEvent, buf.toString());
    }

    // gets the mean & total data for a atomicEvent
    public void getAtomicEventDetail(AtomicEvent atomicEvent) {
        StringBuffer buf = new StringBuffer();
        buf.append(" WHERE e.id = " + atomicEvent.getID());
        AtomicLocationProfile.getAtomicEventDetail(db, atomicEvent, buf.toString());
    }

    // returns a List of AtomicEvents
    public List<AtomicEvent> getAtomicEvents() {
        String whereClause = new String();
        if (trial != null) {
            whereClause = " WHERE t.id = " + trial.getID();
        } else if (experiment != null) {
            whereClause = " WHERE t.experiment = " + experiment.getID();
        } else if (application != null) {
            whereClause = " WHERE e.application = " + application.getID();
        }
        atomicEvents = AtomicEvent.getAtomicEvents(this, db, whereClause);
        if (atomicEventHash == null)
            atomicEventHash = new Hashtable<Integer, AtomicEvent>();
        AtomicEvent ue;
        for (Enumeration<AtomicEvent> en = atomicEvents.elements(); en.hasMoreElements();) {
            ue = en.nextElement();
            atomicEventHash.put(new Integer(ue.getID()), ue);
        }
        return atomicEvents;
    }

    // sets the current intervalEvent
    public IntervalEvent setIntervalEvent(int id) {
        IntervalEvent intervalEvent = null;
        this.intervalEvents = new Vector<IntervalEvent>();
        intervalEvent = getIntervalEvent(id);
        if (intervalEvent != null) {
            this.intervalEvents.addElement(intervalEvent);
            // we need this to get the metric data...
            setTrial(intervalEvent.getTrialID(), false);
        }
        return intervalEvent;
    }

    // sets the current user event
    public AtomicEvent setAtomicEvent(int id) {
        AtomicEvent atomicEvent = null;
        this.atomicEvents = new Vector<AtomicEvent>();
        atomicEvent = getAtomicEvent(id);
        if (atomicEvent != null)
            this.atomicEvents.addElement(atomicEvent);
        return atomicEvent;
    }

    // clears the interval event selection
    public void clearIntervalEvents() {
        this.intervalEvents = null;
        return;
    }

    // sets the current node ID
    public void setNode(int id) {
        Integer node = new Integer(id);
        this.nodes = new ArrayList<Integer>();
        this.nodes.add(node);
        return;
    }

    // sets the current context ID
    public void setContext(int id) {
        Integer context = new Integer(id);
        this.contexts = new ArrayList<Integer>();
        this.contexts.add(context);
        return;
    }

    // sets the current thread ID
    public void setThread(int id) {
        Integer thread = new Integer(id);
        this.threads = new ArrayList<Integer>();
        this.threads.add(thread);
        return;
    }

    public List<IntervalLocationProfile> getIntervalEventData() throws SQLException {
        // check to make sure this is a meaningful request
        if (trial == null && intervalEvents == null) {
            System.out.println("Please select a trial or a set of intervalEvents before getting intervalEvent data.");
            return null;
        }

        // get the hash of intervalEvent names first
        //if (intervalEvents == null) {
        //getIntervalEvents();
        //}

        // get the metric count
        int metricCount = 0;
        if (metrics != null && metrics.size() > 0) {
            metricCount = metrics.size();
        } else {
            metricCount = trial.getMetricCount();
        }

        // create a string to hit the database
        boolean gotWhere = false;
        StringBuffer buf = new StringBuffer();
        if (trial != null) {
            buf.append(" WHERE e.trial = " + trial.getID());
            gotWhere = true;
        }
        if (intervalEvents != null && intervalEvents.size() > 0) {
            if (gotWhere)
                buf.append(" AND p.interval_event in (");
            else
                buf.append(" WHERE p.interval_event in (");
            IntervalEvent intervalEvent;
            for (Enumeration<IntervalEvent> en = intervalEvents.elements(); en.hasMoreElements();) {
                intervalEvent = en.nextElement();
                buf.append(intervalEvent.getID());
                if (en.hasMoreElements())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (nodes != null && nodes.size() > 0) {
            buf.append(" AND p.node IN (");
            Integer node;
            for (Iterator<Integer> iter = nodes.iterator(); iter.hasNext();) {
                node = iter.next();
                buf.append(node);
                if (iter.hasNext())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (contexts != null && contexts.size() > 0) {
            buf.append(" AND p.context IN (");
            Integer context;
            for (Iterator<Integer> iter = contexts.iterator(); iter.hasNext();) {
                context = iter.next();
                buf.append(context);
                if (iter.hasNext())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (threads != null && threads.size() > 0) {
            buf.append(" AND p.thread IN (");
            Integer thread;
            for (Iterator<Integer> iter = threads.iterator(); iter.hasNext();) {
                thread = iter.next();
                buf.append(thread);
                if (iter.hasNext())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (metrics != null && metrics.size() > 0) {
            buf.append(" AND p.metric IN (");
            Metric metric;
            for (Iterator<Metric> en = metrics.iterator(); en.hasNext();) {
                metric = en.next();
                buf.append(metric.getID());
                if (en.hasNext())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        intervalEventData = IntervalLocationProfile.getIntervalEventData(db, metricCount, buf.toString());
        return intervalEventData;
    }

    public List<AtomicLocationProfile> getAtomicEventData() {
        // check to make sure this is a meaningful request
        if (trial == null && atomicEvents == null) {
            System.out.println("Please select a trial or a set of user events before getting user event data.");
            return null;
        }

        // get the hash of atomicEvent names first
        if (atomicEvents == null)
            getAtomicEvents();

        boolean gotWhere = false;
        StringBuffer buf = new StringBuffer();
        if (trial != null) {
            buf.append(" WHERE e.trial = " + trial.getID());
            gotWhere = true;
        }

        if (atomicEvents != null && atomicEvents.size() > 0) {
            if (gotWhere)
                buf.append(" AND e.id IN (");
            else
                buf.append(" WHERE e.id IN (");
            AtomicEvent atomicEvent;
            for (Enumeration<AtomicEvent> en = atomicEvents.elements(); en.hasMoreElements();) {
                atomicEvent = en.nextElement();
                buf.append(atomicEvent.getID());
                if (en.hasMoreElements())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }

        if (nodes != null && nodes.size() > 0) {
            buf.append(" AND p.node IN (");
            Integer node;
            for (Iterator<Integer> iter = nodes.iterator(); iter.hasNext();) {
                node = iter.next();
                buf.append(node);
                if (iter.hasNext())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (contexts != null && contexts.size() > 0) {
            buf.append(" AND p.context IN (");
            Integer context;
            for (Iterator<Integer> iter = contexts.iterator(); iter.hasNext();) {
                context = iter.next();
                buf.append(context);
                if (iter.hasNext())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (threads != null && threads.size() > 0) {
            buf.append(" AND p.thread IN (");
            Integer thread;
            for (Iterator<Integer> iter = threads.iterator(); iter.hasNext();) {
                thread = iter.next();
                buf.append(thread);
                if (iter.hasNext())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }

        atomicEventData = AtomicLocationProfile.getAtomicEventData(db, buf.toString());
        return atomicEventData;
    }

    public IntervalEvent getIntervalEvent(int id) {
        IntervalEvent intervalEvent = null;
        if (intervalEventHash != null) {
            intervalEvent = intervalEventHash.get(new Integer(id));
        }
        if (intervalEvent == null) {
            // create a string to hit the database
            String whereClause;
            whereClause = " WHERE id = " + id;
            Vector<IntervalEvent> intervalEvents = IntervalEvent.getIntervalEvents(this, db, whereClause);
            if (intervalEvents.size() == 1) {
                intervalEvent = intervalEvents.elementAt(0);
            } //else exception?
            if (intervalEventHash == null)
                intervalEventHash = new Hashtable<Integer, IntervalEvent>();
            intervalEventHash.put(new Integer(intervalEvent.getID()), intervalEvent);
        }
        return intervalEvent;
    }

    public AtomicEvent getAtomicEvent(int id) {
        AtomicEvent atomicEvent = null;
        if (atomicEventHash != null) {
            atomicEvent = atomicEventHash.get(new Integer(id));
        }
        if (atomicEvent == null) {
            // create a string to hit the database
            String whereClause;
            whereClause = " WHERE u.id = " + id;
            Vector<AtomicEvent> atomicEvents = AtomicEvent.getAtomicEvents(this, db, whereClause);
            if (atomicEvents.size() == 1) {
                atomicEvent = atomicEvents.elementAt(0);
            } //else exception?
            if (atomicEventHash == null)
                atomicEventHash = new Hashtable<Integer, AtomicEvent>();
            atomicEventHash.put(new Integer(atomicEvent.getID()), atomicEvent);
        }
        return atomicEvent;
    }

    public int saveApplication(Application app) {
        try {
            return app.saveApplication(db);
        } catch (SQLException e) {
            throw new DatabaseException("Error saving application", e);
        }
    }

    public int saveExperiment(Experiment exp) throws DatabaseException {
        try {
            return exp.saveExperiment(db);
        } catch (SQLException e) {
            throw new DatabaseException("Error saving experiment", e);
        }
    }

    // override the saveTrial method
    public int saveTrial() {
        int newTrialID = trial.saveTrial(db);
        // Hashtable newMetHash = saveMetrics(newTrialID, trial, -1);
        // Hashtable newFunHash = saveIntervalEvents(newTrialID, newMetHash,
        // -1);
        // saveIntervalEventData(newFunHash, newMetHash, -1);
        // Hashtable newUEHash = saveAtomicEvents(newTrialID);
        // saveAtomicEventData(newUEHash);
        return newTrialID;
    }

    private int saveMetric(int trialID, Metric metric) throws SQLException {
        int newMetricID = 0;
        PreparedStatement stmt1 = null;
        stmt1 = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix() + "metric (name, trial) VALUES (?, ?)");
        stmt1.setString(1, metric.getName());
        stmt1.setInt(2, trialID);
        stmt1.executeUpdate();
        stmt1.close();

        String tmpStr = new String();
        if (db.getDBType().compareTo("mysql") == 0)
            tmpStr = "select LAST_INSERT_ID();";
        else if (db.getDBType().compareTo("db2") == 0)
            tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
        else if (db.getDBType().compareTo("derby") == 0)
            tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
        else if (db.getDBType().compareTo("oracle") == 0)
            tmpStr = "select " + db.getSchemaPrefix() + "metric_id_seq.currval FROM dual";
        else
            tmpStr = "select currval('metric_id_seq');";
        newMetricID = Integer.parseInt(db.getDataItem(tmpStr));
        return newMetricID;
    }

    private Hashtable<Integer, Integer> saveMetrics(int newTrialID, Trial trial, int saveMetricIndex) throws SQLException {
        Hashtable<Integer, Integer> metricHash = new Hashtable<Integer, Integer>();
        int idx = 0;
        for (Iterator<Metric> it = trial.getDataSource().getMetrics().iterator(); it.hasNext();) {
            Metric metric = it.next();
            int newMetricID = -1;
            if (saveMetricIndex < 0 || saveMetricIndex == idx) {
                newMetricID = saveMetric(newTrialID, metric);
            }
            metricHash.put(new Integer(idx), new Integer(newMetricID));
            idx++;
        }
        return metricHash;
    }

    private Hashtable<Integer, Integer> saveIntervalEvents(int newTrialID, Hashtable<Integer, Integer> newMetHash, int saveMetricIndex) throws SQLException {
        //      System.out.print("Saving the intervalEvents: ");
        Hashtable<Integer, Integer> newFunHash = new Hashtable<Integer, Integer>();
        Enumeration<IntervalEvent> en = intervalEvents.elements();
        IntervalEvent intervalEvent;
        //int count = 0;
        while (en.hasMoreElements()) {
            intervalEvent = en.nextElement();
            int newIntervalEventID = intervalEvent.saveIntervalEvent(db, newTrialID, newMetHash, saveMetricIndex);
            newFunHash.put(new Integer(intervalEvent.getID()), new Integer(newIntervalEventID));
            //System.out.print("\rSaving the intervalEvents: " + ++count + " records saved...");
            //DatabaseAPI.itemsDone++;

        }
        //     System.out.print("\n");
        return newFunHash;
    }

    private Hashtable<Integer, Integer> saveAtomicEvents(int newTrialID) {
        //        System.out.print("Saving the user events:");
        Hashtable<Integer, Integer> newUEHash = new Hashtable<Integer, Integer>();
        Enumeration<AtomicEvent> en = atomicEvents.elements();
        //int count = 0;
        AtomicEvent atomicEvent;
        while (en.hasMoreElements()) {
            atomicEvent = en.nextElement();
            int newAtomicEventID = atomicEvent.saveAtomicEvent(db, newTrialID);
            newUEHash.put(new Integer(atomicEvent.getID()), new Integer(newAtomicEventID));
            //System.out.print("\rSaving the user events: " + ++count + " records saved...");
            //DatabaseAPI.itemsDone++;

        }
        //    System.out.print("\n");
        return newUEHash;
    }

    private void saveAtomicEventData(Hashtable<Integer, Integer> newUEHash) {
        //    System.out.print("Saving the user event data:");
        Enumeration<AtomicLocationProfile> en = atomicEventData.elements();
        AtomicLocationProfile uedo;
        //int count = 0;
        while (en.hasMoreElements()) {
            uedo = en.nextElement();
            Integer newAtomicEventID = newUEHash.get(new Integer(uedo.getAtomicEventID()));
            uedo.saveAtomicEventData(db, newAtomicEventID.intValue());
            //       System.out.print("\rSaving the user event data: " + ++count + " records saved...");
        }
        //     System.out.print("\n");
    }

    /**
     * Saves the Trial.
     * 
     * @param trial
     * @return database index ID of the saved trial record
     */
    public int saveTrial(Trial trial) {
        return trial.saveTrial(db);
    }

    /**
     * Saves the IntervalEvent.
     * 
     * @param intervalEvent
     * @param newTrialID
     * @param newMetHash
     * @return database index ID of the saved intervalEvent record
     */
    public int saveIntervalEvent(IntervalEvent intervalEvent, int newTrialID, Hashtable<Integer, Integer> newMetHash) throws SQLException {
        return intervalEvent.saveIntervalEvent(db, newTrialID, newMetHash, -1);
    }

    /**
     * Saves the AtomicEvent object.
     * 
     * @param atomicEvent
     * @return database index ID of the saved user_event record
     */
    public int saveAtomicEvent(AtomicEvent atomicEvent, int newTrialID) {
        return atomicEvent.saveAtomicEvent(db, newTrialID);
    }

    /**
     * Saves the atomicEventData object.
     * 
     * @param atomicEventData
     * @return database index ID of the saved atomic_location_profile record
     */
    public void saveAtomicEventData(AtomicLocationProfile atomicEventData, int newAtomicEventID) {
        atomicEventData.saveAtomicEventData(db, newAtomicEventID);
        return;
    }

    // this stuff is a total hack to get some functionality that the new database API will have
    private int totalItems;
    private int itemsDone;

    public int getProgress() {
        if (totalItems != 0)
            return (int) ((float) itemsDone / (float) totalItems * 100);
        return 0;
    }

    /**
     * Saves the Trial object to the database
     * 
     * @param trial
     * @param saveMetricIndex
     * @return the database index ID of the saved trial record
     */

    public synchronized int saveTrial(Trial trial, Metric saveMetric) throws DatabaseException {
        //long start = System.currentTimeMillis();

        DataSource dataSource = trial.getDataSource();

        //Build an array of group names. This speeds lookup of group names.
        List<String> groupList = new ArrayList<String>();
        for (Iterator<Group> groupIterator = dataSource.getGroups(); groupIterator.hasNext();) {
            Group g = groupIterator.next();
            groupList.add(g.getName());
        }

        //String groupNames[] = new String[groupList.size()];
        //groupNames = groupList.toArray(groupNames);

        //String groupNames[] = (String[]) groupList.toArray();

        // get the metric count
        metrics = trial.getDataSource().getMetrics();
        int metricCount = metrics.size();

        // create the Vectors to store the data
        intervalEvents = new Vector<IntervalEvent>();
        intervalEventData = new Vector<IntervalLocationProfile>();
        atomicEvents = new Vector<AtomicEvent>();
        atomicEventData = new Vector<AtomicLocationProfile>();

        //int fcount = 0;
        //int ucount = 0;

        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

        // create the intervalEvents
        for (Iterator<Function> it = dataSource.getFunctions(); it.hasNext();) {
            Function f = it.next();
            if (!f.isGroupMember(derived)) {
                // create a intervalEvent
                IntervalEvent intervalEvent = new IntervalEvent(this);
                intervalEvent.setName(f.getName());
                intervalEvent.setID(f.getID());
                // intervalEvent.setTrialID(newTrialID);
                // build the group name
                List<Group> groups = f.getGroups();
                StringBuffer buf = new StringBuffer();
                if (groups != null) {
                    for (int i = 0; i < groups.size(); i++) {
                        if (i > 0) {
                            buf.append("|");
                        }
                        buf.append(groups.get(i).getName());
                    }

                    if (groups.size() > 0) {
                        intervalEvent.setGroup(buf.toString());
                    }
                }

                // put the intervalEvent in the vector
                intervalEvents.add(intervalEvent);

                int numThreads = trial.getDataSource().getAllThreads().size();

                IntervalLocationProfile ilpTotal = new IntervalLocationProfile(metricCount);
                IntervalLocationProfile ilpMean = new IntervalLocationProfile(metricCount);
                for (int i = 0; i < metricCount; i++) {
                    ilpTotal.setNumCalls(f.getTotalNumCalls());
                    ilpTotal.setNumSubroutines(f.getTotalNumSubr());
                    ilpTotal.setInclusivePercentage(i, f.getTotalInclusivePercent(i));
                    ilpTotal.setInclusive(i, f.getTotalInclusive(i));
                    ilpTotal.setExclusivePercentage(i, f.getTotalExclusivePercent(i));
                    ilpTotal.setExclusive(i, f.getTotalExclusive(i));
                    ilpTotal.setInclusivePerCall(i, f.getTotalInclusivePerCall(i));
                    ilpMean.setNumCalls(f.getTotalNumCalls() / numThreads);
                    ilpMean.setNumSubroutines(f.getTotalNumSubr() / numThreads);
                    ilpMean.setInclusivePercentage(i, f.getTotalInclusivePercent(i));
                    ilpMean.setInclusive(i, f.getTotalInclusive(i) / numThreads);
                    ilpMean.setExclusivePercentage(i, f.getTotalExclusivePercent(i));
                    ilpMean.setExclusive(i, f.getTotalExclusive(i) / numThreads);
                    ilpMean.setInclusivePerCall(i, f.getTotalInclusivePerCall(i));
                }
                intervalEvent.setTotalSummary(ilpTotal);
                intervalEvent.setMeanSummary(ilpMean);
            }
        }

        // create the user events
        for (Iterator<UserEvent> it = dataSource.getUserEvents(); it.hasNext();) {
            UserEvent ue = it.next();
            if (ue != null) {
                AtomicEvent atomicEvent = new AtomicEvent(this);
                atomicEvent.setName(ue.getName());
                atomicEvent.setID(ue.getID());
                atomicEvents.add(atomicEvent);
            }
        }

        for (Iterator<Thread> it = trial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            edu.uoregon.tau.perfdmf.Thread thread = it.next();
            List<FunctionProfile> intervalEvents = thread.getFunctionProfiles();

            // create interval location profiles
            for (Iterator<FunctionProfile> e4 = intervalEvents.iterator(); e4.hasNext();) {
                FunctionProfile fp = e4.next();
                if (fp != null && !fp.getFunction().isGroupMember(derived)) {
                    IntervalLocationProfile ilp = new IntervalLocationProfile(metricCount);
                    ilp.setNode(thread.getNodeID());
                    ilp.setContext(thread.getContextID());
                    ilp.setThread(thread.getThreadID());
                    ilp.setIntervalEventID(fp.getFunction().getID());
                    ilp.setNumCalls(fp.getNumCalls());
                    ilp.setNumSubroutines(fp.getNumSubr());
                    for (int i = 0; i < metricCount; i++) {
                        ilp.setInclusive(i, fp.getInclusive(i));
                        ilp.setExclusive(i, fp.getExclusive(i));
                        ilp.setInclusivePercentage(i, fp.getInclusivePercent(i));
                        ilp.setExclusivePercentage(i, fp.getExclusivePercent(i));
                        ilp.setInclusivePerCall(i, fp.getInclusivePerCall(i));
                    }
                    intervalEventData.add(ilp);
                }
            }

            // create atomic events

            for (Iterator<UserEventProfile> e4 = thread.getUserEventProfiles(); e4.hasNext();) {
                UserEventProfile uep = e4.next();
                if (uep != null) {

                    AtomicLocationProfile udo = new AtomicLocationProfile();
                    udo.setAtomicEventID(uep.getUserEvent().getID());
                    udo.setNode(thread.getNodeID());
                    udo.setContext(thread.getContextID());
                    udo.setThread(thread.getThreadID());
                    udo.setSampleCount((int) uep.getNumSamples());
                    udo.setMaximumValue(uep.getMaxValue());
                    udo.setMinimumValue(uep.getMinValue());
                    udo.setMeanValue(uep.getMeanValue());
                    udo.setSumSquared(uep.getSumSquared());
                    atomicEventData.add(udo);
                }
            }
        }

        totalItems = intervalEvents.size() + intervalEventData.size() + atomicEvents.size() + atomicEventData.size();

        // Now upload to the database

        try {
            db.setAutoCommit(false);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: couldn't set AutoCommit to false", e);
        }

        int newTrialID = 0;

        int saveMetricIndex = -1;
        if (saveMetric != null) {
            saveMetricIndex = saveMetric.getID();
        }

        try {
            // output the trial data, which also saves the intervalEvents,
            // intervalEvent data, user events and user event data

            Hashtable<Integer, Integer> metricHash = null;
            if (saveMetric == null) { // this means save the whole thing???
                newTrialID = trial.saveTrial(db);
                trial.setID(newTrialID);
                metricHash = saveMetrics(newTrialID, trial, saveMetricIndex);//Commit here

                if (intervalEvents != null && intervalEvents.size() > 0) {
                    Hashtable<Integer, Integer> functionHash = saveIntervalEvents(newTrialID, metricHash, saveMetricIndex);
                    saveIntervalLocationProfiles(db, functionHash, intervalEventData.elements(), metricHash, saveMetricIndex);//Commit here, internally per process/thread
                }
                if (atomicEvents != null && atomicEvents.size() > 0) {
                    Hashtable<Integer, Integer> atomicEventHash = saveAtomicEvents(newTrialID);
                    if (atomicEventData != null && atomicEventData.size() > 0) {
                        saveAtomicEventData(atomicEventHash);
                    }
                }

            } else {
                newTrialID = trial.getID();
                metricHash = saveMetrics(newTrialID, trial, saveMetricIndex);

                if (intervalEvents != null && intervalEvents.size() > 0) {
                    Hashtable<Integer, Integer> newFunHash = saveIntervalEvents(newTrialID, metricHash, saveMetricIndex);
                    saveIntervalLocationProfiles(db, newFunHash, intervalEventData.elements(), metricHash, saveMetricIndex);

                }
            }

            for (Iterator<Integer> it = metricHash.keySet().iterator(); it.hasNext();) {
                Integer key = it.next();
                int value = metricHash.get(key).intValue();

                for (Iterator<Metric> it2 = trial.getDataSource().getMetrics().iterator(); it2.hasNext();) {
                    Metric metric = it2.next();
                    if (metric.getID() == key.intValue()) {
                        if (value != -1) {
                            metric.setDbMetricID(value);
                        }
                    }
                }
            }

        } catch (SQLException e) {
            try {
                db.rollback();
                e.printStackTrace();
                throw new DatabaseException("Saving Trial Failed, rollbacks successful", e);
            } catch (SQLException e2) {
                throw new DatabaseException("Saving Trial Failed, rollbacks failed!", e2);
            }

        }

        try {
            db.commit();
            db.setAutoCommit(true);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: commit failed!", e);
        }

        //long stop = System.currentTimeMillis();
        //long elapsedMillis = stop - start;
        //double elapsedSeconds = (double) (elapsedMillis) / 1000.0;
        //System.out.println("Elapsed time: " + elapsedSeconds + " seconds.");
        return newTrialID;
    }

    private Map<Metric, Integer> uploadMetrics(int trialID, DataSource dataSource) throws SQLException {
        Map<Metric, Integer> map = new HashMap<Metric, Integer>();

        for (Iterator<Metric> it = dataSource.getMetrics().iterator(); it.hasNext();) {
            Metric metric = it.next();

            PreparedStatement stmt = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                    + "metric (name, trial) VALUES (?, ?)");
            stmt.setString(1, metric.getName());
            stmt.setInt(2, trialID);
            stmt.executeUpdate();
            stmt.close();

            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0)
                tmpStr = "select LAST_INSERT_ID();";
            else if (db.getDBType().compareTo("db2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
            else if (db.getDBType().compareTo("derby") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM metric";
            else if (db.getDBType().compareTo("oracle") == 0)
                tmpStr = "select " + db.getSchemaPrefix() + "metric_id_seq.currval FROM dual";
            else
                tmpStr = "select currval('metric_id_seq');";
            int dbMetricID = Integer.parseInt(db.getDataItem(tmpStr));
            map.put(metric, new Integer(dbMetricID));
        }
        return map;
    }

    // fills the interval event table
    private Map<Function, Integer> uploadFunctions(int trialID, DataSource dataSource) throws SQLException {
        Map<Function, Integer> map = new HashMap<Function, Integer>();

        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
        for (Iterator<Function> it = dataSource.getFunctions(); it.hasNext();) {
            Function f = it.next();
            if (f.isGroupMember(derived)) {
                continue;
            }

            String group = null;
            List<Group> groups = f.getGroups();
            StringBuffer allGroups = new StringBuffer();
            if (groups != null) {
                for (int i = 0; i < groups.size(); i++) {
                    if (i > 0)
                        allGroups.append("|");
                    allGroups.append(groups.get(i).getName());
                }
                if (groups.size() > 0)
                    group = allGroups.toString();
            }

            PreparedStatement statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                    + "interval_event (trial, name, group_name) VALUES (?, ?, ?)");
            statement.setInt(1, trialID);
            statement.setString(2, f.getName());
            statement.setString(3, group);
            statement.executeUpdate();
            statement.close();

            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0)
                tmpStr = "select LAST_INSERT_ID();";
            else if (db.getDBType().compareTo("db2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
            else if (db.getDBType().compareTo("derby") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM interval_event";
            else if (db.getDBType().compareTo("oracle") == 0)
                tmpStr = "select " + db.getSchemaPrefix() + "interval_event_id_seq.currval FROM dual";
            else
                tmpStr = "select currval('interval_event_id_seq');";
            int newIntervalEventID = Integer.parseInt(db.getDataItem(tmpStr));

            map.put(f, new Integer(newIntervalEventID));

            this.itemsDone++;
        }
        return map;
    }

    // fills the interval event table
    private Map<UserEvent, Integer> uploadUserEvents(int trialID, DataSource dataSource) throws SQLException {
        Map<UserEvent, Integer> map = new HashMap<UserEvent, Integer>();

        String group = null; // no groups right now?

        for (Iterator<UserEvent> it = dataSource.getUserEvents(); it.hasNext();) {
            UserEvent ue = it.next();

            PreparedStatement statement = null;
            statement = db.prepareStatement("INSERT INTO " + db.getSchemaPrefix()
                    + "atomic_event (trial, name, group_name) VALUES (?, ?, ?)");
            statement.setInt(1, trialID);
            statement.setString(2, ue.getName());
            statement.setString(3, group);
            statement.executeUpdate();
            statement.close();

            String tmpStr = new String();
            if (db.getDBType().compareTo("mysql") == 0)
                tmpStr = "select LAST_INSERT_ID();";
            else if (db.getDBType().compareTo("db2") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM atomic_event";
            else if (db.getDBType().compareTo("derby") == 0)
                tmpStr = "select IDENTITY_VAL_LOCAL() FROM atomic_event";
            else if (db.getDBType().compareTo("oracle") == 0)
                tmpStr = "select " + db.getSchemaPrefix() + "atomic_event_id_seq.currval FROM dual";
            else
                tmpStr = "select currval('atomic_event_id_seq');";
            int newAtomicEventID = Integer.parseInt(db.getDataItem(tmpStr));
            map.put(ue, new Integer(newAtomicEventID));

            this.itemsDone++;
        }
        return map;
    }

    private void addBatchFunctionProfile(PreparedStatement stmt, Thread thread, int metricID, int dbMetricID, FunctionProfile fp,
            int intervalEventID, boolean createMean, int numThreads) throws SQLException {

        stmt.setInt(1, intervalEventID);
        stmt.setInt(2, dbMetricID);
        stmt.setDouble(3, fp.getInclusivePercent(metricID));

        if (createMean) {
            stmt.setDouble(4, fp.getInclusive(metricID) / numThreads);
            stmt.setDouble(5, fp.getExclusivePercent(metricID));
            stmt.setDouble(6, fp.getExclusive(metricID) / numThreads);
            stmt.setDouble(7, fp.getNumCalls() / numThreads);
            stmt.setDouble(8, fp.getNumSubr() / numThreads);
            stmt.setDouble(9, fp.getInclusivePerCall(metricID));
        } else {
            stmt.setDouble(4, fp.getInclusive(metricID));
            stmt.setDouble(5, fp.getExclusivePercent(metricID));
            stmt.setDouble(6, fp.getExclusive(metricID));
            stmt.setDouble(7, fp.getNumCalls());
            stmt.setDouble(8, fp.getNumSubr());
            stmt.setDouble(9, fp.getInclusivePerCall(metricID));
        }
        if (thread.getNodeID() >= 0) {
            stmt.setInt(10, thread.getNodeID());
            stmt.setInt(11, thread.getContextID());
            stmt.setInt(12, thread.getThreadID());
        }

        this.itemsDone++;

        //stmt.addBatch();
        //        try {
        stmt.executeUpdate();
        //        } catch (Exception e) {
        //            e.printStackTrace();
        //            System.out.println(e);
        //            System.out.println(stmt.toString());
        //            System.out.println("exclusive: " + fp.getExclusive(metricID));
        //            System.out.println("exclusive percent: " + fp.getExclusivePercent(metricID));
        //            System.out.println("inclusive: " + fp.getInclusive(metricID));
        //            System.out.println("inclusive percent: " + fp.getExclusivePercent(metricID));
        //            System.out.println("numThreads: " + numThreads);
        //            System.out.println("numcalls: " + fp.getNumCalls());
        //            System.out.println("numsubr: " + fp.getNumSubr());
        //            System.out.println("inclusivepercall: " + fp.getInclusivePerCall(metricID));
        //        }
    }

    private void uploadFunctionProfiles(int trialID, DataSource dataSource, Map<Function, Integer> functionMap, Map<Metric, Integer> metricMap, boolean summaryOnly)
            throws SQLException {

        PreparedStatement totalInsertStatement = null;
        PreparedStatement meanInsertStatement = null;
        PreparedStatement threadInsertStatement = null;

        if (db.getDBType().compareTo("oracle") == 0) {
            totalInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_total_summary (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, excl, call, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
            meanInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_mean_summary (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, excl, call, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
            threadInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_location_profile (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, excl, call, subroutines, inclusive_per_call, node, context, thread) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
        } else if (db.getDBType().compareTo("derby") == 0) {
            totalInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_total_summary (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, num_calls, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
            meanInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_mean_summary (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, num_calls, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
            threadInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_location_profile (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, num_calls, subroutines, inclusive_per_call, node, context, thread) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
        } else if (db.getDBType().compareTo("mysql") == 0) {
            totalInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_total_summary (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, `call`, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
            meanInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_mean_summary (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, `call`, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
            threadInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_location_profile (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, `call`, subroutines, inclusive_per_call, node, context, thread) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
        } else {
            totalInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_total_summary (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
            meanInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_mean_summary (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
            threadInsertStatement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_location_profile (interval_event, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call, node, context, thread) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
        }

        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

        for (Iterator<Metric> it5 = dataSource.getMetrics().iterator(); it5.hasNext();) {
            Metric metric = it5.next();
            Integer dbMetricID = metricMap.get(metric);

            for (Iterator<Function> it4 = dataSource.getFunctions(); it4.hasNext();) {
                Function function = it4.next();
                if (function.isGroupMember(derived)) {
                    continue;
                }
                Integer intervalEventID = functionMap.get(function);

                edu.uoregon.tau.perfdmf.Thread totalData = dataSource.getTotalData();
                addBatchFunctionProfile(totalInsertStatement, totalData, metric.getID(), dbMetricID.intValue(),
                        function.getTotalProfile(), intervalEventID.intValue(), false, dataSource.getAllThreads().size());

                //edu.uoregon.tau.dms.dss.Thread meanData = dataSource.getMeanData();
                //addBatchFunctionProfile(meanInsertStatement, meanData, metric.getID(), dbMetricID.intValue(),
                //       function.getMeanProfile(), intervalEventID.intValue(), false, dataSource.getAllThreads().size());

                addBatchFunctionProfile(meanInsertStatement, totalData, metric.getID(), dbMetricID.intValue(),
                        function.getTotalProfile(), intervalEventID.intValue(), true, dataSource.getAllThreads().size());

                for (Iterator<Thread> it = dataSource.getAllThreads().iterator(); it.hasNext() && summaryOnly == false;) {
                    edu.uoregon.tau.perfdmf.Thread thread = it.next();

                    FunctionProfile fp = thread.getFunctionProfile(function);
                    if (fp != null) { // only if this thread calls this function

                        if (this.cancelUpload)
                            return;

                        addBatchFunctionProfile(threadInsertStatement, thread, metric.getID(), dbMetricID.intValue(), fp,
                                intervalEventID.intValue(), false, dataSource.getAllThreads().size());
                    }
                }
            }
        }

        //        totalInsertStatement.executeBatch();
        //        meanInsertStatement.executeBatch();
        //        threadInsertStatement.executeBatch();

        totalInsertStatement.close();
        meanInsertStatement.close();
        threadInsertStatement.close();

    }

    private void uploadUserEventProfiles(int trialID, DataSource dataSource, Map<UserEvent, Integer> userEventMap) throws SQLException {

        for (Iterator<Node> it = dataSource.getNodes(); it.hasNext();) {
            Node node = it.next();
            for (Iterator<Context> it2 = node.getContexts(); it2.hasNext();) {
                Context context = it2.next();
                for (Iterator<Thread> it3 = context.getThreads(); it3.hasNext();) {
                    edu.uoregon.tau.perfdmf.Thread thread = it3.next();

                    for (Iterator<UserEventProfile> it4 = thread.getUserEventProfiles(); it4.hasNext();) {
                        UserEventProfile uep = it4.next();

                        if (this.cancelUpload)
                            return;

                        if (uep != null) {
                            int atomicEventID = userEventMap.get(uep.getUserEvent()).intValue();

                            PreparedStatement statement = null;
                            statement = db.prepareStatement("INSERT INTO "
                                    + db.getSchemaPrefix()
                                    + "atomic_location_profile (atomic_event, node, context, thread, sample_count, maximum_value, minimum_value, mean_value, standard_deviation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");
                            statement.setInt(1, atomicEventID);
                            statement.setInt(2, thread.getNodeID());
                            statement.setInt(3, thread.getContextID());
                            statement.setInt(4, thread.getThreadID());
                            statement.setInt(5, (int) uep.getNumSamples());
                            statement.setDouble(6, uep.getMaxValue());
                            statement.setDouble(7, uep.getMinValue());
                            statement.setDouble(8, uep.getMeanValue());
                            statement.setDouble(9, uep.getSumSquared());
                            statement.executeUpdate();
                            statement.close();
                        }

                    }
                }
            }
        }

    }

    private void computeUploadSize(DataSource dataSource) {
        this.totalItems = 0;

        for (Iterator<Function> it4 = dataSource.getFunctions(); it4.hasNext();) {
            //Function function = (Function) 
            it4.next();
            this.totalItems++;
        }

        int numMetrics = dataSource.getMetrics().size();

        for (Iterator<Function> it4 = dataSource.getFunctions(); it4.hasNext();) {
            Function function = it4.next();

            this.totalItems += numMetrics; // total
            this.totalItems += numMetrics; // mean

            for (Iterator<Thread> it = dataSource.getAllThreads().iterator(); it.hasNext();) {
                edu.uoregon.tau.perfdmf.Thread thread = it.next();
                FunctionProfile fp = thread.getFunctionProfile(function);
                if (fp != null) { // only if this thread calls this function
                    this.totalItems += numMetrics; // this profile
                }
            }
        }
    }

    public synchronized int uploadTrial(Trial trial) throws DatabaseException {
        return uploadTrial(trial, false);
    }

    public synchronized int uploadTrial(Trial trial, boolean summaryOnly) throws DatabaseException {
        //long start = System.currentTimeMillis();

        DataSource dataSource = trial.getDataSource();

        try {
            db.setAutoCommit(false);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: couldn't set AutoCommit to false", e);
        }

        int newTrialID = -1;

        try {
            // save the trial metadata (which returns the new id)
            newTrialID = trial.saveTrial(db);
            trial.setID(newTrialID);

            computeUploadSize(dataSource);
            // upload the metrics and get a map that maps the metrics 0 -> n-1 to their unique DB IDs (e.g. 83, 84)
            Map<Metric, Integer> metricMap = uploadMetrics(newTrialID, dataSource);

            for (Iterator<Metric> it = metricMap.keySet().iterator(); it.hasNext();) {
                Metric key = it.next();
                int value = metricMap.get(key).intValue();
                key.setDbMetricID(value);
            }

            Map<Function, Integer> functionMap = uploadFunctions(newTrialID, dataSource);

            uploadFunctionProfiles(newTrialID, dataSource, functionMap, metricMap, summaryOnly);

            Map<UserEvent, Integer> userEventMap = uploadUserEvents(newTrialID, dataSource);

            uploadUserEventProfiles(newTrialID, dataSource, userEventMap);

            if (this.cancelUpload) {
                db.rollback();
                deleteTrial(newTrialID);
                return -1;
            }

        } catch (SQLException e) {
            try {
                db.rollback();
                e.printStackTrace();
                throw new DatabaseException("Saving Trial Failed, rollbacks successful", e);
            } catch (SQLException e2) {
                throw new DatabaseException("Saving Trial Failed, rollbacks failed!", e2);
            }

        }

        try {
            db.commit();
            db.setAutoCommit(true);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: commit failed!", e);
        }

        //long stop = System.currentTimeMillis();
        //long elapsedMillis = stop - start;
        //double elapsedSeconds = (double) (elapsedMillis) / 1000.0;
        //        System.out.println("Elapsed time: " + elapsedSeconds + " seconds.");
        return newTrialID;
    }

    public int saveApplication() {
        int appid = 0;
        try {
            if (application != null) {
                appid = application.saveApplication(db);
            }
        } catch (SQLException e) {
            throw new DatabaseException("Error saving application", e);
        }
        return appid;
    }

    public int saveExperiment() {
        int expid = 0;
        try {
            if (experiment != null) {
                expid = experiment.saveExperiment(db);
            }
        } catch (SQLException e) {
            throw new DatabaseException("Error saving experiment", e);
        }
        return expid;
    }

    public void deleteTrial(int trialID) throws SQLException {
        Trial.deleteTrial(db, trialID);
    }

    public void deleteExperiment(int experimentID) throws DatabaseException, SQLException {
        // create a new DatabaseAPI to handle this request!
        // Why? Because we have to set the experiment to get the trials
        // and that will screw up the state of the current object.
        // the easiest way is to create a new reference to the DB.
        DatabaseAPI tmpSession = new DatabaseAPI();
        // don't initialize (not a new connection) - just reference
        // the other DB connection
        tmpSession.setDB(this.db());

        tmpSession.setExperiment(experimentID);
        ListIterator<Trial> trials = tmpSession.getTrialList(false).listIterator();
        while (trials.hasNext()) {
            Trial trial = trials.next();
            Trial.deleteTrial(db, trial.getID());
        }
        Experiment.deleteExperiment(db, experimentID);
    }

    public void deleteApplication(int applicationID) throws DatabaseException, SQLException {
        // create a new DatabaseAPI to handle this request!
        // Why? Because we have to set the experiment to get the trials
        // and that will screw up the state of the current object.
        // the easiest way is to create a new reference to the DB.
        DatabaseAPI tmpSession = new DatabaseAPI();
        // don't initialize (not a new connection) - just reference
        // the other DB connection
        tmpSession.setDB(this.db());

        tmpSession.setApplication(applicationID);
        ListIterator<Experiment> experiments = tmpSession.getExperimentList().listIterator();
        while (experiments.hasNext()) {
            Experiment experiment = experiments.next();
            tmpSession.setExperiment(experiment.getID());
            ListIterator<Trial> trials = tmpSession.getTrialList(false).listIterator();
            while (trials.hasNext()) {
                Trial trial = trials.next();
                Trial.deleteTrial(db, trial.getID());
            }
            Experiment.deleteExperiment(db, experiment.getID());
        }
        Application.deleteApplication(db, applicationID);
    }

    // This method has been added to let applications get the number of metrics
    // after the setApplication, setExperiment, setTrial have been called.
    // It does not affect the state of this object in any way.
    public int getNumberOfMetrics() {
        StringBuffer buf = new StringBuffer();
        buf.append("SELECT id, name ");
        buf.append("FROM " + db.getSchemaPrefix() + "metric ");
        buf.append("WHERE trial = ");
        buf.append(this.trial.getID());
        buf.append(" ORDER BY id ");

        // get the results
        try {
            ResultSet resultSet = db.executeQuery(buf.toString());
            int counter = 0;
            while (resultSet.next() != false) {
                counter++;
            }
            resultSet.close();
            return counter;
        } catch (Exception ex) {
            ex.printStackTrace();
            return -1;
        }
    }

    public void saveIntervalLocationProfiles(DB db, Hashtable<Integer, Integer> newFunHash, Enumeration<IntervalLocationProfile> en, Hashtable<Integer, Integer> newMetHash,
            int saveMetricIndex) throws SQLException {
        PreparedStatement statement = null;
        if (db.getDBType().compareTo("oracle") == 0) {
            statement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_location_profile (interval_event, node, context, thread, metric, inclusive_percentage, inclusive, exclusive_percentage, excl, call, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
        } else if (db.getDBType().compareTo("derby") == 0) {
            statement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_location_profile (interval_event, node, context, thread, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, num_calls, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
        } else if (db.getDBType().compareTo("mysql") == 0) {
            statement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_location_profile (interval_event, node, context, thread, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, `call`, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
        } else {
            statement = db.prepareStatement("INSERT INTO "
                    + db.getSchemaPrefix()
                    + "interval_location_profile (interval_event, node, context, thread, metric, inclusive_percentage, inclusive, exclusive_percentage, exclusive, call, subroutines, inclusive_per_call) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");
        }
        IntervalLocationProfile ilp;
        int i = 0;
        Integer newMetricID = null;
        while (en.hasMoreElements()) {
            ilp = en.nextElement();
            Integer newIntervalEventID = newFunHash.get(new Integer(ilp.getIntervalEventID()));
            // get the interval_event details
            i = 0;
            newMetricID = newMetHash.get(new Integer(i));
            while (newMetricID != null) {
                if (saveMetricIndex < 0 || i == saveMetricIndex) {
                    statement.setInt(1, newIntervalEventID.intValue());
                    statement.setInt(2, ilp.getNode());
                    statement.setInt(3, ilp.getContext());
                    statement.setInt(4, ilp.getThread());
                    statement.setInt(5, newMetricID.intValue());
                    statement.setDouble(6, ilp.getInclusivePercentage(i));
                    statement.setDouble(7, ilp.getInclusive(i));
                    statement.setDouble(8, ilp.getExclusivePercentage(i));
                    statement.setDouble(9, ilp.getExclusive(i));
                    statement.setDouble(10, ilp.getNumCalls());
                    statement.setDouble(11, ilp.getNumSubroutines());
                    statement.setDouble(12, ilp.getInclusivePerCall(i));
                    statement.executeUpdate();
                }
                newMetricID = newMetHash.get(new Integer(++i));
            }

            //DatabaseAPI.itemsDone++;
        }
        statement.close();
    }

    public Application getApplication(String name, boolean create) {
        List<Application> apps = getApplicationList();
        for (Iterator<Application> it = apps.iterator(); it.hasNext();) {
            Application app = it.next();
            if (app.getName().equals(name)) {
                return app;
            }
        }
        // didn't find one with that name
        if (create) {
            Application newApp = new Application();
            newApp.setDatabase(database);
            newApp.setName(name);
            setApplication(newApp);
            int appId = saveApplication();
            newApp.setID(appId);
            return newApp;
        }
        return null;
    }

    public Experiment getExperiment(Application app, String name, boolean create) {
        setApplication(app);
        List<Experiment> exps = getExperimentList();
        for (Iterator<Experiment> it = exps.iterator(); it.hasNext();) {
            Experiment exp = it.next();
            if (exp.getName().equals(name)) {
                return exp;
            }
        }

        if (create) {
            Experiment newExp = new Experiment();
            newExp.setName(name);
            newExp.setApplicationID(app.getID());
            newExp.setDatabase(database);
            setExperiment(newExp);
            int expId = saveExperiment();
            newExp.setID(expId);
            return newExp;
        }

        return null;
    }

    public Trial getTrial(Application app, Experiment exp, String name) {
        setApplication(app);
        List<Experiment> exps = getExperimentList();
        for (Iterator<Experiment> it = exps.iterator(); it.hasNext();) {
            Experiment tmp = it.next();
            if (tmp.getName().equals(name)) {
                setExperiment(tmp);
                List<Trial> trials = getTrialList(false);
                for (Iterator<Trial> it2 = trials.iterator(); it2.hasNext();) {
                    Trial trial = it2.next();
                    if (exp.getName().equals(name)) {
                        return trial;
                    }
                }
            }
        }

        return null;
    }

    public Experiment getExperiment(String appName, String expName, boolean create) {
        Application app = getApplication(appName, create);
        if (app == null) {
            return null;
        }
        return getExperiment(app, expName, create);
    }

    public DB getDb() {
        return db;
    }

    public Trial getTrial() {
        return trial;
    }
};
