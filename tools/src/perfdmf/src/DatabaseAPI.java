package edu.uoregon.tau.perfdmf;

import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;

import edu.uoregon.tau.perfdmf.database.ConnectionManager;
import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.database.ParseConfig;

/**
 * This is the top level class for the Database API.
 * 
 * <P>
 * CVS $Id: DatabaseAPI.java,v 1.14 2007/05/16 20:06:53 amorris Exp $
 * </P>
 * 
 * @author Kevin Huck, Robert Bell
 * @version $Revision: 1.14 $
 */
public class DatabaseAPI {

    private Application application = null;
    private Experiment experiment = null;
    private Trial trial = null;
    private Vector nodes = null;
    private Vector contexts = null;
    private Vector threads = null;
    private Vector intervalEvents = null;
    private List metrics = null;
    private Vector intervalEventData = null;
    private Vector atomicEvents = null;
    private Vector atomicEventData = null;

    // from datasession
    private DB db = null;
    private ConnectionManager connector;
    private Hashtable intervalEventHash = null;
    private Hashtable atomicEventHash = null;
    private String configFileName = null;

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
            return ((Metric) this.metrics.get(metricID)).getName();
        else
            return null;
    }

    public void setApplication(Application application) {
        this.application = application;
    }

    public void setExperiment(Experiment experiment) {
        this.experiment = experiment;
    }

    public DatabaseAPI() {
        super();
    }

    public DB db() {
        return db;
    }

    public void setDB(DB db) {
        this.db = db;
    }

    // Initialization / termination routines

    public void initialize(Object obj, boolean prompt) throws SQLException {
        configFileName = new String((String) (obj));
        //Initialize the connection to the database,
        //using the configuration settings.
        connector = new ConnectionManager(configFileName, prompt);
        connector.connect();
        db = connector.getDB();
        Application.getMetaData(db);
        Experiment.getMetaData(db);
        Trial.getMetaData(db);
    }

    public void initialize(ParseConfig parser, String password) throws SQLException {
        connector = new ConnectionManager(parser, password);
        connector.connect();
        db = connector.getDB();
        Application.getMetaData(db);
        Experiment.getMetaData(db);
        Trial.getMetaData(db);
    }

    public void initialize(Object obj, String password) throws SQLException {
        String configFileName = (String) (obj);
        //Initialize the connection to the database,
        //using the configuration settings.
        connector = new ConnectionManager(configFileName, password);
        connector.connect();
        db = connector.getDB();
        Application.getMetaData(db);
        Experiment.getMetaData(db);
        Trial.getMetaData(db);
    }

    public void initialize(Database database) throws SQLException {
        this.database = database;
        connector = new ConnectionManager(database.getConfig());
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
    public List getApplicationList() throws DatabaseException {
        String whereClause = "";
        return Application.getApplicationList(db, whereClause, database);
    }

    // returns Vector of Experiment objects
    public List getExperimentList() throws DatabaseException {

        String whereClause = "";
        if (application != null)
            whereClause = "WHERE application = " + application.getID();
        return Experiment.getExperimentList(db, whereClause);

    }

    // returns Vector of Trial objects
    public List getTrialList() {
        StringBuffer whereClause = new StringBuffer();
        if (experiment != null) {
            whereClause.append("WHERE t.experiment = " + experiment.getID());
        } else if (application != null) {
            whereClause.append("WHERE e.application = " + application.getID());
        }
        return Trial.getTrialList(db, whereClause.toString());
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
        Vector applications = Application.getApplicationList(db, whereClause, database);
        if (applications.size() == 1) {
            this.application = (Application) applications.elementAt(0);
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
        Vector applications = Application.getApplicationList(db, whereClause.toString(), database);
        if (applications.size() == 1) {
            this.application = (Application) applications.elementAt(0);
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
        Vector experiments = Experiment.getExperimentList(db, whereClause);
        if (experiments.size() == 1) {
            this.experiment = (Experiment) experiments.elementAt(0);
            return this.experiment;
        } else {
            return null;
        }
    }

    // set the Trial for this session
    public Trial setTrial(int id) {
        return setTrial(id, true);
    }

    private Trial setTrial(int id, boolean clearHashes) {
        this.trial = null;
        this.metrics = null;
        if (clearHashes) {
            this.intervalEventHash = null;
            this.atomicEventHash = null;
        }
        // create a string to hit the database
        String whereClause;
        whereClause = " WHERE t.id = " + id;
        Vector trials = Trial.getTrialList(db, whereClause);
        if (trials.size() == 1) {
            this.trial = (Trial) trials.elementAt(0);
        } //else exception?

        return this.trial;
    }

    // returns a List of IntervalEvents
    public List getIntervalEvents() {
        String whereClause = new String();
        if (trial != null) {
            whereClause = " WHERE trial = " + trial.getID();
        } else if (experiment != null) {
            whereClause = " WHERE experiment = " + experiment.getID();
        } else if (application != null) {
            whereClause = " WHERE application = " + application.getID();
        }

        intervalEvents = IntervalEvent.getIntervalEvents(this, db, whereClause);

        if (intervalEventHash == null)
            intervalEventHash = new Hashtable();
        IntervalEvent fun;
        for (Enumeration en = intervalEvents.elements(); en.hasMoreElements();) {
            fun = (IntervalEvent) en.nextElement();
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
            for (Iterator en = metrics.iterator(); en.hasNext();) {
                metric = (Metric) en.next();
                buf.append(metric.getID());
                if (en.hasNext())
                    buf.append(", ");
                else
                    buf.append(") ");
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
    public List getAtomicEvents() {
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
            atomicEventHash = new Hashtable();
        AtomicEvent ue;
        for (Enumeration en = atomicEvents.elements(); en.hasMoreElements();) {
            ue = (AtomicEvent) en.nextElement();
            atomicEventHash.put(new Integer(ue.getID()), ue);
        }
        return atomicEvents;
    }

    // sets the current intervalEvent
    public IntervalEvent setIntervalEvent(int id) {
        IntervalEvent intervalEvent = null;
        this.intervalEvents = new Vector();
        intervalEvent = getIntervalEvent(id);
        if (intervalEvent != null)
            this.intervalEvents.addElement(intervalEvent);
        // we need this to get the metric data...
        setTrial(intervalEvent.getTrialID(), false);
        return intervalEvent;
    }

    // sets the current user event
    public AtomicEvent setAtomicEvent(int id) {
        AtomicEvent atomicEvent = null;
        this.atomicEvents = new Vector();
        atomicEvent = getAtomicEvent(id);
        if (atomicEvent != null)
            this.atomicEvents.addElement(atomicEvent);
        return atomicEvent;
    }

    public List getIntervalEventData() throws SQLException {
        // check to make sure this is a meaningful request
        if (trial == null && intervalEvents == null) {
            System.out.println("Please select a trial or a set of intervalEvents before getting intervalEvent data.");
            return null;
        }

        // get the hash of intervalEvent names first
        if (intervalEvents == null) {
            getIntervalEvents();
        }

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
            for (Enumeration en = intervalEvents.elements(); en.hasMoreElements();) {
                intervalEvent = (IntervalEvent) en.nextElement();
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
            for (Enumeration en = nodes.elements(); en.hasMoreElements();) {
                node = (Integer) en.nextElement();
                buf.append(node);
                if (en.hasMoreElements())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (contexts != null && contexts.size() > 0) {
            buf.append(" AND p.context IN (");
            Integer context;
            for (Enumeration en = contexts.elements(); en.hasMoreElements();) {
                context = (Integer) en.nextElement();
                buf.append(context);
                if (en.hasMoreElements())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (threads != null && threads.size() > 0) {
            buf.append(" AND p.thread IN (");
            Integer thread;
            for (Enumeration en = threads.elements(); en.hasMoreElements();) {
                thread = (Integer) en.nextElement();
                buf.append(thread);
                if (en.hasMoreElements())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (metrics != null && metrics.size() > 0) {
            buf.append(" AND p.metric IN (");
            Metric metric;
            for (Iterator en = metrics.iterator(); en.hasNext();) {
                metric = (Metric) en.next();
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

    public List getAtomicEventData() {
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
            for (Enumeration en = atomicEvents.elements(); en.hasMoreElements();) {
                atomicEvent = (AtomicEvent) en.nextElement();
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
            for (Enumeration en = nodes.elements(); en.hasMoreElements();) {
                node = (Integer) en.nextElement();
                buf.append(node);
                if (en.hasMoreElements())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (contexts != null && contexts.size() > 0) {
            buf.append(" AND p.context IN (");
            Integer context;
            for (Enumeration en = contexts.elements(); en.hasMoreElements();) {
                context = (Integer) en.nextElement();
                buf.append(context);
                if (en.hasMoreElements())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        if (threads != null && threads.size() > 0) {
            buf.append(" AND p.thread IN (");
            Integer thread;
            for (Enumeration en = threads.elements(); en.hasMoreElements();) {
                thread = (Integer) en.nextElement();
                buf.append(thread);
                if (en.hasMoreElements())
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
            intervalEvent = (IntervalEvent) intervalEventHash.get(new Integer(id));
        }
        if (intervalEvent == null) {
            // create a string to hit the database
            String whereClause;
            whereClause = " WHERE id = " + id;
            Vector intervalEvents = IntervalEvent.getIntervalEvents(this, db, whereClause);
            if (intervalEvents.size() == 1) {
                intervalEvent = (IntervalEvent) intervalEvents.elementAt(0);
            } //else exception?
            if (intervalEventHash == null)
                intervalEventHash = new Hashtable();
            intervalEventHash.put(new Integer(intervalEvent.getID()), intervalEvent);
        }
        return intervalEvent;
    }

    public AtomicEvent getAtomicEvent(int id) {
        AtomicEvent atomicEvent = null;
        if (atomicEventHash != null) {
            atomicEvent = (AtomicEvent) atomicEventHash.get(new Integer(id));
        }
        if (atomicEvent == null) {
            // create a string to hit the database
            String whereClause;
            whereClause = " WHERE u.id = " + id;
            Vector atomicEvents = AtomicEvent.getAtomicEvents(this, db, whereClause);
            if (atomicEvents.size() == 1) {
                atomicEvent = (AtomicEvent) atomicEvents.elementAt(0);
            } //else exception?
            if (atomicEventHash == null)
                atomicEventHash = new Hashtable();
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

    private Hashtable saveMetrics(int newTrialID, Trial trial, int saveMetricIndex) throws SQLException {
        Hashtable metricHash = new Hashtable();
        Iterator en = trial.getDataSource().getMetrics().iterator();
        int i = 0;
        while (en.hasNext()) {
            Metric metric = (Metric) en.next();
            int newMetricID = 0;
            if (saveMetricIndex < 0 || saveMetricIndex == i) {
                newMetricID = saveMetric(newTrialID, metric);
            }
            metricHash.put(new Integer(i), new Integer(newMetricID));
            i++;
        }
        return metricHash;
    }

    private Hashtable saveIntervalEvents(int newTrialID, Hashtable newMetHash, int saveMetricIndex) throws SQLException {
        //      System.out.print("Saving the intervalEvents: ");
        Hashtable newFunHash = new Hashtable();
        Enumeration en = intervalEvents.elements();
        IntervalEvent intervalEvent;
        int count = 0;
        while (en.hasMoreElements()) {
            intervalEvent = (IntervalEvent) en.nextElement();
            int newIntervalEventID = intervalEvent.saveIntervalEvent(db, newTrialID, newMetHash, saveMetricIndex);
            newFunHash.put(new Integer(intervalEvent.getID()), new Integer(newIntervalEventID));
            //System.out.print("\rSaving the intervalEvents: " + ++count + " records saved...");
            //DatabaseAPI.itemsDone++;

        }
        //     System.out.print("\n");
        return newFunHash;
    }

    private Hashtable saveAtomicEvents(int newTrialID) {
        //        System.out.print("Saving the user events:");
        Hashtable newUEHash = new Hashtable();
        Enumeration en = atomicEvents.elements();
        int count = 0;
        AtomicEvent atomicEvent;
        while (en.hasMoreElements()) {
            atomicEvent = (AtomicEvent) en.nextElement();
            int newAtomicEventID = atomicEvent.saveAtomicEvent(db, newTrialID);
            newUEHash.put(new Integer(atomicEvent.getID()), new Integer(newAtomicEventID));
            //System.out.print("\rSaving the user events: " + ++count + " records saved...");
            //DatabaseAPI.itemsDone++;

        }
        //    System.out.print("\n");
        return newUEHash;
    }

    private void saveAtomicEventData(Hashtable newUEHash) {
        //    System.out.print("Saving the user event data:");
        Enumeration en = atomicEventData.elements();
        AtomicLocationProfile uedo;
        int count = 0;
        while (en.hasMoreElements()) {
            uedo = (AtomicLocationProfile) en.nextElement();
            Integer newAtomicEventID = (Integer) newUEHash.get(new Integer(uedo.getAtomicEventID()));
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
    public int saveIntervalEvent(IntervalEvent intervalEvent, int newTrialID, Hashtable newMetHash) throws SQLException {
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

    public synchronized int saveTrial(Trial trial, int saveMetricIndex) throws DatabaseException {
        long start = System.currentTimeMillis();

        DataSource dataSource = trial.getDataSource();

        //Build an array of group names. This speeds lookup of group names.
        List groupList = new ArrayList();
        for (Iterator groupIterator = dataSource.getGroups(); groupIterator.hasNext();) {
            Group g = (Group) groupIterator.next();
            groupList.add(g.getName());
        }

        //String groupNames[] = new String[groupList.size()];
        //groupNames = groupList.toArray(groupNames);

        //String groupNames[] = (String[]) groupList.toArray();

        // get the metric count
        metrics = trial.getDataSource().getMetrics();
        int metricCount = metrics.size();

        // create the Vectors to store the data
        intervalEvents = new Vector();
        intervalEventData = new Vector();
        atomicEvents = new Vector();
        atomicEventData = new Vector();

        int fcount = 0;
        int ucount = 0;

        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");

        // create the intervalEvents
        for (Iterator it = dataSource.getFunctions(); it.hasNext();) {
            Function f = (Function) it.next();
            if (!f.isGroupMember(derived)) {
                // create a intervalEvent
                IntervalEvent intervalEvent = new IntervalEvent(this);
                intervalEvent.setName(f.getName());
                intervalEvent.setID(f.getID());
                // intervalEvent.setTrialID(newTrialID);
                // build the group name
                List groups = f.getGroups();
                StringBuffer buf = new StringBuffer();
                if (groups != null) {
                    for (int i = 0; i < groups.size(); i++) {
                        if (i > 0)
                            buf.append("|");
                        buf.append(((Group) groups.get(i)).getName());
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
        for (Iterator it = dataSource.getUserEvents(); it.hasNext();) {
            UserEvent ue = (UserEvent) it.next();
            if (ue != null) {
                AtomicEvent atomicEvent = new AtomicEvent(this);
                atomicEvent.setName(ue.getName());
                atomicEvent.setID(ue.getID());
                atomicEvents.add(atomicEvent);
            }
        }

        for (Iterator it = trial.getDataSource().getAllThreads().iterator(); it.hasNext();) {
            edu.uoregon.tau.perfdmf.Thread thread = (edu.uoregon.tau.perfdmf.Thread) it.next();
            List intervalEvents = thread.getFunctionProfiles();
            List userevents = thread.getUserEventProfiles();

            // create interval location profiles
            for (Iterator e4 = intervalEvents.iterator(); e4.hasNext();) {
                FunctionProfile fp = (FunctionProfile) e4.next();
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
            if (userevents != null) {
                for (Iterator e4 = userevents.iterator(); e4.hasNext();) {
                    UserEventProfile uep = (UserEventProfile) e4.next();
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
        }

        totalItems = intervalEvents.size() + intervalEventData.size() + atomicEvents.size() + atomicEventData.size();

        // Now upload to the database

        try {
            db.setAutoCommit(false);
        } catch (SQLException e) {
            throw new DatabaseException("Saving Trial Failed: couldn't set AutoCommit to false", e);
        }

        int newTrialID = 0;

        try {
            // output the trial data, which also saves the intervalEvents,
            // intervalEvent data, user events and user event data
            if (saveMetricIndex < 0) { // this means save the whole thing???
                newTrialID = trial.saveTrial(db);
                trial.setID(newTrialID);
                Hashtable metricHash = saveMetrics(newTrialID, trial, saveMetricIndex);

                if (intervalEvents != null && intervalEvents.size() > 0) {
                    Hashtable functionHash = saveIntervalEvents(newTrialID, metricHash, saveMetricIndex);

                    saveIntervalLocationProfiles(db, functionHash, intervalEventData.elements(), metricHash, saveMetricIndex);
                }
                if (atomicEvents != null && atomicEvents.size() > 0) {
                    Hashtable atomicEventHash = saveAtomicEvents(newTrialID);
                    if (atomicEventData != null && atomicEventData.size() > 0) {
                        saveAtomicEventData(atomicEventHash);
                    }
                }

                //  System.out.println("New Trial ID: " + newTrialID);
            } else {
                newTrialID = trial.getID();
                //   System.out.println("\nSaving the metric...");
                Hashtable newMetHash = saveMetrics(newTrialID, trial, saveMetricIndex);

                if (intervalEvents != null && intervalEvents.size() > 0) {
                    Hashtable newFunHash = saveIntervalEvents(newTrialID, newMetHash, saveMetricIndex);
                    saveIntervalLocationProfiles(db, newFunHash, intervalEventData.elements(), newMetHash, saveMetricIndex);

                }

                //   System.out.println("Modified Trial ID: " + newTrialID);
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

        long stop = System.currentTimeMillis();
        long elapsedMillis = stop - start;
        double elapsedSeconds = (double) (elapsedMillis) / 1000.0;
        //System.out.println("Elapsed time: " + elapsedSeconds + " seconds.");
        return newTrialID;
    }

    private Map uploadMetrics(int trialID, DataSource dataSource) throws SQLException {
        Map map = new HashMap();

        for (Iterator it = dataSource.getMetrics().iterator(); it.hasNext();) {
            Metric metric = (Metric) it.next();

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
    private Map uploadFunctions(int trialID, DataSource dataSource) throws SQLException {
        Map map = new HashMap();

        Group derived = dataSource.getGroup("TAU_CALLPATH_DERIVED");
        for (Iterator it = dataSource.getFunctions(); it.hasNext();) {
            Function f = (Function) it.next();
            if (f.isGroupMember(derived)) {
                continue;
            }

            String group = null;
            List groups = f.getGroups();
            StringBuffer allGroups = new StringBuffer();
            if (groups != null) {
                for (int i = 0; i < groups.size(); i++) {
                    if (i > 0)
                        allGroups.append("|");
                    allGroups.append(((Group) groups.get(i)).getName());
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
    private Map uploadUserEvents(int trialID, DataSource dataSource) throws SQLException {
        Map map = new HashMap();

        String group = null; // no groups right now?

        for (Iterator it = dataSource.getUserEvents(); it.hasNext();) {
            UserEvent ue = (UserEvent) it.next();

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

    private void uploadFunctionProfiles(int trialID, DataSource dataSource, Map functionMap, Map metricMap) throws SQLException {

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

        for (Iterator it5 = dataSource.getMetrics().iterator(); it5.hasNext();) {
            Metric metric = (Metric) it5.next();
            Integer dbMetricID = (Integer) metricMap.get(metric);

            for (Iterator it4 = dataSource.getFunctions(); it4.hasNext();) {
                Function function = (Function) it4.next();
                if (function.isGroupMember(derived)) {
                    continue;
                }
                Integer intervalEventID = (Integer) functionMap.get(function);

                edu.uoregon.tau.perfdmf.Thread totalData = dataSource.getTotalData();
                addBatchFunctionProfile(totalInsertStatement, totalData, metric.getID(), dbMetricID.intValue(),
                        function.getTotalProfile(), intervalEventID.intValue(), false, dataSource.getAllThreads().size());

                //edu.uoregon.tau.dms.dss.Thread meanData = dataSource.getMeanData();
                //addBatchFunctionProfile(meanInsertStatement, meanData, metric.getID(), dbMetricID.intValue(),
                //       function.getMeanProfile(), intervalEventID.intValue(), false, dataSource.getAllThreads().size());

                addBatchFunctionProfile(meanInsertStatement, totalData, metric.getID(), dbMetricID.intValue(),
                        function.getTotalProfile(), intervalEventID.intValue(), true, dataSource.getAllThreads().size());

                for (Iterator it = dataSource.getAllThreads().iterator(); it.hasNext();) {
                    edu.uoregon.tau.perfdmf.Thread thread = (edu.uoregon.tau.perfdmf.Thread) it.next();

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

    private void uploadUserEventProfiles(int trialID, DataSource dataSource, Map userEventMap) throws SQLException {

        for (Iterator it = dataSource.getNodes(); it.hasNext();) {
            Node node = (Node) it.next();
            for (Iterator it2 = node.getContexts(); it2.hasNext();) {
                Context context = (Context) it2.next();
                for (Iterator it3 = context.getThreads(); it3.hasNext();) {
                    edu.uoregon.tau.perfdmf.Thread thread = (edu.uoregon.tau.perfdmf.Thread) it3.next();

                    for (Iterator it4 = thread.getUserEventProfiles().iterator(); it4.hasNext();) {
                        UserEventProfile uep = (UserEventProfile) it4.next();

                        if (this.cancelUpload)
                            return;

                        if (uep != null) {
                            int atomicEventID = ((Integer) userEventMap.get(uep.getUserEvent())).intValue();

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

        for (Iterator it4 = dataSource.getFunctions(); it4.hasNext();) {
            Function function = (Function) it4.next();
            this.totalItems++;
        }

        int numMetrics = dataSource.getMetrics().size();

        for (Iterator it4 = dataSource.getFunctions(); it4.hasNext();) {
            Function function = (Function) it4.next();

            this.totalItems += numMetrics; // total
            this.totalItems += numMetrics; // mean

            for (Iterator it = dataSource.getAllThreads().iterator(); it.hasNext();) {
                edu.uoregon.tau.perfdmf.Thread thread = (edu.uoregon.tau.perfdmf.Thread) it.next();
                FunctionProfile fp = thread.getFunctionProfile(function);
                if (fp != null) { // only if this thread calls this function
                    this.totalItems += numMetrics; // this profile
                }
            }
        }
    }

    public synchronized int uploadTrial(Trial trial) throws DatabaseException {
        long start = System.currentTimeMillis();

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
            // upload the metrics and get a map that maps the metrics 0 -> n-1 to their unique DB IDs
            Map metricMap = uploadMetrics(newTrialID, dataSource);
            Map functionMap = uploadFunctions(newTrialID, dataSource);

            uploadFunctionProfiles(newTrialID, dataSource, functionMap, metricMap);

            Map userEventMap = uploadUserEvents(newTrialID, dataSource);

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

        long stop = System.currentTimeMillis();
        long elapsedMillis = stop - start;
        double elapsedSeconds = (double) (elapsedMillis) / 1000.0;
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
        ListIterator trials = tmpSession.getTrialList().listIterator();
        while (trials.hasNext()) {
            Trial trial = (Trial) trials.next();
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
        ListIterator experiments = tmpSession.getExperimentList().listIterator();
        while (experiments.hasNext()) {
            Experiment experiment = (Experiment) experiments.next();
            tmpSession.setExperiment(experiment.getID());
            ListIterator trials = tmpSession.getTrialList().listIterator();
            while (trials.hasNext()) {
                Trial trial = (Trial) trials.next();
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

    public void saveIntervalLocationProfiles(DB db, Hashtable newFunHash, Enumeration en, Hashtable newMetHash,
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
        } else if (db.getDBType().compareTo("derby") == 0) {
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
            ilp = (IntervalLocationProfile) en.nextElement();
            Integer newIntervalEventID = (Integer) newFunHash.get(new Integer(ilp.getIntervalEventID()));
            // get the interval_event details
            i = 0;
            newMetricID = (Integer) newMetHash.get(new Integer(i));
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
                newMetricID = (Integer) newMetHash.get(new Integer(++i));
            }

            //DatabaseAPI.itemsDone++;
        }
        statement.close();
    }

    public Application getApplication(String name, boolean create) {
        List apps = getApplicationList();
        for (Iterator it = apps.iterator(); it.hasNext();) {
            Application app = (Application) it.next();
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
        List exps = getExperimentList();
        for (Iterator it = exps.iterator(); it.hasNext();) {
            Experiment exp = (Experiment) it.next();
            if (exp.getName().equals(name)) {
                return exp;
            }
        }

        if (create) {
            Experiment newExp = new Experiment();
            newExp.setName(name);
            newExp.setApplicationID(app.getID());
            setExperiment(newExp);
            int expId = saveExperiment();
            newExp.setID(expId);
            return newExp;
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
