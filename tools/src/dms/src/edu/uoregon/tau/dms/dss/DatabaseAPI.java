package edu.uoregon.tau.dms.dss;

import edu.uoregon.tau.dms.database.*;
import java.util.*;
import java.sql.*;

/**
 * This is the top level class for the Database implementation of the API.
 * 
 * <P>
 * CVS $Id: DatabaseAPI.java,v 1.9 2004/12/23 00:25:51 amorris Exp $
 * </P>
 * 
 * @author Kevin Huck, Robert Bell
 * @version 0.1
 */
public class DatabaseAPI {

    // from datasession
    protected Application application = null;
    protected Experiment experiment = null;
    protected Trial trial = null;
    protected Vector nodes = null;
    protected Vector contexts = null;
    protected Vector threads = null;
    protected Vector intervalEvents = null;
    protected Vector metrics = null;
    protected Vector intervalEventData = null;
    protected Vector atomicEvents = null;
    protected Vector atomicEventData = null;

    public String getMetricName(int metricID) {
        if (this.metrics == null) {
            if (this.trial != null) {
                this.metrics = this.trial.getMetrics();
            }
        }

        //Try getting the metric name.
        if ((this.metrics != null) && (metricID < this.metrics.size()))
            return ((Metric) this.metrics.elementAt(metricID)).getName();
        else
            return null;
    }

    public void setApplication(Application application) {
        this.application = application;
    }

    public void setExperiment(Experiment experiment) {
        this.experiment = experiment;
    }

    // from datasession

    private DB db = null;
    private ConnectionManager connector;
    private Hashtable intervalEventHash = null;
    private Hashtable atomicEventHash = null;
    private String configFileName = null;

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

    public void initialize(Object obj) {
        try {
            initialize(obj, true, false);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(0);
        }
    }

    public void initialize(Object obj, boolean prompt, boolean catchException) throws Exception {
        configFileName = new String((String) (obj));
        //Initialize the connection to the database,
        //using the configuration settings.
        try {
            connector = new ConnectionManager(configFileName, prompt);
            connector.connect();
            db = connector.getDB();
        } catch (Exception e) {
            if (catchException)
                System.exit(0);
            else
                throw e;
        }
    }

    public void initialize(Object obj, String password, boolean catchException) throws Exception {
        String configFileName = (String) (obj);
        //Initialize the connection to the database,
        //using the configuration settings.
        try {
            connector = new ConnectionManager(configFileName, password);
            connector.connect();
            db = connector.getDB();
        } catch (Exception e) {
            if (catchException)
                System.exit(0);
            else
                throw e;
        }
    }

    public void terminate() {
        connector.dbclose();
    }

    public ConnectionManager getConnector() {
        return connector;
    }

    // returns Vector of ALL Application objects
    public ListIterator getApplicationList() {
        String whereClause = "";
        return new DssIterator(Application.getApplicationList(db, whereClause));
    }

    // returns Vector of Experiment objects
    public ListIterator getExperimentList() {

        String whereClause = "";
        if (application != null)
            whereClause = "WHERE application = " + application.getID();
        return new DssIterator(Experiment.getExperimentList(db, whereClause));
    }

    // returns Vector of Trial objects
    public ListIterator getTrialList() {
        StringBuffer whereClause = new StringBuffer();
        if (experiment != null) {
            whereClause.append("WHERE t.experiment = " + experiment.getID());
        } else if (application != null) {
            whereClause.append("WHERE e.application = " + application.getID());
        }
        return new DssIterator(Trial.getTrialList(db, whereClause.toString()));
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
        Vector applications = Application.getApplicationList(db, whereClause);
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
        Vector applications = Application.getApplicationList(db, whereClause.toString());
        if (applications.size() == 1) {
            this.application = (Application) applications.elementAt(0);
        } // else exception?
        return this.application;
    }

    // set the Experiment for this session
    public Experiment setExperiment(int id) {
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
        } //else exception?

        return this.experiment;
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

    // returns a ListIterator of IntervalEvents
    public ListIterator getIntervalEvents() {
        String whereClause = new String();
        if (trial != null) {
            whereClause = " WHERE trial = " + trial.getID();
        } else if (experiment != null) {
            whereClause = " WHERE experiment = " + experiment.getID();
        } else if (application != null) {
            whereClause = " WHERE application = " + application.getID();
        }

        intervalEvents = IntervalEvent.getIntervalEvents(this, db, whereClause);
        
        DatabaseAPI.totalItems *= intervalEvents.size();
        
        if (intervalEventHash == null)
            intervalEventHash = new Hashtable();
        IntervalEvent fun;
        for (Enumeration en = intervalEvents.elements(); en.hasMoreElements();) {
            fun = (IntervalEvent) en.nextElement();
            intervalEventHash.put(new Integer(fun.getID()), fun);
        }
        return new DssIterator(intervalEvents);
    }

    // gets the mean & total data for a intervalEvent
    public void getIntervalEventDetail(IntervalEvent intervalEvent) {
        StringBuffer buf = new StringBuffer();
        buf.append(" WHERE ms.interval_event = " + intervalEvent.getID());
        if (metrics != null && metrics.size() > 0) {
            buf.append(" AND ms.metric in (");
            Metric metric;
            for (Enumeration en = metrics.elements(); en.hasMoreElements();) {
                metric = (Metric) en.nextElement();
                buf.append(metric.getID());
                if (en.hasMoreElements())
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

    // returns a ListIterator of AtomicEvents
    public ListIterator getAtomicEvents() {
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
        return new DssIterator(atomicEvents);
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

    public ListIterator getIntervalEventData() {
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
            for (Enumeration en = metrics.elements(); en.hasMoreElements();) {
                metric = (Metric) en.nextElement();
                buf.append(metric.getID());
                if (en.hasMoreElements())
                    buf.append(", ");
                else
                    buf.append(") ");
            }
        }
        intervalEventData = IntervalLocationProfile.getIntervalEventData(db, metricCount,
                buf.toString());
        return new DssIterator(intervalEventData);
    }

    public ListIterator getAtomicEventData() {
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
        return new DssIterator(atomicEventData);
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
        return app.saveApplication(db);
    }

    public int saveExperiment(Experiment exp) {
        return exp.saveExperiment(db);
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

    // save the metrics
    private Hashtable saveMetrics(int newTrialID, Trial trial, int saveMetricIndex) {
       // System.out.print("Saving the metrics: ");
        Hashtable newMetHash = new Hashtable();
        Enumeration en = trial.getDataSource().getMetrics().elements();
        Metric metric;
        int i = 0;
        while (en.hasMoreElements()) {
            metric = (Metric) en.nextElement();
            int newMetricID = 0;
            if (saveMetricIndex < 0 || saveMetricIndex == i) {
                newMetricID = metric.saveMetric(db, newTrialID);
            //    System.out.print("\rSaving the metrics: " + (i + 1) + " records saved...");
            }
            newMetHash.put(new Integer(i), new Integer(newMetricID));
            i++;
        }
      //  System.out.print("\n");
        return newMetHash;
    }

    // save the intervalEvents
    private Hashtable saveIntervalEvents(int newTrialID, Hashtable newMetHash, int saveMetricIndex) {
  //      System.out.print("Saving the intervalEvents: ");
        Hashtable newFunHash = new Hashtable();
        Enumeration en = intervalEvents.elements();
        IntervalEvent intervalEvent;
        int count = 0;
        while (en.hasMoreElements()) {
            intervalEvent = (IntervalEvent) en.nextElement();
            int newIntervalEventID = intervalEvent.saveIntervalEvent(db, newTrialID, newMetHash,
                    saveMetricIndex);
            newFunHash.put(new Integer(intervalEvent.getID()), new Integer(newIntervalEventID));
            //System.out.print("\rSaving the intervalEvents: " + ++count + " records saved...");
            DatabaseAPI.itemsDone++;

        }
   //     System.out.print("\n");
        return newFunHash;
    }

    // save the intervalEvents
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
            DatabaseAPI.itemsDone++;

        }
    //    System.out.print("\n");
        return newUEHash;
    }

    // save the intervalEvent data
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
    public int saveIntervalEvent(IntervalEvent intervalEvent, int newTrialID, Hashtable newMetHash) {
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
    static int totalItems;
    static int itemsDone;

    public static int getProgress() {
        System.out.println ("itemsDone = (" + itemsDone + "/" + totalItems + ")");
        
        if (totalItems != 0)
            return (int) ((float) itemsDone / (float) totalItems * 100);
        return 0;
    }
    
    public static void setTotalItems(int items) {
        totalItems = items;
    }
    

    /**
     * Saves the ParaProfTrial object to the database
     * 
     * @param trial
     * @param saveMetricIndex
     * @return the database index ID of the saved trial record
     */

    public synchronized int saveParaProfTrial(Trial trial, int saveMetricIndex) {
        long start = System.currentTimeMillis();
        TrialData mapping = trial.getDataSource().getTrialData();

        //Build an array of group names. This speeds lookup of group names.
        Iterator groupIterator = mapping.getGroups();

        List groupList = new ArrayList();

        for (; groupIterator.hasNext();) {
            Group g = (Group) groupIterator.next();
            groupList.add(g.getName());
        }

        //String groupNames[] = new String[groupList.size()];
        //groupNames = groupList.toArray(groupNames);

        //String groupNames[] = (String[]) groupList.toArray();

        // get the metric count
        metrics = trial.getDataSource().getMetrics();
        int metricCount = metrics.size();
      //  System.out.println("Found " + metricCount + " metrics...");

        // create the Vectors to store the data
        intervalEvents = new Vector();
        intervalEventData = new Vector();
        atomicEvents = new Vector();
        atomicEventData = new Vector();

        int fcount = 0;
        int ucount = 0;
        // create the intervalEvents
    //    System.out.print("Creating the intervalEvents:");

        for (Iterator it = mapping.getFunctions(); it.hasNext();) {
            Function f = (Function) it.next();
            if (f != null) {
                // create a intervalEvent
                IntervalEvent intervalEvent = new IntervalEvent(this);
                intervalEvent.setName(f.getName());
                intervalEvent.setID(f.getID());
                // intervalEvent.setTrialID(newTrialID);
                // build the group name
                Vector groups = f.getGroups();
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

                // get the total data
           //     System.out.print("\rCreating the intervalEvents: " + ++fcount
          //              + " intervalEvents found...");
                IntervalLocationProfile funTS = new IntervalLocationProfile(metricCount);
                IntervalLocationProfile funMS = new IntervalLocationProfile(metricCount);
                for (int i = 0; i < metricCount; i++) {
                    funTS.setNumCalls(f.getTotalNumCalls());
                    funTS.setNumSubroutines(f.getTotalNumSubr());
                    funTS.setInclusivePercentage(i, f.getTotalInclusivePercent(i));
                    funTS.setInclusive(i, f.getTotalInclusive(i));
                    funTS.setExclusivePercentage(i, f.getTotalExclusivePercent(i));
                    funTS.setExclusive(i, f.getTotalExclusive(i));
                    funTS.setInclusivePerCall(i, f.getTotalInclusivePerCall(i));
                    funMS.setNumCalls(f.getMeanNumCalls());
                    funMS.setNumSubroutines(f.getMeanNumSubr());
                    funMS.setInclusivePercentage(i, f.getMeanInclusivePercent(i));
                    funMS.setInclusive(i, f.getMeanInclusive(i));
                    funMS.setExclusivePercentage(i, f.getMeanExclusivePercent(i));
                    funMS.setExclusive(i, f.getMeanExclusive(i));
                    funMS.setInclusivePerCall(i, f.getMeanInclusivePerCall(i));
                }
                intervalEvent.setTotalSummary(funTS);
                intervalEvent.setMeanSummary(funMS);
            }
        }

        // create the user events
    //    System.out.print("\nCreating user events:");
        for (Iterator it = mapping.getUserEvents(); it.hasNext();) {
            UserEvent ue = (UserEvent) it.next();
            if (ue != null) {
    //            System.out.print(".");
      //          System.out.print("\rCreating the user events: " + ++ucount
    //                    + " user events found...");
                // create a user event
                AtomicEvent atomicEvent = new AtomicEvent(this);
                atomicEvent.setName(ue.getName());
                atomicEvent.setID(ue.getID());

                //                // build the group name
                //                int[] groupIDs = element.getGroups();
                //                StringBuffer buf = new StringBuffer();
                //                for (int i = 0; i < element.getNumberOfGroups(); i++) {
                //                    if (i > 0)
                //                        buf.append("|");
                //                    buf.append(groupNames[groupIDs[i]]);
                //                }
                //                atomicEvent.setGroup(buf.toString());
                // put the atomicEvent in the vector
                atomicEvents.add(atomicEvent);
            }
        }

       
    //    System.out.print("\nCreating the intervalEvent / user event data:");
        StringBuffer groupsStringBuffer = new StringBuffer(10);
        Vector nodes = trial.getDataSource().getNCT().getNodes();
        for (Enumeration e1 = nodes.elements(); e1.hasMoreElements();) {
            Node node = (Node) e1.nextElement();
            Vector contexts = node.getContexts();
            for (Enumeration e2 = contexts.elements(); e2.hasMoreElements();) {
                Context context = (Context) e2.nextElement();
                Vector threads = context.getThreads();
                for (Enumeration e3 = threads.elements(); e3.hasMoreElements();) {
                    edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
                    Vector intervalEvents = thread.getFunctionList();
                    Vector userevents = thread.getUsereventList();
                    //Write out intervalEvent data for this thread.
                    for (Enumeration e4 = intervalEvents.elements(); e4.hasMoreElements();) {
                        FunctionProfile fp = (FunctionProfile) e4.nextElement();
                        if (fp != null) {

                      

                            IntervalLocationProfile fdo = new IntervalLocationProfile(metricCount);
                            fdo.setNode(thread.getNodeID());
                            fdo.setContext(thread.getContextID());
                            fdo.setThread(thread.getThreadID());
                            fdo.setIntervalEventID(fp.getFunction().getID());
                            fdo.setNumCalls(fp.getNumCalls());
                            fdo.setNumSubroutines(fp.getNumSubr());
                            // fdo.setInclusivePerCall(fp.getUserSecPerCall());
                            for (int i = 0; i < metricCount; i++) {
                                fdo.setInclusive(i, fp.getInclusive(i));
                                fdo.setExclusive(i, fp.getExclusive(i));
                                fdo.setInclusivePercentage(i, fp.getInclusivePercent(i));
                                fdo.setExclusivePercentage(i, fp.getExclusivePercent(i));
                                fdo.setInclusivePerCall(i, fp.getInclusivePerCall(i));
                            }
                            intervalEventData.add(fdo);
                        }
                    }

                    //Write out user event data for this thread.
                    if (userevents != null) {
                        for (Enumeration e4 = userevents.elements(); e4.hasMoreElements();) {
                            UserEventProfile uep = (UserEventProfile) e4.nextElement();
                            if (uep != null) {

                              
                                AtomicLocationProfile udo = new AtomicLocationProfile();
                                udo.setAtomicEventID(uep.getUserEvent().getID());
                                udo.setNode(thread.getNodeID());
                                udo.setContext(thread.getContextID());
                                udo.setThread(thread.getThreadID());
                                udo.setSampleCount(uep.getUserEventNumberValue());
                                udo.setMaximumValue(uep.getUserEventMaxValue());
                                udo.setMinimumValue(uep.getUserEventMinValue());
                                udo.setMeanValue(uep.getUserEventMeanValue());
                                udo.setSumSquared(uep.getUserEventSumSquared());
                                atomicEventData.add(udo);
                            }
                        }
                    }
                }
            }
        }

        totalItems = intervalEvents.size() + intervalEventData.size() + atomicEvents.size()
                + atomicEventData.size();

        try {
            db.setAutoCommit(false);
        } catch (SQLException e) {
            e.printStackTrace();
            return -1;
        }

        int newTrialID = 0;
        // output the trial data, which also saves the intervalEvents,
        // intervalEvent data, user events and user event data
        if (saveMetricIndex < 0) {
       //     System.out.println("\nSaving the trial...");
            newTrialID = trial.saveTrial(db);
            Hashtable newMetHash = saveMetrics(newTrialID, trial, saveMetricIndex);

            if (intervalEvents != null && intervalEvents.size() > 0) {
                Hashtable newFunHash = saveIntervalEvents(newTrialID, newMetHash, saveMetricIndex);

                IntervalLocationProfile.saveIntervalEventData(db, newFunHash,
                        intervalEventData.elements(), newMetHash, saveMetricIndex);
            }
            if (atomicEvents != null && atomicEvents.size() > 0) {
                Hashtable newUEHash = saveAtomicEvents(newTrialID);
                if (atomicEventData != null && atomicEventData.size() > 0) {
                    saveAtomicEventData(newUEHash);
                }
            }

          //  System.out.println("New Trial ID: " + newTrialID);
        } else {
            newTrialID = trial.getID();
         //   System.out.println("\nSaving the metric...");
            Hashtable newMetHash = saveMetrics(newTrialID, trial, saveMetricIndex);
            if (intervalEvents != null && intervalEvents.size() > 0) {
                Hashtable newFunHash = saveIntervalEvents(newTrialID, newMetHash, saveMetricIndex);
                IntervalLocationProfile.saveIntervalEventData(db, newFunHash,
                        intervalEventData.elements(), newMetHash, saveMetricIndex);

            }

         //   System.out.println("Modified Trial ID: " + newTrialID);
        }

        try {
            db.commit();
            db.setAutoCommit(true);
        } catch (SQLException e) {
            e.printStackTrace();
            return -1;
        }

        long stop = System.currentTimeMillis();
        long elapsedMillis = stop - start;
        double elapsedSeconds = (double) (elapsedMillis) / 1000.0;
        //System.out.println("Elapsed time: " + elapsedSeconds + " seconds.");
        return newTrialID;
    }

   

    public int saveApplication() {
        int appid = 0;
        if (application != null) {
            appid = application.saveApplication(db);
        }
        return appid;
    }

    public int saveExperiment() {
        int expid = 0;
        if (experiment != null) {
            expid = experiment.saveExperiment(db);
        }
        return expid;
    }

    public void deleteTrial(int trialID) {
        Trial.deleteTrial(db, trialID);
    }

    public void deleteExperiment(int experimentID) {
        // create a new DatabaseAPI to handle this request!
        // Why? Because we have to set the experiment to get the trials
        // and that will screw up the state of the current object.
        // the easiest way is to create a new reference to the DB.
        DatabaseAPI tmpSession = new DatabaseAPI();
        // don't initialize (not a new connection) - just reference
        // the other DB connection
        tmpSession.setDB(this.db());

        tmpSession.setExperiment(experimentID);
        ListIterator trials = tmpSession.getTrialList();
        while (trials.hasNext()) {
            Trial trial = (Trial) trials.next();
            Trial.deleteTrial(db, trial.getID());
        }
        Experiment.deleteExperiment(db, experimentID);
    }

    public void deleteApplication(int applicationID) {
        // create a new DatabaseAPI to handle this request!
        // Why? Because we have to set the experiment to get the trials
        // and that will screw up the state of the current object.
        // the easiest way is to create a new reference to the DB.
        DatabaseAPI tmpSession = new DatabaseAPI();
        // don't initialize (not a new connection) - just reference
        // the other DB connection
        tmpSession.setDB(this.db());

        tmpSession.setApplication(applicationID);
        ListIterator experiments = tmpSession.getExperimentList();
        while (experiments.hasNext()) {
            Experiment experiment = (Experiment) experiments.next();
            tmpSession.setExperiment(experiment.getID());
            ListIterator trials = tmpSession.getTrialList();
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
        System.out.println(buf.toString());

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
};
