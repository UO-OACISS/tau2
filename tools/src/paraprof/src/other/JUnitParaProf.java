package edu.uoregon.tau.paraprof.other;

import java.awt.Component;
import java.awt.event.ActionEvent;
import java.io.File;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Vector;

import javax.swing.*;

import junit.extensions.jfcunit.JFCTestCase;
import junit.extensions.jfcunit.JFCTestHelper;
import junit.extensions.jfcunit.TestHelper;
import junit.extensions.jfcunit.finder.ComponentFinder;
import junit.extensions.jfcunit.finder.JMenuItemFinder;
import junit.extensions.jfcunit.finder.NamedComponentFinder;
import junit.framework.Assert;
import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.paraprof.treetable.TreeTableWindow;
import edu.uoregon.tau.perfdmf.*;

public class JUnitParaProf extends JFCTestCase {

    private Vector trials = new Vector();

    public void wildDerivedMetrics(ParaProfTrial ppTrial) {

        //ParaProfManagerWindow paraProfManager = ParaProf.paraProfManager;

        ParaProfMetric ppMetric = DerivedMetrics.applyOperation(ppTrial.getMetric(0),
                ppTrial.getMetric(ppTrial.getNumberOfMetrics() - 1), "Divide");

        Assert.assertNotNull(ppMetric);
        ppMetric = DerivedMetrics.applyOperation(ppTrial.getMetric(1),
                ppTrial.getMetric(ppTrial.getNumberOfMetrics() - 1), "Add");
        Assert.assertNotNull(ppMetric);

        ppMetric = DerivedMetrics.applyOperation(ppTrial.getMetric(2),
                ppTrial.getMetric(ppTrial.getNumberOfMetrics() - 1), "Subtract");
        Assert.assertNotNull(ppMetric);

        ppMetric = DerivedMetrics.applyOperation(ppTrial.getMetric(3),
                ppTrial.getMetric(ppTrial.getNumberOfMetrics() - 1), "Multiply");
        Assert.assertNotNull(ppMetric);

        ppMetric = DerivedMetrics.applyOperation(ppTrial.getMetric(4), "val 10", "Divide");
        Assert.assertNotNull(ppMetric);
        ppMetric = DerivedMetrics.applyOperation(ppTrial.getMetric(5), "val 256", "Add");
        Assert.assertNotNull(ppMetric);
        ppMetric = DerivedMetrics.applyOperation(ppTrial.getMetric(6), "val 8.5", "Subtract");
        Assert.assertNotNull(ppMetric);
        ppMetric = DerivedMetrics.applyOperation(ppTrial.getMetric(7), "val 42.12", "Multiply");
        Assert.assertNotNull(ppMetric);

        //            java.lang.Thread.sleep(5000000);

    }

    public ParaProfTrial loadTrial(File files[], int type) {
        ParaProfApplication application = ParaProf.applicationManager.addApplication();
        application.setName("New Application");

        ParaProfExperiment experiment = application.addExperiment();

        ParaProf.paraProfManagerWindow.addTrial(application, experiment, files, type, false, false);

        Vector trials = experiment.getTrials();

        ParaProfTrial ppTrial = (ParaProfTrial) trials.get(0);

        while (ppTrial.loading()) {
            sleep(500);
        }

        return ppTrial;
    }

    private void sleep(int msec) {
        try {
            java.lang.Thread.sleep(msec);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void checkHistogram(int limit) {
        System.out.println("Testing Histogram Window");

        for (int i = 0; i < trials.size(); i++) {
            int count = limit;
            ParaProfTrial ppTrial = (ParaProfTrial) trials.get(i);
            System.out.println("Trial:" + ppTrial.getName());

            for (Iterator it = ppTrial.getDataSource().getFunctions(); it.hasNext() && count-- != 0;) {

                Function f = (Function) it.next();
                HistogramWindow hw = new HistogramWindow(ppTrial, f, null);
                hw.show();

                for (int n = 1; n <= 100; n += 5) {
                    hw.setNumBins(n);
                    sleep(5);
                }
                //sleep(1);
                hw.closeThisWindow();
            }
        }

    }

    public void checkHistogram(ParaProfTrial ppTrial, Function f) {

        HistogramWindow hw = new HistogramWindow(ppTrial, f, null);
        hw.show();

        for (int n = 1; n <= 100; n += 5) {
            hw.setNumBins(n);
            sleep(5);
        }
        //sleep(1);
        hw.closeThisWindow();
    }

    public void checkThreadWindow(ParaProfTrial ppTrial, edu.uoregon.tau.perfdmf.Thread thread) {

        FunctionBarChartWindow tw = new FunctionBarChartWindow(ppTrial, thread, null, null);
        tw.show();

        sleep(500);
        tw.closeThisWindow();
    }

    public void checkTreeTable(ParaProfTrial ppTrial, edu.uoregon.tau.perfdmf.Thread thread) {

        TreeTableWindow tw = new TreeTableWindow(ppTrial, thread, null);
        tw.show();

        sleep(500);
        tw.closeThisWindow();
    }

    public void checkStatWindow(ParaProfTrial ppTrial, edu.uoregon.tau.perfdmf.Thread thread, boolean userEvent) {

        StatWindow w = new StatWindow(ppTrial, thread, userEvent, null, null);
        w.show();

        sleep(500);
        w.closeThisWindow();
    }

    
    public void checkCallGraphWindow(ParaProfTrial ppTrial, edu.uoregon.tau.perfdmf.Thread thread) {

        CallGraphWindow cgw = new CallGraphWindow(ppTrial, thread, null);
        cgw.show();

        sleep(500);
        cgw.closeThisWindow();
    }

    public ActionEvent createMenuAction(String s) {
        JMenuItem jMenuItem = new JMenuItem(s);
        ActionEvent evt = new ActionEvent(jMenuItem, 0, s);
        return evt;
    }

    JFCTestHelper helper = new JFCTestHelper();

    private Component getMenuItem(JMenu menu, String menuItemName) throws Exception {
        Component comp = TestHelper.findComponent(new JMenuItemFinder(menuItemName), menu.getPopupMenu(), 0);
        assertNotNull("Could not find menu item: " + menuItemName, comp);
        return comp;
    }

    public void checkFunctionWindow(ParaProfTrial ppTrial, Function f) throws Exception {

        FunctionBarChartWindow fw = new FunctionBarChartWindow(ppTrial, f, null);
        fw.show();
        //sleep(200000);

        ComponentFinder crapFinder = new ComponentFinder(JMenuBar.class);

        JMenuBar jMenuBar = (JMenuBar) crapFinder.find(fw, 0);

        NamedComponentFinder menuFinder = new NamedComponentFinder(JMenu.class, "Options");

        //JMenu optionsMenu = (JMenu) menuFinder.find(jMenuBar.getRootPane(), 0);

        JMenu optionsMenu = jMenuBar.getMenu(1);

        //JMenuItemFinder finder = new JMenuItemFinder("Sort By N,C,T");
        //JCheckBoxMenuItem sortByNCT = (JCheckBoxMenuItem) finder.find(fw, 0);

        JCheckBoxMenuItem sortByNCT = (JCheckBoxMenuItem) getMenuItem(optionsMenu, "Sort By N,C,T");
        JCheckBoxMenuItem descending = (JCheckBoxMenuItem) getMenuItem(optionsMenu, "Descending Order");
        JCheckBoxMenuItem percent = (JCheckBoxMenuItem) getMenuItem(optionsMenu, "Show Values as Percent");

        int amount = 50;

        sleep(amount);

        sortByNCT.doClick();
        sleep(amount);
        descending.doClick();
        sleep(amount);
        percent.doClick();
        sleep(amount);
//
//        fw.actionPerformed(createMenuAction("Inclusive"));
//        sleep(amount);
//        fw.actionPerformed(createMenuAction("Show Values as Percent"));
//        sleep(amount);
//        sortByNCT.setSelected(true);
//        fw.actionPerformed(createMenuAction("Sort By N,C,T"));
//        sleep(amount);
//        fw.actionPerformed(createMenuAction("Descending Order"));
//        sleep(amount);
//
//        fw.actionPerformed(createMenuAction("Number of Calls"));
//        sleep(amount);
//        sortByNCT.setSelected(false);
//        fw.actionPerformed(createMenuAction("Sort By N,C,T"));
//        sleep(amount);
//        fw.actionPerformed(createMenuAction("Descending Order"));
//        sleep(amount);
//
//        fw.actionPerformed(createMenuAction("Number of Subroutines"));
//        sleep(amount);
//        sortByNCT.setSelected(true);
//        fw.actionPerformed(createMenuAction("Sort By N,C,T"));
//        sleep(amount);
//        fw.actionPerformed(createMenuAction("Descending Order"));
//        sleep(amount);
//
//        fw.actionPerformed(createMenuAction("Inclusive Per Call"));
//        sleep(amount);
//        sortByNCT.setSelected(false);
//        fw.actionPerformed(createMenuAction("Sort By N,C,T"));
//        sleep(amount);
//        fw.actionPerformed(createMenuAction("Descending Order"));
//        sleep(amount);

        //sleep(200000);

        fw.closeThisWindow();

    }

    public void checkDerivedMetrics() {
        System.out.println("Testing Derived Metrics");

        for (int i = 0; i < trials.size(); i++) {

            ParaProfTrial ppTrial = (ParaProfTrial) trials.get(i);
            System.out.println("Trial:" + ppTrial.getName());

            wildDerivedMetrics(ppTrial);

        }

        //sleep(5000000);
    }

    public void testEverything() throws Exception {

        boolean doDatabase = false;

        System.out.println("---Setting up ParaProf---");
        
        final ParaProf paraProf = new ParaProf();
        paraProf.initialize();
        
        //String args[] = new String[0];
        //ParaProf.main(args);

        System.out.println("---Loading Trials---");

        File[] files = new File[1];
        files[0] = new File("/home/amorris/data/tau/uintah16");
        trials.add(loadTrial(files, 0));

        files[0] = new File("/home/amorris/data/pprof/uintah/pprof.dat");
        trials.add(loadTrial(files, 1));

        files[0] = new File("/home/amorris/data/pprof/mpilieb/pprof.dat");
        trials.add(loadTrial(files, 1));

        files[0] = new File("/home/amorris/data/mpip/mpilieb-noinst.32.25678.mpiP");
        trials.add(loadTrial(files, 3));

        files[0] = new File("/home/amorris/data/dynaprof/simple.papiprobe.790616");
        trials.add(loadTrial(files, 2));

        files[0] = new File("/home/amorris/data/dynaprof/manyfunc.papiprobe.745692");
        trials.add(loadTrial(files, 2));

        files[0] = new File("/home/amorris/data/gprof/mpilieb.out");
        trials.add(loadTrial(files, 5));
        
        files = new File[4];
        files[0] = new File("/home/amorris/data/hpm/mpilieb/eventset1/perfhpm0000.7495902");
        files[1] = new File("/home/amorris/data/hpm/mpilieb/eventset1/perfhpm0001.1466514");
        files[2] = new File("/home/amorris/data/hpm/mpilieb/eventset1/perfhpm0002.6742226");
        files[3] = new File("/home/amorris/data/hpm/mpilieb/eventset1/perfhpm0003.5062700");
        trials.add(loadTrial(files, 4));

        
        files[0] = new File("/home/amorris/data/psrun/91060-tuna184.31362.xml");
        files[1] = new File("/home/amorris/data/psrun/91060-tuna184.31363.xml");
        files[2] = new File("/home/amorris/data/psrun/91060-tuna184.31530.xml");
        files[3] = new File("/home/amorris/data/psrun/91060-tuna184.31558.xml");
        trials.add(loadTrial(files, 6));

        
        
        
        checkDerivedMetrics();


        if (doDatabase) {
            System.out.println("---Checking the database---");
            DatabaseAPI dbApi = new DatabaseAPI();

            dbApi.initialize(ParaProf.preferences.getDatabaseConfigurationFile(), false);

            dbApi.getApplicationList();
            ListIterator apps = dbApi.getApplicationList().listIterator();

            while (apps.hasNext()) {
                Application app = (Application) apps.next();
                dbApi.setApplication(app);
                for (Iterator exps = dbApi.getExperimentList().listIterator(); exps.hasNext();) {
                    Experiment exp = (Experiment) exps.next();
                    dbApi.setExperiment(exp);
                    for (Iterator trls = dbApi.getTrialList(true).listIterator(); trls.hasNext();) {
                        Trial trial = (Trial) trls.next();

                        dbApi.setTrial(trial.getID(), true);
                        DBDataSource dbDataSource = new DBDataSource(dbApi);
                        System.out.println("loading " + trial.getName());
                        dbDataSource.load();

                        ParaProfTrial ppTrial = new ParaProfTrial(trial);
                        //ppTrial.update(dbDataSource);
                        trials.add(ppTrial);
                    }
                }
            }
        }

        System.out.println("---Ready to go---");

        int limit = 2;

        try {
        for (int i = 0; i < trials.size(); i++) {
            int count = limit;
            ParaProfTrial ppTrial = (ParaProfTrial) trials.get(i);
            System.out.println("Trial:" + ppTrial.getName());

            for (Iterator it = ppTrial.getDataSource().getFunctions(); it.hasNext() && count-- != 0;) {
                Function f = (Function) it.next();

                checkHistogram(ppTrial, f);
                checkFunctionWindow(ppTrial, f);
            }

            count = limit;
            for (Iterator it = ppTrial.getDataSource().getAllThreads().iterator(); it.hasNext() && count-- != 0;) {
                edu.uoregon.tau.perfdmf.Thread thread = (edu.uoregon.tau.perfdmf.Thread) it.next();

                checkStatWindow(ppTrial, thread, false);
                checkThreadWindow(ppTrial, thread);
                checkTreeTable(ppTrial, thread);
                checkStatWindow(ppTrial, thread, true);
                checkCallGraphWindow(ppTrial, thread);
            }
        }

        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("---Finished---");

    }

    public static void main(String[] args) {
    }

    /*
     * @see TestCase#setUp()
     */
    protected void setUp() throws Exception {
        super.setUp();
        setHelper(new JFCTestHelper()); // Uses the AWT Event Queue.

    }

    /*
     * @see TestCase#tearDown()
     */
    protected void tearDown() throws Exception {
        super.tearDown();
    }

}
