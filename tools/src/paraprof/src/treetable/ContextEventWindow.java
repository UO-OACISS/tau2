package edu.uoregon.tau.paraprof.treetable;

import java.awt.*;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.util.Observable;
import java.util.Observer;

import javax.swing.JFrame;
import javax.swing.event.TreeExpansionEvent;
import javax.swing.event.TreeExpansionListener;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;

public class ContextEventWindow extends JFrame implements TreeExpansionListener, Observer, ParaProfWindow, Printable,
        UnitListener, ImageExport {

    private ParaProfTrial ppTrial;
    private Thread thread;

    public ContextEventWindow(ParaProfTrial ppTrial, Thread thread) {
        this(ppTrial, thread, null);
    }

    public ContextEventWindow(ParaProfTrial ppTrial, Thread thread, Component invoker) {
        this.ppTrial = ppTrial;
        this.thread = thread;
        ppTrial.addObserver(this);

        //        if (!ppTrial.getDataSource().getReverseDataAvailable()) {
        //            reverseTreeMenuItem.setEnabled(false);
        //        }

        setSize(ParaProfUtils.checkSize(new Dimension(1000, 600)));

        setLocation(WindowPlacer.getNewLocation(this, invoker));

        if (thread.getNodeID() == -1) {
            this.setTitle("TAU: ParaProf: Mean Context Events - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else if (thread.getNodeID() == -2) {
            this.setTitle("TAU: ParaProf: Total Context Events - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else if (thread.getNodeID() == -3) {
            this.setTitle("TAU: ParaProf: Std. Dev. Context Events - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else {
            this.setTitle("TAU: ParaProf: Context Events for thread: " + "n,c,t, " + thread.getNodeID() + "," + thread.getContextID()
                    + "," + thread.getThreadID() + " - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        }
        ParaProfUtils.setFrameIcon(this);

        if (ParaProf.getHelpWindow().isVisible()) {
            this.help(false);
        }

        //setupMenus();
        //setupData();

        ParaProf.incrementNumWindows();

    }

    public void treeCollapsed(TreeExpansionEvent event) {
    // TODO Auto-generated method stub

    }

    public void treeExpanded(TreeExpansionEvent event) {
    // TODO Auto-generated method stub

    }

    public void update(Observable o, Object arg) {
    // TODO Auto-generated method stub

    }

    public void closeThisWindow() {
    // TODO Auto-generated method stub

    }

    public void help(boolean display) {
    // TODO Auto-generated method stub

    }

    public int print(Graphics graphics, PageFormat pageFormat, int pageIndex) throws PrinterException {
        // TODO Auto-generated method stub
        return 0;
    }

    public void setUnits(int units) {
    // TODO Auto-generated method stub

    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
    // TODO Auto-generated method stub

    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        // TODO Auto-generated method stub
        return null;
    }

}
