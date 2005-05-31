package edu.uoregon.tau.paraprof.treetable;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.event.TreeExpansionEvent;
import javax.swing.event.TreeExpansionListener;
import javax.swing.table.TableColumn;

import edu.uoregon.tau.paraprof.*;
import edu.uoregon.tau.paraprof.interfaces.ImageExport;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.paraprof.interfaces.UnitListener;
import edu.uoregon.tau.paraprof.treetable.TreeTableColumn.*;

public class TreeTableWindow extends JFrame implements TreeExpansionListener, Observer, ParaProfWindow, Printable,
        UnitListener, ImageExport {

    private CallPathModel model;
    private JTreeTable treeTable;
    private ParaProfTrial ppTrial;
    private JScrollPane scrollPane;
    private edu.uoregon.tau.dms.dss.Thread thread;
    private int colorMetricID;
    private int units = ParaProf.preferences.getUnits();

    private final JMenuItem showAsTreeMenuItem = new JCheckBoxMenuItem("Show as Call Path Tree");
    private final JMenuItem showInclExclMenuItem = new JCheckBoxMenuItem("Show Inclusive/Exclusive");

    private List columns;

    public TreeTableWindow(ParaProfTrial ppTrial, edu.uoregon.tau.dms.dss.Thread thread) {

        this.ppTrial = ppTrial;
        this.thread = thread;

        setSize(1000, 600);
        setLocation(300, 200);

        //Now set the title.
        if (thread.getNodeID() < 0)
            this.setTitle("Mean Statistics - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        else
            this.setTitle("Thread Statistics: " + "n,c,t, " + thread.getNodeID() + "," + thread.getContextID() + ","
                    + thread.getThreadID() + " - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

        //Set the help window text if required.
        if (ParaProf.helpWindow.isVisible()) {
            this.help(false);
        }

        setupMenus();

        setupData();

        ParaProf.incrementNumWindows();
    }

    private void setupData() {

        getContentPane().removeAll();
        getContentPane().setLayout(new GridBagLayout());

        GridBagConstraints gbc = new GridBagConstraints();

        if (showAsTreeMenuItem.isSelected() == false) {
            showInclExclMenuItem.setEnabled(false);
            showInclExclMenuItem.setSelected(true);
        } else {
            showInclExclMenuItem.setEnabled(true);

            gbc.fill = GridBagConstraints.HORIZONTAL;
            gbc.anchor = GridBagConstraints.NORTH;
            gbc.weightx = 1.0;
            gbc.weighty = 0.0;
            addCompItem(new ColorBar(), gbc, 0, 0, 1, 1);
        }

        if (scrollPane != null) {
            getContentPane().remove(scrollPane);
        }

        columns = new ArrayList();

        for (int i = 0; i < ppTrial.getNumberOfMetrics(); i++) {
            if (showInclExclMenuItem.isSelected()) {
                columns.add(new InclusiveColumn(this, i));
                columns.add(new ExclusiveColumn(this, i));
            } else {
                columns.add(new RegularMetricColumn(this, i));
            }
        }

        columns.add(new NumCallsColumn(this));
        columns.add(new NumSubrColumn(this));

        //columns.add(new MiniHistogramColumn(this));

        model = new CallPathModel(this, ppTrial, thread);
        JTreeTable treeTable = createTreeTable(model);
        scrollPane = new JScrollPane(treeTable);

        for (int i = 0; i < columns.size(); i++) {
            treeTable.setDefaultRenderer(columns.get(i).getClass(),
                    ((TreeTableColumn) columns.get(i)).getCellRenderer());
        }

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.SOUTH;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;
        addCompItem(scrollPane, gbc, 0, 1, GridBagConstraints.REMAINDER, GridBagConstraints.REMAINDER);

        validate();
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

    private void setupMenus() {

        JMenuBar mainMenu = new JMenuBar();

        JMenu optionsMenu = new JMenu("Options");

        ActionListener actionListener = new ActionListener() {

            public void actionPerformed(ActionEvent evt) {
                try {
                    Object EventSrc = evt.getSource();

                    if (EventSrc instanceof JMenuItem) {

                        String arg = evt.getActionCommand();

                        if (arg.equals("Show as Call Path Tree")) {
                            if (showAsTreeMenuItem.isSelected()) {
                                showInclExclMenuItem.setSelected(false);
                            }

                            setupData();
                        } else if (arg.equals("Show Inclusive/Exclusive")) {
                            setupData();
                        }

                    }
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }

            }

        };

        showAsTreeMenuItem.addActionListener(actionListener);
        optionsMenu.add(showAsTreeMenuItem);

        showInclExclMenuItem.addActionListener(actionListener);
        optionsMenu.add(showInclExclMenuItem);

        JMenu unitsSubMenu = ParaProfUtils.createUnitsMenu(this, units, false);
        optionsMenu.add(unitsSubMenu);

        showAsTreeMenuItem.setSelected(true);

        if (!ppTrial.callPathDataPresent()) {
            showAsTreeMenuItem.setSelected(false);
            showAsTreeMenuItem.setEnabled(false);
            showInclExclMenuItem.setSelected(true);
            showInclExclMenuItem.setEnabled(false);
        }

        mainMenu.add(ParaProfUtils.createFileMenu((ParaProfWindow) this, this, this));
        mainMenu.add(optionsMenu);
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);

    }

    private JTreeTable createTreeTable(AbstractTreeTableModel model) {
        //        treeTable = new JTreeTable(model, showAsTreeMenuItem.isSelected());
        treeTable = new JTreeTable(model, true);

        treeTable.getTree().addTreeExpansionListener(this);
        treeTable.getTree().setCellRenderer(new TreePortionCellRenderer());

        treeTable.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
        treeTable.setAutoResizeMode(JTable.AUTO_RESIZE_SUBSEQUENT_COLUMNS);

        TableColumn col = treeTable.getColumnModel().getColumn(0);

        int nameWidth = 500;

        if (ppTrial.getNumberOfMetrics() > 1) {
            nameWidth = 350;
        }
        col.setPreferredWidth(nameWidth);

        int numCols = treeTable.getColumnCount();
        col = treeTable.getColumnModel().getColumn(numCols - 2);
        col.setPreferredWidth(10);
        col = treeTable.getColumnModel().getColumn(numCols - 1);
        col.setPreferredWidth(10);

        return treeTable;
    }

    public void treeCollapsed(TreeExpansionEvent event) {
        TreeTableNode node = (TreeTableNode) event.getPath().getLastPathComponent();
        node.setExpanded(false);
    }

    public void treeExpanded(TreeExpansionEvent event) {
        TreeTableNode node = (TreeTableNode) event.getPath().getLastPathComponent();
        node.setExpanded(true);
    }

    public void update(Observable o, Object arg) {
        // TODO Auto-generated method stub
        //System.err.println("update!");
        setupData();
    }

    public void help(boolean display) {
        // TODO Auto-generated method stub

    }

    public void closeThisWindow() {
        setVisible(false);
        ppTrial.getSystemEvents().deleteObserver(this);
        ParaProf.decrementNumWindows();
        dispose();
    }

    public int print(Graphics graphics, PageFormat pageFormat, int pageIndex) throws PrinterException {
        try {
            if (pageIndex >= 1) {
                return NO_SUCH_PAGE;
            }

            ParaProfUtils.scaleForPrint(graphics, pageFormat, (int) treeTable.getSize().getWidth(),
                    (int) treeTable.getSize().getHeight());
            export((Graphics2D) graphics, false, true, false);

            return Printable.PAGE_EXISTS;
        } catch (Exception e) {
            new ParaProfErrorDialog(e);
            return NO_SUCH_PAGE;
        }
    }

    public boolean getExclusiveInclusiveMode() {
        return showInclExclMenuItem.isSelected();
    }

    public boolean getTreeMode() {
        return showAsTreeMenuItem.isSelected();
    }

    public int getColorMetricID() {
        return colorMetricID;
    }

    public void setUnits(int units) {
        this.units = units;
        treeTable.forceRedraw();
    }

    public int getUnits() {
        return units;
    }

    public List getColumns() {
        return columns;
    }

    public ParaProfTrial getPPTrial() {
        return ppTrial;
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        if (fullWindow) {
            treeTable.paintAll(g2D);
        } else {
            scrollPane.paintAll(g2D);
        }
    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        if (fullScreen) {
            return scrollPane.getSize();
        } else {
            return treeTable.getSize();
        }
    }

}
