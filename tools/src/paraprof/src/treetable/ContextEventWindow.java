package edu.uoregon.tau.paraprof.treetable;

import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.util.Observable;
import java.util.Observer;

import javax.swing.JFrame;
import javax.swing.JMenuBar;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTree;
import javax.swing.table.TableColumn;
import javax.swing.tree.DefaultTreeCellRenderer;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.common.treetable.AbstractTreeTableModel;
import edu.uoregon.tau.common.treetable.JTreeTable;
import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfTrial;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.paraprof.WindowPlacer;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.Thread;

public class ContextEventWindow extends JFrame implements Observer, ParaProfWindow, Printable, ImageExport {

    /**
	 * 
	 */
	private static final long serialVersionUID = 1234017079628001497L;
	private ParaProfTrial ppTrial;
    private Thread thread;

    private ContextEventModel model;
    private JTreeTable treeTable;

    private JScrollPane scrollPane;

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
            this.setTitle("TAU: ParaProf: Context Events for thread: " + "n,c,t, " + thread.getNodeID() + ","
                    + thread.getContextID() + "," + thread.getThreadID() + " - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        }
        ParaProfUtils.setFrameIcon(this);

        if (ParaProf.getHelpWindow().isVisible()) {
            this.help(false);
        }

        setupMenus();
        setupData();

        ParaProf.incrementNumWindows();

    }

    private void setupMenus() {

        JMenuBar mainMenu = new JMenuBar();

        mainMenu.add(ParaProfUtils.createFileMenu((ParaProfWindow) this, this, this));
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
    }

    private void setupData() {
        model = new ContextEventModel(this, ppTrial, thread, false);
        createTreeTable(model);
        addComponents();
    }

    private void createTreeTable(AbstractTreeTableModel model) {
        treeTable = new JTreeTable(model, true, true);

       //final JTree tree = 
        	treeTable.getTree();

        // Add a mouse listener for this tree.
        MouseListener ml = new MouseAdapter() {
            public void mousePressed(MouseEvent evt) {
            //                try {
            //                    int selRow = tree.getRowForLocation(evt.getX(), evt.getY());
            //                    TreePath path = tree.getPathForLocation(evt.getX(), evt.getY());
            //                    if (path != null) {
            //                        TreeTableNode node = (TreeTableNode) path.getLastPathComponent();
            //                        if (ParaProfUtils.rightClick(evt)) {
            //                            JPopupMenu popup;
            //                            if (node.getFunctionProfile() != null) {
            //                                popup = ParaProfUtils.createFunctionClickPopUp(node.getModel().getPPTrial(),
            //                                node.getFunctionProfile().getFunction(), getThread(), treeTable);
            //                                popup.show(treeTable, evt.getX(), evt.getY());
            //                            } else {
            //                                //popup = new JPopupMenu();
            //                            }
            //                        }
            //                    }
            //                } catch (Exception e) {
            //                    ParaProfUtils.handleException(e);
            //                }
            }
        };

        treeTable.addMouseListener(ml);

        DefaultTreeCellRenderer renderer = new DefaultTreeCellRenderer() {

            /**
			 * 
			 */
			private static final long serialVersionUID = 2780103869814842355L;

			public Component getTreeCellRendererComponent(JTree tree, Object value, boolean selected, boolean expanded,
                    boolean leaf, int row, boolean hasFocus) {
                super.getTreeCellRendererComponent(tree, value, selected, expanded, leaf, row, hasFocus);

                this.setIcon(null);

                // shade every other row
                setBackgroundNonSelectionColor(null);
                if (row % 2 == 0) {
                    setBackgroundNonSelectionColor(new Color(235, 235, 235));
                } else {
                    this.setBackground(tree.getBackground());
                }

                return this;
            }
        };
        treeTable.getTree().setCellRenderer(renderer);

        Font font = ParaProf.preferencesWindow.getFont();
        treeTable.setFont(font);
        renderer.setFont(font);

        FontMetrics fontMetrics = getFontMetrics(font);
        treeTable.setRowHeight(fontMetrics.getMaxAscent() + fontMetrics.getMaxDescent() + 3);

        treeTable.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
        treeTable.setAutoResizeMode(JTable.AUTO_RESIZE_SUBSEQUENT_COLUMNS);

        TableColumn col = treeTable.getColumnModel().getColumn(0);

        int nameWidth = 450;

        col.setPreferredWidth(nameWidth);

        //        int numCols = treeTable.getColumnCount();
        //        col = treeTable.getColumnModel().getColumn(numCols - 2);
        //        col.setPreferredWidth(10);
        //        col = treeTable.getColumnModel().getColumn(numCols - 1);
        //        col.setPreferredWidth(10);
    }

    private void addComponents() {
        GridBagConstraints gbc = new GridBagConstraints();
        getContentPane().removeAll();
        getContentPane().setLayout(new GridBagLayout());
        scrollPane = new JScrollPane(treeTable);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.SOUTH;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;
        Utility.addCompItem(this, scrollPane, gbc, 0, 0, GridBagConstraints.REMAINDER, GridBagConstraints.REMAINDER);

        validate();
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
            ParaProfUtils.handleException(e);
            return NO_SUCH_PAGE;
        }
    }

    public void setUnits(int units) {
    // TODO Auto-generated method stub

    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        if (fullWindow) {
            // first draw the column headers
            scrollPane.getColumnHeader().paintAll(g2D);
            // translate past the column headers
            g2D.translate(0, scrollPane.getColumnHeader().getHeight());
            // draw the entire treetable
            treeTable.paintAll(g2D);
        } else {
            scrollPane.paintAll(g2D);
        }

    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        if (fullScreen) {
            Dimension d = treeTable.getSize();
            // we want to draw the column headers as well
            d.setSize(d.getWidth(), d.getHeight() + scrollPane.getColumnHeader().getHeight());
            return d;
        } else {
            return scrollPane.getSize();
        }
    }

    public boolean getTreeMode() {
        return true;
    }

    public void update(Observable o, Object arg) {
        treeTable.repaint();
        //setupData();
    }

    public void closeThisWindow() {
        setVisible(false);
        ppTrial.deleteObserver(this);
        ParaProf.decrementNumWindows();
        dispose();
    }

    public void help(boolean display) {
        ParaProf.getHelpWindow().clearText();
        if (display) {
            ParaProf.getHelpWindow().setVisible(true);
        }
        ParaProf.getHelpWindow().writeText("This is the Context Event Window.\n");
        ParaProf.getHelpWindow().writeText("This window shows context events in a tree-table.)\n");
    }
    public JFrame getFrame() {
        return this;
    }

}
