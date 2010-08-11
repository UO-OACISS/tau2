package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Observable;
import java.util.Observer;
import java.util.Stack;

import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.JScrollPane;
import javax.swing.JSlider;
import javax.swing.ToolTipManager;
import javax.swing.border.BevelBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.jgraph.JGraph;
import org.jgraph.graph.CellView;
import org.jgraph.graph.ConnectionSet;
import org.jgraph.graph.DefaultEdge;
import org.jgraph.graph.DefaultGraphCell;
import org.jgraph.graph.DefaultGraphModel;
import org.jgraph.graph.DefaultGraphSelectionModel;
import org.jgraph.graph.DefaultPort;
import org.jgraph.graph.Edge;
import org.jgraph.graph.GraphConstants;
import org.jgraph.graph.GraphModel;
import org.jgraph.graph.Port;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.paraprof.enums.CallGraphOption;
import edu.uoregon.tau.paraprof.graph.Layout;
import edu.uoregon.tau.paraprof.graph.Vertex;
import edu.uoregon.tau.paraprof.graph.Vertex.BackEdge;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.CallPathUtilFuncs;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Thread;

/**
 * CallGraphWindow.java
 * This window displays the callpath data as a graph.
 *   
 * TODO: Infinite.  The 2nd half of Sugiyama's algorithm should probably
 *       be implemented.  Plenty of other things could be done as well, such
 *       as using box height as another metric.
 *       
 * <P>CVS $Id: CallGraphWindow.java,v 1.15 2009/09/10 00:13:44 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.15 $
 */
public class CallGraphWindow extends JFrame implements ActionListener, KeyListener, MouseWheelListener, ChangeListener, Observer, ImageExport,
        Printable, ParaProfWindow {

    /**
	 * 
	 */
	private static final long serialVersionUID = -8532506804592254096L;
	private static final int MARGIN = 20;
    //private static final int HORIZONTAL_SPACING = 10;
    private static final int VERTICAL_SPACING = 120;

    private ParaProfTrial ppTrial;
    private Thread thread;

    private JMenu optionsMenu;

    private JCheckBoxMenuItem slidersCheckBox;

    private Graph graph;
    private JScrollPane jGraphPane;

    private CallGraphOption widthOption = CallGraphOption.INCLUSIVE;
    private CallGraphOption colorOption = CallGraphOption.EXCLUSIVE;

    private int boxWidth = 120;

    private JLabel boxWidthLabel = new JLabel("Box width");
    private JSlider boxWidthSlider = new JSlider(0, 500, boxWidth);

    private List<FunctionProfile> functionProfileList;
    private DefaultGraphModel model;
    private List<GraphCell> graphCellList;
    private Object[] cells;
    private List<List<Vertex>> levels;
    private List<BackEdge> backEdges;
    private Map<FunctionProfile, Vertex> vertexMap;

    private Metric widthMetric;
    private Metric colorMetric;
    private Font font;
    private int boxHeight;
    private double scale = 1.0;

    private static class GraphSelectionModel extends DefaultGraphSelectionModel {

        /**
		 * 
		 */
		private static final long serialVersionUID = -5718582372004816966L;

		GraphSelectionModel(JGraph graph) {
            super(graph);
        }

        // this getSelectables has been overridden from DefaultGraphSelectionModel 
        // so that edges never get selected
        /**
         * Returns the cells that are currently selectable.
         * The array is ordered so that the top-most cell
         * appears first.<br>
         */
        public Object[] getSelectables() {
            //            return null;
            if (isChildrenSelectable()) {
                ArrayList<Object> result = new ArrayList<Object>();
                // Roots Are Always Selectable
                Stack<Object> s = new Stack<Object>();
                GraphModel model = this.graph.getModel();
                for (int i = 0; i < model.getRootCount(); i++)
                    s.add(model.getRootAt(i));
                while (!s.isEmpty()) {
                    Object cell = s.pop();
                    if (!model.isPort(cell) && !model.isEdge(cell))
                        result.add(cell);
                    if (isChildrenSelectable(cell)) {
                        for (int i = 0; i < model.getChildCount(cell); i++)
                            s.add(model.getChild(cell, i));
                    }
                }
                return result.toArray();
            }
            return this.graph.getRoots();
        }

    }

    public class GraphCell extends DefaultGraphCell {

        /**
		 * 
		 */
		private static final long serialVersionUID = -8524078198153715353L;
		private final Function function;
        private final FunctionProfile functionProfile;
        private final Vertex vertex;

        public GraphCell(Vertex v) {
            super(((FunctionProfile) v.getUserObject()).getFunction());
            functionProfile = (FunctionProfile) v.getUserObject();
            function = functionProfile.getFunction();
            vertex = v;
        }

        public String getToolTipString() {
            String result = "<html>" + function;

            if (widthOption != CallGraphOption.STATIC && widthOption != CallGraphOption.NAME_LENGTH) {
                float widthValue = (float) getValue(functionProfile, widthOption, 1.0, widthMetric);
                result = result + "<br>Width Value (" + widthOption;
                if (widthOption != CallGraphOption.NUMCALLS && widthOption != CallGraphOption.NUMSUBR) {
                    result = result + ", " + widthMetric.getName();
                }
                result = result + ") : " + widthValue;
            }

            if (colorOption != CallGraphOption.STATIC) {
                float colorValue = (float) getValue(functionProfile, colorOption, 1.0, colorMetric);
                result = result + "<br>Color Value (" + colorOption;
                if (colorOption != CallGraphOption.NUMCALLS && colorOption != CallGraphOption.NUMSUBR) {
                    result = result + ", " + colorMetric.getName();
                }
                result = result + ") : " + colorValue;
            }

            return result;
        }

        public Function getFunction() {
            return function;
        }

        public Vertex getVertex() {
            return vertex;
        }
    }

    private class Graph extends JGraph implements MouseListener {

        /**
		 * 
		 */
		private static final long serialVersionUID = 5752402574670873205L;

		public String getToolTipText(MouseEvent event) {
            double x = event.getX() / this.getScale();
            double y = event.getY() / this.getScale();

            GraphCell gc = callGraphWindow.getGraphCellForLocation((int) x, (int) y);

            if (gc != null) {
                return gc.getToolTipString();
            }

            return null;
        }

        public void mousePressed(MouseEvent evt) {}

        public void mouseReleased(MouseEvent evt) {}

        public void mouseEntered(MouseEvent evt) {}

        public void mouseExited(MouseEvent evt) {}

        public void mouseClicked(MouseEvent evt) {
            try {
                // Get Cell under Mousepointer

                // scale the x and y (we could be zoomed in or out)
                double x = evt.getX() / this.getScale();
                double y = evt.getY() / this.getScale();

                GraphCell gc = callGraphWindow.getGraphCellForLocation((int) x, (int) y);

                if (gc != null) {
                    Function f = ((Function) gc.getFunction());

                    if (ParaProfUtils.rightClick(evt)) {
                        JPopupMenu popup = ParaProfUtils.createFunctionClickPopUp(ppTrial, f, thread, this);
                        popup.show(this, evt.getX(), evt.getY());
                    } else {
                        ppTrial.toggleHighlightedFunction(f);
                    }
                }
            } catch (Exception e) {
                ParaProfUtils.handleException(e);
            }

        }

        // override the JGraph getPreferredSize to add 10 pixels
        public Dimension getPreferredSize() {
            Dimension inner = super.getPreferredSize();
            inner.setSize(inner.width + 10, inner.height + 10);
            return inner;
        }

        public Graph(GraphModel gm, CallGraphWindow cgw) {
            super(gm);
            this.callGraphWindow = cgw;
            this.setSelectionModel(new GraphSelectionModel(this));
        }

        private CallGraphWindow callGraphWindow;
    }

    public CallGraphWindow(ParaProfTrial ppTrial, Thread thread, Component invoker) {
        this.ppTrial = ppTrial;
        ppTrial.addObserver(this);
        this.colorMetric = ppTrial.getDefaultMetric();
        this.widthMetric = ppTrial.getDefaultMetric();

        this.thread = thread;

        if (ppTrial.callPathDataPresent())
            CallPathUtilFuncs.buildThreadRelations(ppTrial.getDataSource(), thread);

        functionProfileList = thread.getFunctionProfiles();

        if (thread.getNodeID() == -1) {
            this.setTitle("TAU: ParaProf: Mean Call Graph - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else if (thread.getNodeID() == -3) {
            this.setTitle("TAU: ParaProf: Standard Deviation Call Graph - "
                    + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        } else {
            this.setTitle("TAU: ParaProf: Call Graph for n,c,t, " + thread.getNodeID() + "," + thread.getContextID() + ","
                    + thread.getThreadID() + " - " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        }
        ParaProfUtils.setFrameIcon(this);

        //Add some window listener code
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                thisWindowClosing(evt);
            }
        });

        //Set the help window text if required.
        if (ParaProf.getHelpWindow().isVisible()) {
            this.help(false);
        }

        setupMenus();

        boxWidthSlider.setPaintTicks(true);
        boxWidthSlider.setMajorTickSpacing(50);
        boxWidthSlider.setMinorTickSpacing(10);
        boxWidthSlider.setPaintLabels(true);
        boxWidthSlider.setSnapToTicks(false);
        boxWidthSlider.addChangeListener(this);
        boxWidthSlider.addKeyListener(this);

        GridBagLayout gbl = new GridBagLayout();
        this.getContentPane().setLayout(gbl);

        // obtain the font and its metrics
        font = ParaProf.preferencesWindow.getFont();
        FontMetrics fm = getFontMetrics(font);

        // set the box height to the font height + 5
        boxHeight = fm.getHeight() + 5;

        // Create the colorbar
        ColorBar cb = new ColorBar();

        GridBagConstraints gbc = new GridBagConstraints();

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;
        gbc.weightx = 1.0;
        gbc.weighty = 0.0;
        addCompItem(cb, gbc, 0, 0, 2, 1);

        // create the graph
        createGraph();

        // sizing, get the preferred size of the graph and cut it down if necessary
        Dimension prefSize = jGraphPane.getPreferredSize();

        prefSize.width += 25;
        prefSize.height += 75 + 20;

        if (prefSize.width > 1000)
            prefSize.width = 1000;

        if (prefSize.height > 1000)
            prefSize.height = 1000;

        setSize(ParaProfUtils.checkSize(prefSize));
        setLocation(WindowPlacer.getNewLocation(this, invoker));

        this.setVisible(true);

        ParaProf.incrementNumWindows();
    }

    private Component createWidthMetricMenu(final CallGraphOption option, boolean enabled, ButtonGroup group) {
        JRadioButtonMenuItem button = null;

        if (ppTrial.getNumberOfMetrics() == 1) {
            button = new JRadioButtonMenuItem(option.toString(), enabled);

            button.addActionListener(new ActionListener() {
                public void actionPerformed(ActionEvent evt) {
                    widthOption = option;
                    recreateGraph();
                }
            });
            group.add(button);
            return button;
        } else {
            JMenu subSubMenu = new JMenu(option.toString() + "...");

            for (Iterator<Metric> it = ppTrial.getMetrics().iterator(); it.hasNext();) {
                Metric metric = it.next();

                if (metric == this.widthMetric && enabled) {
                    button = new JRadioButtonMenuItem(metric.getName(), true);
                } else {
                    button = new JRadioButtonMenuItem(metric.getName());
                }
                final Metric m = metric;

                button.addActionListener(new ActionListener() {
                    final Metric metric = m;

                    public void actionPerformed(ActionEvent evt) {
                        widthOption = option;
                        widthMetric = metric;
                        recreateGraph();
                    }
                });
                group.add(button);
                subSubMenu.add(button);
            }
            return subSubMenu;
        }
    }

    private Component createColorMetricMenu(final CallGraphOption option, boolean enabled, ButtonGroup group) {
        JRadioButtonMenuItem button = null;

        if (ppTrial.getNumberOfMetrics() == 1) {
            button = new JRadioButtonMenuItem(option.toString(), enabled);

            button.addActionListener(new ActionListener() {
                public void actionPerformed(ActionEvent evt) {
                    colorOption = option;
                    recreateGraph();
                }
            });
            group.add(button);
            return button;
        } else {
            JMenu subSubMenu = new JMenu(option.toString() + "...");

            for (Iterator<Metric> it = ppTrial.getMetrics().iterator(); it.hasNext();) {
                Metric metric = it.next();

                if (metric == this.widthMetric && enabled) {
                    button = new JRadioButtonMenuItem(metric.getName(), true);
                } else {
                    button = new JRadioButtonMenuItem(metric.getName());
                }
                final Metric m = metric;

                button.addActionListener(new ActionListener() {
                    final Metric metric = m;

                    public void actionPerformed(ActionEvent evt) {
                        colorOption = option;
                        colorMetric = metric;
                        recreateGraph();
                    }
                });
                group.add(button);
                subSubMenu.add(button);
            }
            return subSubMenu;
        }
    }

    private void setupMenus() {

        JMenuBar mainMenu = new JMenuBar();
        JMenu subMenu = null;

        // options menu 
        optionsMenu = new JMenu("Options");

        ButtonGroup group = null;
        JRadioButtonMenuItem button = null;

        slidersCheckBox = new JCheckBoxMenuItem("Show Width Slider", false);
        slidersCheckBox.addActionListener(this);
        optionsMenu.add(slidersCheckBox);

        // box width submenu
        subMenu = new JMenu("Box width by...");
        group = new ButtonGroup();

        subMenu.add(createWidthMetricMenu(CallGraphOption.EXCLUSIVE, CallGraphOption.EXCLUSIVE == widthOption, group));
        subMenu.add(createWidthMetricMenu(CallGraphOption.INCLUSIVE, CallGraphOption.INCLUSIVE == widthOption, group));
        subMenu.add(createWidthMetricMenu(CallGraphOption.EXCLUSIVE_PER_CALL, CallGraphOption.EXCLUSIVE_PER_CALL == widthOption,
                group));
        subMenu.add(createWidthMetricMenu(CallGraphOption.INCLUSIVE_PER_CALL, CallGraphOption.INCLUSIVE_PER_CALL == widthOption,
                group));

        button = new JRadioButtonMenuItem("Number of Calls", CallGraphOption.NUMCALLS == widthOption);
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                widthOption = CallGraphOption.NUMCALLS;
                recreateGraph();
            }
        });
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Number of Child Calls", CallGraphOption.NUMSUBR == widthOption);
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                widthOption = CallGraphOption.NUMSUBR;
                recreateGraph();
            }
        });
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Static", CallGraphOption.STATIC == widthOption);
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                widthOption = CallGraphOption.STATIC;
                recreateGraph();
            }
        });
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Name Length", CallGraphOption.NAME_LENGTH == widthOption);
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                widthOption = CallGraphOption.NAME_LENGTH;
                recreateGraph();
            }
        });
        group.add(button);
        subMenu.add(button);

        optionsMenu.add(subMenu);

        // box color submenu
        subMenu = new JMenu("Box color by...");
        group = new ButtonGroup();

        subMenu.add(createColorMetricMenu(CallGraphOption.EXCLUSIVE, CallGraphOption.EXCLUSIVE == colorOption, group));
        subMenu.add(createColorMetricMenu(CallGraphOption.INCLUSIVE, CallGraphOption.INCLUSIVE == colorOption, group));
        subMenu.add(createColorMetricMenu(CallGraphOption.EXCLUSIVE_PER_CALL, CallGraphOption.EXCLUSIVE_PER_CALL == colorOption,
                group));
        subMenu.add(createColorMetricMenu(CallGraphOption.INCLUSIVE_PER_CALL, CallGraphOption.INCLUSIVE_PER_CALL == colorOption,
                group));

        button = new JRadioButtonMenuItem("Number of Calls", CallGraphOption.NUMCALLS == colorOption);
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                colorOption = CallGraphOption.NUMCALLS;
                recreateGraph();
            }
        });
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Number of Child Calls", CallGraphOption.NUMSUBR == colorOption);
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                colorOption = CallGraphOption.NUMSUBR;
                recreateGraph();
            }
        });
        group.add(button);
        subMenu.add(button);

        button = new JRadioButtonMenuItem("Static", CallGraphOption.STATIC == colorOption);
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                colorOption = CallGraphOption.STATIC;
                recreateGraph();
            }
        });
        group.add(button);
        subMenu.add(button);

        optionsMenu.add(subMenu);

        // now add all the menus to the main menu
        mainMenu.add(ParaProfUtils.createFileMenu(this, this, this));
        mainMenu.add(optionsMenu);
        //mainMenu.add(ParaProfUtils.createTrialMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);
    }

    private double getMaxValue(CallGraphOption option, Metric metric) {
        return getMaxValue(option, metric.getID());
    }

    private double getMaxValue(CallGraphOption option, int metric) {
        double maxValue = 1;
        int snapshot = ppTrial.getSelectedSnapshot();
        if (snapshot == -1) {
            snapshot = thread.getNumSnapshots() - 1;
        }
        if (option == CallGraphOption.EXCLUSIVE) {
            maxValue = thread.getMaxExclusive(metric, snapshot);
        } else if (option == CallGraphOption.INCLUSIVE) {
            maxValue = thread.getMaxInclusive(metric, snapshot);
        } else if (option == CallGraphOption.NUMCALLS) {
            maxValue = thread.getMaxNumCalls(snapshot);
        } else if (option == CallGraphOption.NUMSUBR) {
            maxValue = thread.getMaxNumSubr(snapshot);
        } else if (option == CallGraphOption.INCLUSIVE_PER_CALL) {
            maxValue = thread.getMaxInclusivePerCall(metric, snapshot);
        } else if (option == CallGraphOption.EXCLUSIVE_PER_CALL) {
            maxValue = thread.getMaxExclusivePerCall(metric, snapshot);
        } else if (option == CallGraphOption.STATIC) {
            maxValue = 1;
        } else if (this.widthOption == CallGraphOption.NAME_LENGTH) {
            maxValue = 1;
        } else {
            throw new ParaProfException("Unexpected CallGraphOption : " + option);
        }
        return maxValue;
    }

    private double getValue(FunctionProfile fp, CallGraphOption option, double maxValue, Metric metric) {
        return getValue(fp, option, maxValue, metric.getID());
    }

    private double getValue(FunctionProfile fp, CallGraphOption option, double maxValue, int metric) {
        int snapshot = ppTrial.getSelectedSnapshot();
        double value = 1;
        if (option == CallGraphOption.STATIC) {
            value = 1;
        } else if (option == CallGraphOption.EXCLUSIVE) {
            value = fp.getExclusive(snapshot, metric) / maxValue;
        } else if (option == CallGraphOption.INCLUSIVE) {
            value = fp.getInclusive(snapshot, metric) / maxValue;
        } else if (option == CallGraphOption.NUMCALLS) {
            value = fp.getNumCalls(snapshot) / maxValue;
        } else if (option == CallGraphOption.NUMSUBR) {
            value = fp.getNumSubr(snapshot) / maxValue;
        } else if (option == CallGraphOption.INCLUSIVE_PER_CALL) {
            value = fp.getInclusivePerCall(snapshot, metric) / maxValue;
        } else if (option == CallGraphOption.EXCLUSIVE_PER_CALL) {
            value = fp.getExclusivePerCall(snapshot, metric) / maxValue;
        } else if (option == CallGraphOption.STATIC) {
            value = 1;
        } else {
            throw new ParaProfException("Unexpected CallGraphOption : " + option);
        }

        return value;
    }

    private int getWidth(FunctionProfile fp, double maxValue) {
        int width = 0;
        if (this.widthOption == CallGraphOption.NAME_LENGTH) {
            FontMetrics fm = getFontMetrics(this.font);
            width = fm.stringWidth(fp.getName()) + 5;
        } else {
            width = (int) (boxWidth * getValue(fp, this.widthOption, maxValue, widthMetric));
        }
        return width;
    }

    private List<List<Vertex>> constructGraph() {

        vertexMap = new HashMap<FunctionProfile, Vertex>();
        backEdges = new ArrayList<BackEdge>();

        double maxWidthValue = getMaxValue(this.widthOption, widthMetric);
        double maxColorValue = getMaxValue(this.colorOption, colorMetric);

        for (int i = 0; i < functionProfileList.size(); i++) {
            FunctionProfile fp = functionProfileList.get(i);
            if (fp == null) // skip it if this thread didn't call this function
                continue;

            if (!fp.isCallPathFunction()) { // skip callpath functions (we only want the actual functions)

                Vertex v = new Vertex(fp, getWidth(fp, maxWidthValue), boxHeight);
                v.setColorRatio((float) getValue(fp, this.colorOption, maxColorValue, colorMetric));
                vertexMap.put(fp, v);
            }
        }

        // now we follow the call paths and eliminate back edges
        Stack<FunctionProfile> toVisit = new Stack<FunctionProfile>();
        Stack<FunctionProfile> currentPath = new Stack<FunctionProfile>();

        for (int i = 0; i < functionProfileList.size(); i++) {
            FunctionProfile fp = functionProfileList.get(i);
            if (fp == null) // skip it if this thread didn't call this function
                continue;

            if (!fp.isCallPathFunction()) { // skip callpath functions (we only want the actual functions)

                // get the vertex for this FunctionProfile 
                Vertex root = vertexMap.get(fp);

                if (!root.getVisited()) {

                    currentPath.add(fp);
                    toVisit.add(null); // null in the toVisit stack marks the end of a set of children (they must get pushed into the stack prior to the children)

                    // add all the children to the toVisit list
                    for (Iterator<FunctionProfile> it = fp.getChildProfiles(); it.hasNext();) {
                        FunctionProfile childFp = it.next();
                        toVisit.add(childFp);
                    }

                    while (!toVisit.empty()) {
                        FunctionProfile childFp = toVisit.pop();

                        if (childFp == null) {
                            // this marks the end of a set of children, so pop the current path
                            // and move on to the next one in toVisit
                            currentPath.pop();
                            continue;
                        }

                        Vertex child = vertexMap.get(childFp);
                        FunctionProfile parentFp = currentPath.peek();

                        Vertex parent = vertexMap.get(parentFp);

                        // run through the currentPath and see if childFp is in it, if so, this is a backedge
                        boolean back = false;
                        for (Iterator<FunctionProfile> it = currentPath.iterator(); it.hasNext();) {
                            if (it.next() == childFp) {
                                back = true;
                                break;
                            }
                        }

                        if (back) {
                            backEdges.add(new BackEdge(parent, child));
                        } else {

                            boolean found = false;
                            for (int j = 0; j < parent.getChildren().size(); j++) {
                                if (parent.getChildren().get(j) == child)
                                    found = true;
                            }
                            if (!found)
                                parent.getChildren().add(child);

                            found = false;
                            for (int j = 0; j < child.getParents().size(); j++) {
                                if (child.getParents().get(j) == parent)
                                    found = true;
                            }
                            if (!found)
                                child.getParents().add(parent);

                            if (child.getVisited() == false) {

                                child.setVisited(true);

                                currentPath.add(childFp);

                                toVisit.add(null);
                                for (Iterator<FunctionProfile> it = childFp.getChildProfiles(); it.hasNext();) {
                                    FunctionProfile grandChildFunction = it.next();

                                    toVisit.add(grandChildFunction);
                                }
                            }

                        }
                    }
                }
            }
        }
        // now we should have a DAG, now find the roots

        // Find Roots
        List<Vertex> roots = Layout.findRoots(vertexMap);

        // Assigning Levels
        for (int i = 0; i < functionProfileList.size(); i++) {
            FunctionProfile fp = functionProfileList.get(i);
            if (fp == null)
                continue;

            if (!fp.isCallPathFunction()) {
                Vertex vertex = vertexMap.get(fp);

                if (vertex.getLevel() == -1) {
                    Layout.assignLevel(vertex);
                }
            }

        }

        // Insert Dummies
        for (int i = 0; i < functionProfileList.size(); i++) {
            FunctionProfile fp = functionProfileList.get(i);
            if (fp == null)
                continue;

            if (!fp.isCallPathFunction()) {
                Vertex vertex = vertexMap.get(fp);
                Layout.insertDummies(vertex);
            }

        }

        // fill level lists
        for (int i = 0; i < functionProfileList.size(); i++) {
            FunctionProfile fp = functionProfileList.get(i);
            if (fp == null)
                continue;

            if (!fp.isCallPathFunction()) {
                Vertex vertex = vertexMap.get(fp);
                vertex.setVisited(false);
            }

        }

        List<List<Vertex>> levels = new ArrayList<List<Vertex>>();

        // Fill Levels
        for (int i = 0; i < roots.size(); i++) {
            Vertex root = roots.get(i);
            Layout.fillLevels(root, levels, 0);
        }

        // Order Levels
        Layout.runSugiyama(levels);
        Layout.assignPositions(levels);
        return levels;
    }

    private void createGraph() {

        levels = constructGraph();

        // Construct Model and Graph
        model = new DefaultGraphModel();
        graph = new Graph(model, this);
        graph.addMouseListener(graph);
        graph.addKeyListener(this);
	graph.addMouseWheelListener(this);

        //graph.setAntiAliased(true);
        ToolTipManager.sharedInstance().registerComponent(graph);

        createCustomGraph(levels, backEdges);

        jGraphPane = new JScrollPane(graph);

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.SOUTH;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;
        addCompItem(jGraphPane, gbc, 0, 1, GridBagConstraints.REMAINDER, GridBagConstraints.REMAINDER);

    }

    void recreateGraph() {

        for (int i = 0; i < graphCellList.size(); i++) {
            DefaultGraphCell dgc = graphCellList.get(i);
            dgc.removeAllChildren();
        }

        model.remove(cells);

        reassignWidths(levels);
        Layout.assignPositions(levels);

        createCustomGraph(levels, backEdges);
    }

    void reassignWidths(List<List<Vertex>> levels) {

        double maxWidthValue = getMaxValue(this.widthOption, widthMetric);
        double maxColorValue = getMaxValue(this.colorOption, colorMetric);

        for (int i = 0; i < levels.size(); i++) {
            List<Vertex> level = levels.get(i);

            for (int j = 0; j < level.size(); j++) {
                Vertex v = level.get(j);

                if (v.getUserObject() != null) {
                    FunctionProfile fp = (FunctionProfile) v.getUserObject();

                    v.setWidth(getWidth(fp, maxWidthValue));

                    // we have to do this check here since we're treating it like a struct
                    if (v.getWidth() < 5)
                        v.setWidth(5);

                    v.setColorRatio((float) getValue(fp, this.colorOption, maxColorValue, colorMetric));

                    v.setHeight(boxHeight);
                }
            }
        }
    }

    @SuppressWarnings("rawtypes")
	void createCustomGraph(List<List<Vertex>> levels, List<BackEdge> backEdges) {

        Map<DefaultGraphCell, Map> attributes = new HashMap<DefaultGraphCell, Map>();

        graphCellList = new ArrayList<GraphCell>();
        List<DefaultGraphCell> cellList = new ArrayList<DefaultGraphCell>();

        for (int i = 0; i < levels.size(); i++) {
            List<Vertex> level = levels.get(i);

            for (int j = 0; j < level.size(); j++) {
                Vertex v = level.get(j);

                GraphCell dgc = null;

                if (v.getUserObject() != null) {
                    dgc = createGraphCell(v, v.getPosition() - (v.getWidth() / 2), MARGIN + i * VERTICAL_SPACING, v.getHeight(),
                            v.getWidth(), v.getColorRatio(), attributes);

                    v.setGraphCell(dgc);
                    cellList.add(dgc);
                    graphCellList.add(dgc);
                } else {
                    // dummy node, don't make a graph cell
                }

            }
        }

        ConnectionSet cs = new ConnectionSet();
        List<DefaultEdge> edgeList = new ArrayList<DefaultEdge>();

        for (int i = 0; i < levels.size(); i++) {
            List<Vertex> level = levels.get(i);

            for (int j = 0; j < level.size(); j++) {
                Vertex v = level.get(j);

                if (v.getUserObject() != null) {
                    GraphCell dgcParent = v.getGraphCell();

                    for (Iterator<Vertex> it = v.getChildren().iterator(); it.hasNext();) {
                        Vertex child = it.next();

                        if (child.getUserObject() != null) {
                            // simply connect the GraphCells
                            GraphCell dgcChild = child.getGraphCell();

                            DefaultEdge e = createEdge(dgcParent, dgcChild, attributes, cs, null);
                            cellList.add(e);
                            edgeList.add(e);
                        } else {
                            // follow the chain of vertices whose functions are null to find the real
                            // child vertex.  All of these inbetween vertices are "dummy nodes"
                            ArrayList<Point> points = new ArrayList<Point>();
                            int l = 1; // how many levels down this dummy node is

                            points.add(new Point(3000, 3000)); // this point's position doesn't matter because of the connect call

                            while (child.getUserObject() == null) {
                                points.add(new Point(child.getPosition(), MARGIN + ((i + l) * VERTICAL_SPACING) + (boxHeight / 2)));
                                // find the end of the dummy chain
                                child = child.getChildren().get(0); // there can only be exactly one child
                                l++;
                            }

                            points.add(new Point(3000, 3000)); // this point's position doesn't matter because of the connect call

                            DefaultEdge e = createEdge(dgcParent, child.getGraphCell(), attributes, cs, points);
                            cellList.add(e);
                            edgeList.add(e);
                        }
                    }
                }
            }
        }

        // Now create the back edges
        for (int i = 0; i < backEdges.size(); i++) {
            BackEdge backEdge = backEdges.get(i);

            ArrayList<Point> points = new ArrayList<Point>();

            // this point's position doesn't matter because of the connect call
            points.add(new Point(3000, 3000));

            points.add(new Point(backEdge.a.getPosition() + backEdge.a.getWidth() / 2 + 50, (backEdge.a.getLevel())
                    * VERTICAL_SPACING + MARGIN + (boxHeight / 2)));

            points.add(new Point(backEdge.b.getPosition() + 25, (backEdge.b.getLevel()) * VERTICAL_SPACING - 25 + MARGIN));

            // this point's position doesn't matter because of the connect call
            points.add(new Point(3000, 3000));

            DefaultEdge edge = createEdge(backEdge.a.getGraphCell(), backEdge.b.getGraphCell(), attributes, cs, points);
            cellList.add(edge);
            edgeList.add(edge);
        }

        cells = cellList.toArray();

        model.insert(cells, attributes, cs, null, null);

        // now make sure everything is visible (this fixes big edges that go off the top of the screen
        moveDownToVisible(cellList, edgeList);
    }

    @SuppressWarnings("rawtypes")
	private void moveDownToVisible(List<DefaultGraphCell> cellList, List<DefaultEdge> edgeList) {
        // find the minimum y value of any edge point and shift everything down by that much
        int minY = 0;

        for (int i = 0; i < edgeList.size(); i++) {
            CellView cv = graph.getGraphLayoutCache().getMapping(edgeList.get(i), false);
            Rectangle2D rc = cv.getBounds();
            if (rc.getY() < minY) {
                minY = (int) rc.getY();
            }
        }

        if (minY != 0) {
            minY -= 5; // shift by a minimum of 5
            Map<Object, Map> attributeMap = new HashMap<Object, Map>();

            for (int i = 0; i < cellList.size(); i++) {
                DefaultGraphCell dgc = cellList.get(i);
                Map attrib = dgc.getAttributes();
                translate(attrib, 0, -minY);
                attributeMap.put(dgc, attrib);
            }
            graph.getGraphLayoutCache().edit(attributeMap, null, null, null);
        }
    }

    @SuppressWarnings({ "unchecked", "rawtypes" })
	public static void translate(Map map, double dx, double dy) {
        // Translate Bounds 
        if (GraphConstants.isMoveable(map)) {

            Rectangle2D bounds = GraphConstants.getBounds(map);
            if (bounds != null) {
                int moveableAxis = GraphConstants.getMoveableAxis(map);
                if (moveableAxis == GraphConstants.X_AXIS)
                    dy = 0;
                else if (moveableAxis == GraphConstants.Y_AXIS)
                    dx = 0;
                bounds.setFrame(Math.max(0, bounds.getX() + dx), Math.max(0, bounds.getY() + dy), bounds.getWidth(),
                        bounds.getHeight());
                GraphConstants.setBounds(map, bounds);
            }
            // Translate Points 
            java.util.List<Point> points = GraphConstants.getPoints(map);
            if (points != null) {
                for (int i = 0; i < points.size(); i++) {
                    Object obj = points.get(i);
                    if (obj instanceof Point2D) {
                        Point2D pt = (Point2D) obj;
                        pt.setLocation(Math.max(0, pt.getX() + dx), Math.max(0, pt.getY() + dy));
                    }
                }
                GraphConstants.setPoints(map, points);
            }
        }
    }

    @SuppressWarnings("rawtypes")
	public GraphCell createGraphCell(Vertex v, int x, int y, int height, int width, float color, Map<DefaultGraphCell, Map> attributes) {
        // Create Hello Vertex
        GraphCell vertex = new GraphCell(v);

        // Create Hello Vertex Attributes
        Map attrib = new Hashtable();
        attributes.put(vertex, attrib);

        //        System.out.println("placing at x=" + x + ", y = " + y);

        // Set bounds
        //Rectangle helloBounds = new Rectangle(20, 20, 40, 20);
        Rectangle bounds = new Rectangle(x, y, width, height);
        GraphConstants.setBounds(attrib, bounds);

        // Set black border
        GraphConstants.setBorderColor(attrib, Color.black);

        // Set fill color

        if (this.colorOption == CallGraphOption.STATIC) {
            GraphConstants.setBackground(attrib, Color.orange);
            GraphConstants.setForeground(attrib, Color.black);
        } else {
            GraphConstants.setBackground(attrib, ColorBar.getColor(color));
            GraphConstants.setForeground(attrib, ColorBar.getContrast(ColorBar.getColor(color)));
        }

        GraphConstants.setOpaque(attrib, true);
        GraphConstants.setEditable(attrib, false);
        GraphConstants.setFont(attrib, font);

        // Set raised border
        GraphConstants.setBorder(attrib, BorderFactory.createRaisedBevelBorder());

        //GraphConstants.setBorder(helloAttrib, BorderFactory.createBevelBorder(BevelBorder.RAISED,Color.blue,Color.blue));
        //GraphConstants.setBorder(helloAttrib, BorderFactory.createBevelBorder(BevelBorder.RAISED,Color.black,Color.black));
        //GraphConstants.setBorder(helloAttrib, BorderFactory.createBevelBorder(BevelBorder.RAISED,Color.black,Color.black));

        // Add a Port
        DefaultPort hp = new DefaultPort();
        vertex.add(hp);

        return vertex;
    }

    @SuppressWarnings("rawtypes")
	public DefaultEdge createEdge(DefaultGraphCell v1, DefaultGraphCell v2, Map<DefaultGraphCell, Map> attributes, ConnectionSet cs, ArrayList<Point> points) {

        // Create Edge
        DefaultEdge edge = new DefaultEdge();

        // Create Edge Attributes
        Map edgeAttrib = new Hashtable();
        attributes.put(edge, edgeAttrib);

        if (points != null) {
            GraphConstants.setPoints(edgeAttrib, points);
            GraphConstants.setLineStyle(edgeAttrib, GraphConstants.STYLE_SPLINE);

        }
        // Set Arrow
        GraphConstants.setLineEnd(edgeAttrib, GraphConstants.ARROW_CLASSIC);
        GraphConstants.setEndFill(edgeAttrib, true);

        //        GraphConstants.setEditable(edgeAttrib, false);
        //        GraphConstants.setMoveable(edgeAttrib, false);
        GraphConstants.setDisconnectable(edgeAttrib, false);
        //        GraphConstants.setConnectable(edgeAttrib, false);
        //        GraphConstants.setResize(edgeAttrib, false);

        //        GraphConstants.setLineColor(edgeAttrib, Color.blue);

        if (v1 == v2) {
            // add a new port so that it does go back to the same point
            DefaultPort hp = new DefaultPort();
            v1.add(hp);
            cs.connect(edge, v1.getChildAt(0), v1.getChildAt(1));

        } else {
            cs.connect(edge, v1.getChildAt(0), v2.getChildAt(0));
        }
        return edge;
    }

    private void displaySliders(boolean displaySliders) {
        Container contentPane = this.getContentPane();
        GridBagConstraints gbc = new GridBagConstraints();
        if (displaySliders) {
            contentPane.remove(jGraphPane);

            gbc.fill = GridBagConstraints.NONE;
            gbc.anchor = GridBagConstraints.EAST;
            gbc.weightx = 0.10;
            gbc.weighty = 0.01;
            addCompItem(boxWidthLabel, gbc, 0, 1, 1, 1);

            gbc.fill = GridBagConstraints.HORIZONTAL;
            gbc.anchor = GridBagConstraints.WEST;
            gbc.weightx = 0.70;
            gbc.weighty = 0.01;
            addCompItem(boxWidthSlider, gbc, 1, 1, 1, 1);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1.0;
            gbc.weighty = 0.99;
            addCompItem(jGraphPane, gbc, 0, 2, 2, 1);
        } else {
            contentPane.remove(boxWidthLabel);
            contentPane.remove(boxWidthSlider);
            contentPane.remove(jGraphPane);

            gbc.fill = GridBagConstraints.BOTH;
            gbc.anchor = GridBagConstraints.CENTER;
            gbc.weightx = 1;
            gbc.weighty = 1;
            addCompItem(jGraphPane, gbc, 0, 1, 1, 1);
        }

        //Now call validate so that these component changes are displayed.
        validate();
    }

    @SuppressWarnings("unchecked")
	public Edge getEdge(FunctionProfile p, FunctionProfile c) {

        Vertex parent = vertexMap.get(p);
        Vertex child = vertexMap.get(c);

        int portCount = child.getGraphCell().getChildCount();
        for (int j = 0; j < portCount; j++) {
            Port port = (Port) child.getGraphCell().getChildAt(j);

            for (Iterator<Edge> itrEdges = port.edges(); itrEdges.hasNext();) {
                Edge edge = itrEdges.next();

                if (edge.getTarget() == port) {
                    Port sourcePort = (Port) edge.getSource();
                    Object sourceVertex = model.getParent(sourcePort);

                    CellView sourceVertexView = graph.getGraphLayoutCache().getMapping(sourceVertex, false);

                    GraphCell target = (GraphCell) sourceVertexView.getCell();
                    if (target.getVertex() == parent) {
                        return (Edge) edge;
                    }
                }
            }
        }

        return null;
    }

    private void handlePrefEvent() {

        font = new Font(ppTrial.getPreferencesWindow().getFontName(), ppTrial.getPreferencesWindow().getFontStyle(),
                ppTrial.getPreferencesWindow().getFontSize());

        this.setFont(font);
        FontMetrics fm = getFontMetrics(font);

        boxHeight = fm.getHeight() + 5;

        recreateGraph();
    }

    @SuppressWarnings("rawtypes")
	public void handleColorEvent() {
        Map<Object, Map> attributeMap = new Hashtable<Object, Map>();

        // color all edges black and reset pathHighlight to false
        for (int i = 0; i < graphCellList.size(); i++) {
            GraphCell dgc = graphCellList.get(i);
            Vertex v = dgc.getVertex();
            v.setPathHighlight(false);

            int portCount = model.getChildCount(dgc);
            for (int j = 0; j < portCount; j++) {
                Object port = model.getChild(dgc, j);

                for (Iterator itrEdges = model.edges(port); itrEdges.hasNext();) {
                    Object edge = itrEdges.next();

                    Map attrib = new HashMap();

                    GraphConstants.setLineColor(attrib, Color.black);

                    attributeMap.put(edge, attrib);

                }
            }
        }

        // run through each vertex and find the "highlighted" one
        for (int i = 0; i < graphCellList.size(); i++) {
            GraphCell dgc = graphCellList.get(i);

            if (dgc.function == ppTrial.getHighlightedFunction()) { // this is the one

                // now iterate through each function and check for callpaths that contain it, highlight those edges and vertices
                for (int j = 0; j < functionProfileList.size(); j++) {
                    FunctionProfile fp = functionProfileList.get(j);
                    if (fp == null)
                        continue;
                    Function f = fp.getFunction();

                    if (f.isCallPathFunction()) { // we only care about call path functionProfiles
                        String s = f.getName();
                        if (s.indexOf(dgc.getFunction().getName()) != -1) {

                            int location = s.indexOf("=>");

                            // now iterate through every edge in this callpath
                            while (location != -1) {
                                String parentString = s.substring(0, location);

                                int next = s.indexOf("=>", location + 1);

                                if (next == -1) {
                                    next = s.length();
                                }

                                String childString = s.substring(location + 2, next);

                                //System.out.println(parentString + "=>" + childString);

                                FunctionProfile parentFunction = thread.getFunctionProfile(ppTrial.getDataSource().getFunction(
                                        parentString));
                                FunctionProfile childFunction = thread.getFunctionProfile(ppTrial.getDataSource().getFunction(
                                        childString));

                                Vertex v = vertexMap.get(parentFunction);
                                v.setPathHighlight(true);
                                v = vertexMap.get(childFunction);
                                v.setPathHighlight(true);

                                Edge e = getEdge(parentFunction, childFunction);

                                Map attrib = new HashMap();

                                GraphConstants.setLineColor(attrib, Color.blue);

                                if (e == null) {
                                    //System.out.println("uh oh");
                                } else {
                                    attributeMap.put(e, attrib);
                                }
                                s = s.substring(location + 3);
                                location = s.indexOf("=>");
                            }

                        }
                    }
                }
            }
        }

        // now do the final coloring
        for (int i = 0; i < graphCellList.size(); i++) {

            GraphCell dgc = graphCellList.get(i);
            Map attrib = new HashMap();

            if (dgc.function == ppTrial.getHighlightedFunction()) {
                GraphConstants.setBorder(attrib, BorderFactory.createBevelBorder(BevelBorder.RAISED,
                        ppTrial.getColorChooser().getHighlightColor(), ppTrial.getColorChooser().getHighlightColor()));
            } else if (dgc.getVertex().getPathHighlight()) {
                GraphConstants.setBorder(attrib, BorderFactory.createBevelBorder(BevelBorder.RAISED, Color.blue, Color.blue));

            } else if (dgc.function.isGroupMember(ppTrial.getHighlightedGroup())) {
                GraphConstants.setBorder(attrib, BorderFactory.createBevelBorder(BevelBorder.RAISED,
                        ppTrial.getColorChooser().getGroupHighlightColor(), ppTrial.getColorChooser().getGroupHighlightColor()));
            } else {
                GraphConstants.setBorder(attrib, BorderFactory.createRaisedBevelBorder());
            }

            attributeMap.put(dgc, attrib);
        }

        graph.getGraphLayoutCache().edit(attributeMap, null, null, null);
    }

    public void update(Observable o, Object arg) {
        String tmpString = (String) arg;

        if (tmpString.equals("subWindowCloseEvent")) {
            closeThisWindow();
        } else if (tmpString.equals("prefEvent")) {
            handlePrefEvent();
        } else if (tmpString.equals("colorEvent")) {
            handleColorEvent();
        } else if (tmpString.equals("dataEvent")) {
            setupMenus();
            this.validate();
            recreateGraph();
        }

    }

    public void help(boolean display) {
        //Show the ParaProf help window.
        ParaProf.getHelpWindow().clearText();
        if (display) {
            ParaProf.getHelpWindow().setVisible(true);
        }
        ParaProf.getHelpWindow().writeText("This is the Call Graph Window");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("This window shows you a graph of call paths present in the profile data.");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("Click on a box to highlight paths that go through that function.");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText("Right-click on a box to access the Function Data Window for that function.");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText(
                "Experiment with the \"Box Width by...\" and \"Box Color by...\" menus (under Options) to display different types of data.");
        ParaProf.getHelpWindow().writeText("");
        ParaProf.getHelpWindow().writeText(
                "If you only see a single line of boxes (no edges connecting them), it probably means that your profile data does not contain call path data.  If you believe this to be incorrect please contact us with the data at tau-bugs@cs.uoregon.edu");
        ParaProf.getHelpWindow().writeText("");
    }

    public Dimension getViewportSize() {
        return jGraphPane.getViewport().getExtentSize();
    }

    public Rectangle getViewRect() {
        return jGraphPane.getViewport().getViewRect();
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

    //Respond correctly when this window is closed.
    void thisWindowClosing(java.awt.event.WindowEvent e) {
        closeThisWindow();
    }

    public void closeThisWindow() {
        setVisible(false);
        ppTrial.deleteObserver(this);
        ParaProf.decrementNumWindows();
        dispose();
    }

    @SuppressWarnings("rawtypes")
	public GraphCell getGraphCellForLocation(int x, int y) {
        for (int i = 0; i < graphCellList.size(); i++) {
            GraphCell gc = graphCellList.get(i);

            Map attrib = gc.getAttributes();
            Rectangle2D bounds = GraphConstants.getBounds(attrib);

            if (bounds.contains(x, y))
                return gc;
        }
        return null;
    }

    // listener for the boxWidthSlider
    public void stateChanged(ChangeEvent event) {
        try {
            boxWidth = boxWidthSlider.getValue();
            recreateGraph();
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void mouseWheelMoved(MouseWheelEvent event) {
	// numClicks is negative if scrolling up
	int numClicks = 0;
	numClicks = event.getWheelRotation();
	scale = scale + 0.10*numClicks;
	if (scale > 5.0) {
	    scale = 5.0;
	}
	if (scale < 0.10) {
	    scale = 0.10;
	}
	graph.setScale(scale);
    }

    public void keyTyped(KeyEvent evt) {
        try {
            char key = evt.getKeyChar();
            // zoom in and out on +/-
            if (key == '+' || key == '=') {
                scale = scale + 0.10;
                if (scale > 5.0)
                    scale = 5.0;
                graph.setScale(scale);
            } else if (key == '-' || key == '_') {
                scale = scale - 0.10;
                if (scale < 0.10)
                    scale = 0.10;
                graph.setScale(scale);
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }

    }

    public void keyPressed(KeyEvent evt) {

    }

    public void keyReleased(KeyEvent evt) {

    }

    //ParaProfImageInterface
    public Dimension getImageSize(boolean fullScreen, boolean prependHeader) {
        if (fullScreen)
            return this.getPreferredSize();
        else
            return this.getSize();
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {

        if (toScreen == false) {
            graph.setDoubleBuffered(false);
        }
        if (fullWindow) {
            graph.paintAll(g2D);
        } else {
            graph.paint(g2D);
        }

        if (toScreen == false) {
            graph.setDoubleBuffered(true);
        }

    }

    public int print(Graphics g, PageFormat pageFormat, int page) {
        double oldScale = graph.getScale();
        try {
            // turn off double buffering of graph so we don't get bitmap printed
            graph.setDoubleBuffered(false);

            if (page >= 1) {
                return NO_SUCH_PAGE;
            }

            ParaProfUtils.scaleForPrint(g, pageFormat, graph.getWidth(), graph.getHeight());

            graph.paint(g);
        } finally {
            //  turn double buffering back on
            graph.setDoubleBuffered(true);
            graph.setScale(oldScale);
        }
        return PAGE_EXISTS;
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();
            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Show Width Slider")) {
                    if (slidersCheckBox.isSelected()) {
                        displaySliders(true);
                    } else {
                        displaySliders(false);
                    }
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public JFrame getFrame() {
        return this;
    }

}