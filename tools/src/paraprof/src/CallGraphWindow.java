package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.border.BevelBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.jgraph.JGraph;
import org.jgraph.graph.*;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.paraprof.enums.CallGraphOption;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfdmf.Thread;

/**
 * CallGraphWindow.java
 * This window displays the callpath data as a graph.
 *   
 * TODO: Infinite.  The 2nd half of Sugiyama's algorithm should probably
 *       be implemented.  Plenty of other things could be done as well, such
 *       as using box height as another metric.
 *       
 * <P>CVS $Id: CallGraphWindow.java,v 1.12 2007/05/17 17:18:42 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.12 $
 */
public class CallGraphWindow extends JFrame implements ActionListener, KeyListener, ChangeListener, Observer, ImageExport,
        Printable, ParaProfWindow {

    private static final int MARGIN = 20;
    private static final int HORIZONTAL_SPACING = 10;
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

    private List functionProfileList;
    private DefaultGraphModel model;
    private List graphCellList;
    private Object[] cells;
    private List levels;
    private List backEdges;
    private Map vertexMap;

    private int widthMetricID;
    private int colorMetricID;
    private Font font;
    private int boxHeight;
    private double scale = 1.0;

    private static class GraphSelectionModel extends DefaultGraphSelectionModel {

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
                ArrayList result = new ArrayList();
                // Roots Are Always Selectable
                Stack s = new Stack();
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

    private class GraphCell extends DefaultGraphCell {

        private final Function function;
        private final FunctionProfile functionProfile;
        private final Vertex vertex;

        public GraphCell(Vertex v) {
            super(((FunctionProfile)v.getUserObject()).getFunction());
            functionProfile = (FunctionProfile)v.getUserObject();
            function = functionProfile.getFunction();
            vertex = v;
        }

        public String getToolTipString() {
            String result = "<html>" + function;

            if (widthOption != CallGraphOption.STATIC && widthOption != CallGraphOption.NAME_LENGTH) {
                float widthValue = (float) getValue(functionProfile, widthOption, 1.0, widthMetricID);
                result = result + "<br>Width Value (" + widthOption;
                if (widthOption != CallGraphOption.NUMCALLS && widthOption != CallGraphOption.NUMSUBR) {
                    result = result + ", " + ppTrial.getMetricName(widthMetricID);
                }
                result = result + ") : " + widthValue;
            }

            if (colorOption != CallGraphOption.STATIC) {
                float colorValue = (float) getValue(functionProfile, colorOption, 1.0, colorMetricID);
                result = result + "<br>Color Value (" + colorOption;
                if (colorOption != CallGraphOption.NUMCALLS && colorOption != CallGraphOption.NUMSUBR) {
                    result = result + ", " + ppTrial.getMetricName(colorMetricID);
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

    // A simple structure to hold pairs of vertices
    private static class BackEdge {
        public BackEdge(Vertex a, Vertex b) {
            this.a = a;
            this.b = b;
        }

        private Vertex a, b;
    }

    // Warning: this class violates lots of OO principles, I'm using it as a struct.
    private static class Vertex implements Comparable {
        private List children = new ArrayList();
        private List parents = new ArrayList();
        private Object userObject;
        private boolean visited;

        private int downPriority;
        private int upPriority;

        private int level = -1; // which level this vertex resides on
        private int levelIndex; // the index within the level

        private double baryCenter;
        private double gridBaryCenter;

        private GraphCell graphCell;
        private int position = -1;
        private int width;
        private int height;
        private float colorRatio;

        private boolean pathHighlight = false;

        public Object getUserObject() {
            return userObject;
        }

        public Vertex(Object userObject, int width, int height) {
            this.userObject = userObject;

            this.width = width;
            this.height = height;

            if (userObject != null && width < 5) {
                this.width = 5;
            }
        }

        public int compareTo(Object compare) {
            if (this.baryCenter < ((Vertex) compare).baryCenter)
                return -1;
            if (this.baryCenter > ((Vertex) compare).baryCenter)
                return 1;
            return 0;
        }

        private int getPriority(boolean down) {
            if (down) {
                return downPriority;
            } else {
                return upPriority;
            }
        }

        public void setPathHighlight(boolean pathHighlight) {
            this.pathHighlight = pathHighlight;
        }

        public boolean getPathHighlight() {
            return pathHighlight;
        }

    }

    public CallGraphWindow(ParaProfTrial ppTrial, Thread thread, Component invoker) {
        this.ppTrial = ppTrial;
        ppTrial.addObserver(this);
        this.colorMetricID = ppTrial.getDefaultMetricID();
        this.widthMetricID = ppTrial.getDefaultMetricID();

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
            for (int i = 0; i < ppTrial.getNumberOfMetrics(); i++) {

                if (i == this.widthMetricID && enabled) {
                    button = new JRadioButtonMenuItem(ppTrial.getMetric(i).getName(), true);
                } else {
                    button = new JRadioButtonMenuItem(ppTrial.getMetric(i).getName());
                }
                final int m = i;

                button.addActionListener(new ActionListener() {
                    final int metric = m;

                    public void actionPerformed(ActionEvent evt) {
                        widthOption = option;
                        widthMetricID = metric;
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
            for (int i = 0; i < ppTrial.getNumberOfMetrics(); i++) {

                if (i == this.widthMetricID && enabled) {
                    button = new JRadioButtonMenuItem(ppTrial.getMetric(i).getName(), true);
                } else {
                    button = new JRadioButtonMenuItem(ppTrial.getMetric(i).getName());
                }
                final int m = i;

                button.addActionListener(new ActionListener() {
                    final int metric = m;

                    public void actionPerformed(ActionEvent evt) {
                        colorOption = option;
                        colorMetricID = metric;
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
            width = (int) (boxWidth * getValue(fp, this.widthOption, maxValue, widthMetricID));
        }
        return width;
    }

    private void createGraph() {

        vertexMap = new HashMap();
        backEdges = new ArrayList();

        double maxWidthValue = getMaxValue(this.widthOption, widthMetricID);
        double maxColorValue = getMaxValue(this.colorOption, colorMetricID);

        for (int i = 0; i < functionProfileList.size(); i++) {
            FunctionProfile fp = (FunctionProfile) functionProfileList.get(i);
            if (fp == null) // skip it if this thread didn't call this function
                continue;

            if (!fp.isCallPathFunction()) { // skip callpath functions (we only want the actual functions)

                Vertex v = new Vertex(fp, getWidth(fp, maxWidthValue), boxHeight);
                v.colorRatio = (float) getValue(fp, this.colorOption, maxColorValue, colorMetricID);
                vertexMap.put(fp, v);
            }
        }

        // now we follow the call paths and eliminate back edges
        Stack toVisit = new Stack();
        Stack currentPath = new Stack();

        for (int i = 0; i < functionProfileList.size(); i++) {
            FunctionProfile fp = (FunctionProfile) functionProfileList.get(i);
            if (fp == null) // skip it if this thread didn't call this function
                continue;

            if (!fp.isCallPathFunction()) { // skip callpath functions (we only want the actual functions)

                // get the vertex for this FunctionProfile 
                Vertex root = (Vertex) vertexMap.get(fp);

                if (!root.visited) {

                    currentPath.add(fp);
                    toVisit.add(null); // null in the toVisit stack marks the end of a set of children (they must get pushed into the stack prior to the children)

                    // add all the children to the toVisit list
                    for (Iterator it = fp.getChildProfiles(); it.hasNext();) {
                        FunctionProfile childFp = (FunctionProfile) it.next();
                        toVisit.add(childFp);
                    }

                    while (!toVisit.empty()) {
                        FunctionProfile childFp = (FunctionProfile) toVisit.pop();

                        if (childFp == null) {
                            // this marks the end of a set of children, so pop the current path
                            // and move on to the next one in toVisit
                            currentPath.pop();
                            continue;
                        }

                        Vertex child = (Vertex) vertexMap.get(childFp);
                        FunctionProfile parentFp = (FunctionProfile) currentPath.peek();

                        Vertex parent = (Vertex) vertexMap.get(parentFp);

                        // run through the currentPath and see if childFp is in it, if so, this is a backedge
                        boolean back = false;
                        for (Iterator it = currentPath.iterator(); it.hasNext();) {
                            if ((FunctionProfile) it.next() == childFp) {
                                back = true;
                                break;
                            }
                        }

                        if (back) {
                            backEdges.add(new BackEdge(parent, child));
                        } else {

                            boolean found = false;
                            for (int j = 0; j < parent.children.size(); j++) {
                                if (parent.children.get(j) == child)
                                    found = true;
                            }
                            if (!found)
                                parent.children.add(child);

                            found = false;
                            for (int j = 0; j < child.parents.size(); j++) {
                                if (child.parents.get(j) == parent)
                                    found = true;
                            }
                            if (!found)
                                child.parents.add(parent);

                            if (child.visited == false) {

                                child.visited = true;

                                currentPath.add(childFp);

                                toVisit.add(null);
                                for (Iterator it = childFp.getChildProfiles(); it.hasNext();) {
                                    FunctionProfile grandChildFunction = (FunctionProfile) it.next();

                                    Vertex grandChild = (Vertex) vertexMap.get(grandChildFunction);
                                    //  if (grandChild.visited == false)
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
        List roots = findRoots(vertexMap);

        // Assigning Levels
        for (int i = 0; i < functionProfileList.size(); i++) {
            FunctionProfile fp = (FunctionProfile) functionProfileList.get(i);
            if (fp == null)
                continue;

            if (!fp.isCallPathFunction()) {

                Vertex vertex = (Vertex) vertexMap.get(fp);

                if (vertex.level == -1) {
                    assignLevel(vertex);
                }
            }

        }

        // Insert Dummies
        for (int i = 0; i < functionProfileList.size(); i++) {
            FunctionProfile fp = (FunctionProfile) functionProfileList.get(i);
            if (fp == null)
                continue;

            if (!fp.isCallPathFunction()) {

                Vertex vertex = (Vertex) vertexMap.get(fp);

                insertDummies(vertex);
            }

        }

        // fill level lists
        for (int i = 0; i < functionProfileList.size(); i++) {
            FunctionProfile fp = (FunctionProfile) functionProfileList.get(i);
            if (fp == null)
                continue;

            if (!fp.isCallPathFunction()) {

                Vertex vertex = (Vertex) vertexMap.get(fp);
                vertex.visited = false;
            }

        }
        levels = new ArrayList();

        // Fill Levels

        for (int i = 0; i < roots.size(); i++) {
            Vertex root = (Vertex) roots.get(i);
            fillLevels(root, levels, 0);
        }

        // Order Levels

        runSugiyama(levels);
        assignPositions(levels);

        // Draw Graph

        // Construct Model and Graph
        model = new DefaultGraphModel();
        graph = new Graph(model, this);
        graph.addMouseListener(graph);
        graph.addKeyListener(this);

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
            DefaultGraphCell dgc = (DefaultGraphCell) graphCellList.get(i);
            dgc.removeAllChildren();
        }

        model.remove(cells);

        reassignWidths(levels);
        assignPositions(levels);

        createCustomGraph(levels, backEdges);

    }

    void reassignWidths(List levels) {

        double maxWidthValue = getMaxValue(this.widthOption, widthMetricID);
        double maxColorValue = getMaxValue(this.colorOption, colorMetricID);

        for (int i = 0; i < levels.size(); i++) {
            List level = (List) levels.get(i);

            for (int j = 0; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);

                if (v.getUserObject() != null) {
                    FunctionProfile fp = (FunctionProfile) v.getUserObject();

                    v.width = getWidth(fp, maxWidthValue);

                    // we have to do this check here since we're treating it like a struct
                    if (v.width < 5)
                        v.width = 5;

                    v.colorRatio = (float) getValue(fp, this.colorOption, maxColorValue, colorMetricID);

                    v.height = boxHeight;
                }
            }
        }
    }

    void createCustomGraph(List levels, List backEdges) {

        Map attributes = new HashMap();

        graphCellList = new ArrayList();
        List cellList = new ArrayList();

        for (int i = 0; i < levels.size(); i++) {
            List level = (List) levels.get(i);

            for (int j = 0; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);

                GraphCell dgc = null;

                if (v.getUserObject() != null) {
                    dgc = createGraphCell(v, v.position - (v.width / 2), MARGIN + i * VERTICAL_SPACING, v.height, v.width,
                            v.colorRatio, attributes);

                    v.graphCell = dgc;
                    cellList.add(dgc);
                    graphCellList.add(dgc);
                } else {
                    // dummy node, don't make a graph cell
                }

            }
        }

        ConnectionSet cs = new ConnectionSet();
        List edgeList = new ArrayList();

        for (int i = 0; i < levels.size(); i++) {
            List level = (List) levels.get(i);

            for (int j = 0; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);

                if (v.getUserObject() != null) {
                    GraphCell dgcParent = v.graphCell;

                    for (Iterator it = v.children.iterator(); it.hasNext();) {
                        Vertex child = (Vertex) it.next();

                        if (child.getUserObject() != null) {
                            // simply connect the GraphCells
                            GraphCell dgcChild = child.graphCell;

                            DefaultEdge e = createEdge(dgcParent, dgcChild, attributes, cs, null);
                            cellList.add(e);
                            edgeList.add(e);
                        } else {
                            // follow the chain of vertices whose functions are null to find the real
                            // child vertex.  All of these inbetween vertices are "dummy nodes"
                            ArrayList points = new ArrayList();
                            int l = 1; // how many levels down this dummy node is

                            points.add(new Point(3000, 3000)); // this point's position doesn't matter because of the connect call

                            while (child.getUserObject() == null) {
                                points.add(new Point(child.position, MARGIN + ((i + l) * VERTICAL_SPACING) + (boxHeight / 2)));
                                // find the end of the dummy chain
                                child = (Vertex) child.children.get(0); // there can only be exactly one child
                                l++;
                            }

                            points.add(new Point(3000, 3000)); // this point's position doesn't matter because of the connect call

                            DefaultEdge e = createEdge(dgcParent, child.graphCell, attributes, cs, points);
                            cellList.add(e);
                            edgeList.add(e);
                        }
                    }
                }
            }
        }

        // Now create the back edges
        for (int i = 0; i < backEdges.size(); i++) {
            BackEdge backEdge = (BackEdge) backEdges.get(i);

            ArrayList points = new ArrayList();

            // this point's position doesn't matter because of the connect call
            points.add(new Point(3000, 3000));

            points.add(new Point(backEdge.a.position + backEdge.a.width / 2 + 50, (backEdge.a.level) * VERTICAL_SPACING + MARGIN
                    + (boxHeight / 2)));

            points.add(new Point(backEdge.b.position + 25, (backEdge.b.level) * VERTICAL_SPACING - 25 + MARGIN));

            // this point's position doesn't matter because of the connect call
            points.add(new Point(3000, 3000));

            DefaultEdge edge = createEdge(backEdge.a.graphCell, backEdge.b.graphCell, attributes, cs, points);
            cellList.add(edge);
            edgeList.add(edge);

        }

        cells = cellList.toArray();

        model.insert(cells, attributes, cs, null, null);

        // now make sure everything is visible (this fixes big edges that go off the top of the screen
        moveDownToVisible(cellList, edgeList);
    }

    private void moveDownToVisible(List cellList, List edgeList) {
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
            Map attributeMap = new HashMap();

            for (int i = 0; i < cellList.size(); i++) {
                DefaultGraphCell dgc = (DefaultGraphCell) cellList.get(i);
                Map attrib = dgc.getAttributes();
                translate(attrib, 0, -minY);
                attributeMap.put(dgc, attrib);
            }
            graph.getGraphLayoutCache().edit(attributeMap, null, null, null);
        }
    }

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
            java.util.List points = GraphConstants.getPoints(map);
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

    void runPhaseOne(List levels) {

        int numIterations = 100;

        while (numIterations > 0) {
            for (int i = 0; i < levels.size() - 1; i++) {
                assignBaryCenters((List) levels.get(i), (List) levels.get(i + 1), true);
                Collections.sort((List) levels.get(i));
            }

            for (int i = levels.size() - 1; i > 0; i--) {
                assignBaryCenters((List) levels.get(i), (List) levels.get(i - 1), false);
                Collections.sort((List) levels.get(i));
            }

            numIterations--;
        }

        //Vertex.BaryCenter = Vertex.BARYCENTER_DOWN;

    }

    void assignBaryCenters(List level, List level2, boolean down) {

        for (int j = 0; j < level2.size(); j++) {
            Vertex v = (Vertex) level2.get(j);
            v.levelIndex = j;
        }

        for (int i = 0; i < level.size(); i++) {
            Vertex v = (Vertex) level.get(i);
            if (down) {
                int sum = 0;
                for (int j = 0; j < v.children.size(); j++) {
                    sum += ((Vertex) v.children.get(j)).levelIndex;
                }

                // don't re-assign baryCenter if no children (keep old value, based on parents)
                if (v.children.size() != 0) {
                    v.baryCenter = sum / v.children.size();
                }

            } else {
                int sum = 0;
                for (int j = 0; j < v.parents.size(); j++) {
                    sum += ((Vertex) v.parents.get(j)).levelIndex;
                }

                // don't re-assign baryCenter if no parents (keep old value, based on children)
                if (v.parents.size() != 0) {
                    v.baryCenter = sum / v.parents.size();
                }

            }
        }

    }

    void assignGridBaryCenters(List level, boolean down, boolean finalPass) {

        //        boolean combined = false;
        //
        //        if (!combined) {
        for (int i = 0; i < level.size(); i++) {
            Vertex v = (Vertex) level.get(i);

            //if (finalPass && v.children.size() == 0)
            //    down = false;

            if (down) {
                // don't re-assign baryCenter if no children (keep old value, based on parent)
                if (v.children.size() == 0)
                    continue;

                float sum = 0;
                for (int j = 0; j < v.children.size(); j++) {
                    sum += ((Vertex) v.children.get(j)).position;
                }

                v.gridBaryCenter = sum / v.children.size();

            } else {
                // don't re-assign baryCenter if no parents (keep old value, based on children)
                if (v.parents.size() == 0)
                    continue;

                float sum = 0;
                for (int j = 0; j < v.parents.size(); j++) {
                    sum += ((Vertex) v.parents.get(j)).position;
                }

                v.gridBaryCenter = sum / v.parents.size();

            }
        }
        //        } else {
        //            for (int i = 0; i < level.size(); i++) {
        //                Vertex v = (Vertex) level.get(i);
        //
        //                float sum = 0;
        //                for (int j = 0; j < v.children.size(); j++) {
        //                    sum += ((Vertex) v.children.get(j)).position;
        //                }
        //
        //                for (int j = 0; j < v.parents.size(); j++) {
        //                    sum += ((Vertex) v.parents.get(j)).position;
        //                }
        //
        //                v.gridBaryCenter = sum / (v.parents.size() + v.children.size());
        //
        //            }
        //        }
    }

    void runSugiyama(List levels) {

        runPhaseOne(levels);

        //runPhaseTwo(levels);

    }

    private void assignPositions(List levels) {

        // assign initial positions
        for (int i = 0; i < levels.size(); i++) {
            List level = (List) levels.get(i);

            int lastPosition = 0;
            ((Vertex) level.get(0)).position = 0;
            for (int j = 1; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);
                v.position = lastPosition + HORIZONTAL_SPACING + (((Vertex) level.get(j - 1)).width + v.width) / 2;
                lastPosition = v.position;

                v.downPriority = v.children.size();
                if (v.getUserObject() == null) {
                    //v.downPriority = Integer.MAX_VALUE;
                    v.downPriority = 2;
                }

                v.upPriority = v.parents.size();
                if (v.getUserObject() == null) {
                    //v.upPriority = Integer.MAX_VALUE;
                    v.upPriority = 2;
                }
            }

            // now center everything around zero
            int middle;

            if (level.size() % 2 == 0) {
                int left = ((Vertex) level.get((level.size() - 2) / 2)).position;
                int right = ((Vertex) level.get(level.size() / 2)).position;
                middle = (left + right) / 2;
            } else {
                middle = ((Vertex) level.get((level.size() - 1) / 2)).position;
            }

            for (int j = 0; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);
                v.position = v.position - middle;
            }

        }

        for (int i = 1; i < levels.size(); i++) {
            improvePositions(levels, i, false, false);
        }

        for (int i = levels.size() - 2; i >= 0; i--) {
            improvePositions(levels, i, true, false);
        }

        for (int i = 1; i < levels.size(); i++) {
            improvePositions(levels, i, false, false);
        }

        for (int i = levels.size() - 2; i >= 0; i--) {
            improvePositions(levels, i, true, true);
        }

        for (int i = levels.size() - 2; i >= 0; i--) {
            improvePositions(levels, i, true, false);
        }

        // move everything right (since some of our numbers are negative)

        int minValue = 0;
        for (int i = 0; i < levels.size(); i++) {
            List level = (List) levels.get(i);

            for (int j = 0; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);
                if (v.position - (v.width / 2) < minValue) {
                    minValue = v.position - (v.width / 2);
                }
            }
        }

        for (int i = 0; i < levels.size(); i++) {
            List level = (List) levels.get(i);

            for (int j = 0; j < level.size(); j++) {
                Vertex v = (Vertex) level.get(j);
                v.position += -minValue + MARGIN;
            }
        }

    }

    private int moveRight(List level, int index, int amount, boolean down, int priority) {

        Vertex v = (Vertex) level.get(index);

        int j = index + 1;

        if (j >= level.size()) {
            v.position = v.position + amount;
            return amount;
        }

        Vertex u = (Vertex) level.get(j);

        int myRightSide = v.position + (v.width / 2);
        int neighborLeftSide = u.position - (u.width / 2);

        if (myRightSide + amount + HORIZONTAL_SPACING < neighborLeftSide) {
            v.position = v.position + amount;
            return amount;
        }

        // not enough room between this box and the one to the right

        if (u.getPriority(down) > priority) {
            // we're lower priority, can't move him, place ourselves as far right as possible
            int newPosition = u.position - ((v.width + u.width) / 2) - HORIZONTAL_SPACING;
            int amountMoved = newPosition - v.position;
            v.position = v.position + amountMoved;
            return amountMoved;
        }

        int positionNeighborNeedsToBeAt = (v.position + amount) + ((u.width + v.width) / 2) + HORIZONTAL_SPACING;

        // we can move him, so ask to move '' and add whatever he can (he could be blocked by higher priority)
        moveRight(level, j, positionNeighborNeedsToBeAt - u.position, down, priority);

        int newPosition = u.position - ((v.width + u.width) / 2) - HORIZONTAL_SPACING;
        int amountMoved = newPosition - v.position;
        v.position = v.position + amountMoved;
        return amountMoved;

        //v.position = v.position + amountMoved;
        //return amountMoved;
    }

    private int moveLeft(List level, int index, int amount, boolean down, int priority) {
        //System.out.println("index " + index + ", asked to move " + amount + ", priority = "
        //        + priority);
        Vertex v = (Vertex) level.get(index);

        int j = index - 1;

        if (j < 0) {
            v.position = v.position - amount;
            // System.out.println("index " + index + ", moved " + amount + ", to position "
            //         + v.position);
            return amount;
        }

        Vertex u = (Vertex) level.get(j);

        int myLeftSide = v.position - (v.width / 2);
        int neighborRightSide = u.position + (u.width / 2);

        if (myLeftSide - amount - HORIZONTAL_SPACING > neighborRightSide) {
            v.position = v.position - amount;
            //System.out.println("index " + index + ", moved " + amount + ", to position "
            //       + v.position);
            return amount;
        }

        if (u.getPriority(down) > priority) {
            int newPosition = u.position + ((u.width + v.width) / 2) + HORIZONTAL_SPACING;
            int amountMoved = v.position - newPosition;
            v.position = v.position - amountMoved;
            //System.out.println("index " + index + ", moved " + amountMoved + ", to position "
            //        + v.position);
            return amountMoved;
        }

        int positionNeighborNeedsToBeAt = (v.position - amount) - (v.width / 2) - HORIZONTAL_SPACING - (u.width / 2);

        // we can move him, so ask to move '' and add whatever he can (he could be blocked by higher priority)

        moveLeft(level, j, u.position - positionNeighborNeedsToBeAt, down, priority);

        int newPosition = u.position + ((u.width + v.width) / 2) + HORIZONTAL_SPACING;
        int amountMoved = v.position - newPosition;
        v.position = v.position - amountMoved;
        //System.out.println("index " + index + ", moved " + amountMoved + ", to position "
        //        + v.position);
        return amountMoved;

        //        v.position = v.position - amountMoved;
        //        System.out.println ("index " + index + ", moved " + amountMoved + ", to position " + v.position);
        //        return amountMoved;
    }

    private void improvePositions(List levels, int index, boolean down, boolean finalPass) {
        List level = (List) levels.get(index);

        assignGridBaryCenters(level, down, finalPass);

        for (int i = 0; i < level.size(); i++) {
            Vertex v = (Vertex) level.get(i);

            int wantedPosition = (int) v.gridBaryCenter;
            //System.out.println("--at position: " + v.position + ", want Position = "
            //        + wantedPosition);

            if (down && v.children.size() == 0) {
                continue;
            }

            if (wantedPosition > v.position) {
                int amountMoved = moveRight(level, i, wantedPosition - v.position, down, v.getPriority(down));
                for (int j = i - 1; j >= 0 && finalPass; j--) {
                    moveRight(level, j, amountMoved, down, v.getPriority(down));
                }

            } else {
                int amountMoved = moveLeft(level, i, v.position - wantedPosition, down, v.getPriority(down));
                for (int j = i + 1; j < level.size() && finalPass; j++) {
                    moveLeft(level, j, amountMoved, down, v.getPriority(down));
                }
            }

            //System.out.println("--got position = " + v.position + "\n");

        }

    }

    private void fillLevels(Vertex v, List levels, int level) {

        if (v.visited == true)
            return;

        v.visited = true;

        if (levels.size() == level) {
            levels.add(new ArrayList());
        }

        v.level = level;

        List currentLevel = (List) levels.get(level);
        currentLevel.add(v);

        for (int i = 0; i < v.children.size(); i++) {
            Vertex child = (Vertex) v.children.get(i);
            fillLevels(child, levels, level + 1);
        }
    }

    private void insertDummies(Vertex v) {

        for (int i = 0; i < v.children.size(); i++) {
            Vertex child = (Vertex) v.children.get(i);
            if (child.level - v.level > 1) {

                // break both edges
                v.children.remove(i);
                child.parents.remove(v);

                // create dummy and connect to child
                Vertex dummy = new Vertex(null, 1, boxHeight);
                dummy.level = v.level + 1;
                dummy.children.add(child);
                child.parents.add(dummy);

                // connect dummy to parrent
                v.children.add(i, dummy);
                dummy.parents.add(v);
                insertDummies(dummy);
            }
        }
    }

    private void assignLevel(Vertex v) {

        int maxLevel = 0;
        for (int i = 0; i < v.parents.size(); i++) {
            Vertex parent = (Vertex) v.parents.get(i);
            if (parent.level == -1)
                assignLevel(parent);
            if (parent.level > maxLevel)
                maxLevel = parent.level;
        }
        v.level = maxLevel + 1;
    }

    private List findRoots(Map vertexMap) {
        List roots = new ArrayList();

        for (Iterator it = vertexMap.values().iterator(); it.hasNext();) {
            Vertex v = (Vertex) it.next();
            v.visited = false;
        }

        for (Iterator it = vertexMap.values().iterator(); it.hasNext();) {
            Vertex v = (Vertex) it.next();
            for (int i = 0; i < v.children.size(); i++) {
                Vertex child = (Vertex) v.children.get(i);
                child.visited = true;
            }
        }

        for (Iterator it = vertexMap.values().iterator(); it.hasNext();) {
            Vertex v = (Vertex) it.next();
            if (v.visited == false)
                roots.add(v);
        }
        return roots;
    }

    public GraphCell createGraphCell(Vertex v, int x, int y, int height, int width, float color, Map attributes) {
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

    public DefaultEdge createEdge(DefaultGraphCell v1, DefaultGraphCell v2, Map attributes, ConnectionSet cs, ArrayList points) {

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

    public Edge getEdge(FunctionProfile p, FunctionProfile c) {

        Vertex parent = (Vertex) vertexMap.get(p);
        Vertex child = (Vertex) vertexMap.get(c);

        int portCount = child.graphCell.getChildCount();
        for (int j = 0; j < portCount; j++) {
            Port port = (Port) child.graphCell.getChildAt(j);

            for (Iterator itrEdges = port.edges(); itrEdges.hasNext();) {
                Edge edge = (Edge) itrEdges.next();

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

    public void handleColorEvent() {
        Map attributeMap = new Hashtable();

        // color all edges black and reset pathHighlight to false
        for (int i = 0; i < graphCellList.size(); i++) {
            GraphCell dgc = (GraphCell) graphCellList.get(i);
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
            GraphCell dgc = (GraphCell) graphCellList.get(i);

            if (dgc.function == ppTrial.getHighlightedFunction()) { // this is the one

                // now iterate through each function and check for callpaths that contain it, highlight those edges and vertices
                for (int j = 0; j < functionProfileList.size(); j++) {
                    FunctionProfile fp = (FunctionProfile) functionProfileList.get(j);
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

                                Vertex v = (Vertex) vertexMap.get(parentFunction);
                                v.setPathHighlight(true);
                                v = (Vertex) vertexMap.get(childFunction);
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

            GraphCell dgc = (GraphCell) graphCellList.get(i);
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

    public GraphCell getGraphCellForLocation(int x, int y) {
        for (int i = 0; i < graphCellList.size(); i++) {
            GraphCell gc = (GraphCell) graphCellList.get(i);

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

}