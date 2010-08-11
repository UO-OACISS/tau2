package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Vector;

import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JSeparator;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.perfdmf.CallPathUtilFuncs;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.FunctionProfile;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.perfdmf.UtilFncs;

/**
 * CallPathTextWindowPanel: This is the panel for the CallPathTextWindow
 *   
 * <P>CVS $Id: CallPathTextWindowPanel.java,v 1.45 2009/09/10 00:13:44 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.45 $
 * @see		CallPathDrawObject
 * @see		CallPathTextWindow
 * 
 * TODO:    1) Add printing support. 
 *          2) Need to do quite a bit of work in the renderIt function, such as
 *             adding clipping support, and bringing it more inline with the rest of the
 *             system.
 *          3) (Alan) Actually, renderIt needs to be completely rewritten
 */
public class CallPathTextWindowPanel extends JPanel implements MouseListener, Printable, ImageExport {

    /**
	 * 
	 */
	private static final long serialVersionUID = 9057128779288350787L;
	private int xPanelSize = 625;
    private int yPanelSize = 0;
    private boolean calculatePanelSize = true;

    private Thread thread;

    private ParaProfTrial ppTrial = null;
    private CallPathTextWindow window = null;
    private Font monoFont = null;
    private FontMetrics fontMetrics = null;

    //Some drawing details.
    private Vector<CallPathDrawObject> drawObjectsComplete = null;
    private Vector<CallPathDrawObject> drawObjects = null;

    private int base = 20;
    private int startPosition = 0;
    private int excPos = 0;
    private int incPos = 0;
    private int callsPos1 = 0;
    private int namePos = 0;
    private int yHeightNeeded = 0;
    private int xWidthNeeded = 0;

    private int rowHeight = 10;

    private int lastHeaderEndPosition = 0;

    private Searcher searcher;

    private String normalHeader = "      Exclusive        Inclusive      Calls/Tot.Calls     Name[id]";
    private String normalDashString = "      --------------------------------------------------------------------------------";

    public CallPathTextWindowPanel(ParaProfTrial ppTrial, Thread thread, CallPathTextWindow cPTWindow) {

        this.thread = thread;
        this.ppTrial = ppTrial;
        this.window = cPTWindow;

        setAutoscrolls(true);
        searcher = new Searcher(this, cPTWindow);
        addMouseListener(searcher);
        addMouseMotionListener(searcher);

        setBackground(Color.white);

        addMouseListener(this);

    }

    public void paintComponent(Graphics g) {
        try {
            super.paintComponent(g);
            export((Graphics2D) g, true, false, false);
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
            window.closeThisWindow();
        }
    }

    public int print(Graphics g, PageFormat pageFormat, int page) {
        try {
            if (page >= 1) {
                return NO_SUCH_PAGE;
            }

            ParaProfUtils.scaleForPrint(g, pageFormat, xPanelSize, yPanelSize);
            export((Graphics2D) g, false, true, false);

            return Printable.PAGE_EXISTS;
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
            return NO_SUCH_PAGE;
        }
    }

    private void createDrawObjectsComplete() {
        drawObjectsComplete = new Vector<CallPathDrawObject>();
        //Add five spacer objects representing the column headings.
        drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
        drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
        drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
        drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));

        Iterator<PPFunctionProfile> l1 = window.getDataIterator();
        while (l1.hasNext()) {
            PPFunctionProfile ppFunctionProfile = l1.next();
            //Don't draw callpath functions, only nodes
            if (!(ppFunctionProfile.isCallPathObject())) {
                Iterator<FunctionProfile> l2 = ppFunctionProfile.getParentProfiles();
                while (l2.hasNext()) {
                    FunctionProfile parent = (FunctionProfile) l2.next();
                    Iterator<FunctionProfile> l3 = ppFunctionProfile.getFunctionProfile().getParentProfileCallPathIterator(parent);
                    double d1 = 0.0;
                    double d2 = 0.0;
                    double d3 = 0.0;

                    while (l3.hasNext()) {
                        FunctionProfile callPath = (FunctionProfile) l3.next();
                        d1 = d1 + callPath.getExclusive(ppTrial.getDefaultMetric().getID());
                        d2 = d2 + callPath.getInclusive(ppTrial.getDefaultMetric().getID());
                        d3 = d3 + callPath.getNumCalls();
                    }
                    CallPathDrawObject callPathDrawObject = new CallPathDrawObject(parent.getFunction(), true, false, false);
                    callPathDrawObject.setExclusiveValue(d1);
                    callPathDrawObject.setInclusiveValue(d2);
                    callPathDrawObject.setNumberOfCallsFromCallPathObjects(d3);
                    callPathDrawObject.setNumberOfCalls(ppFunctionProfile.getNumberOfCalls());
                    drawObjectsComplete.add(callPathDrawObject);
                }

                CallPathDrawObject callPathDrawObject = new CallPathDrawObject(ppFunctionProfile.getFunction(), false, false,
                        false);
                callPathDrawObject.setExclusiveValue(ppFunctionProfile.getExclusiveValue());
                callPathDrawObject.setInclusiveValue(ppFunctionProfile.getInclusiveValue());
                callPathDrawObject.setNumberOfCalls(ppFunctionProfile.getNumberOfCalls());
                drawObjectsComplete.add(callPathDrawObject);

                for (Iterator<FunctionProfile> it2 = ppFunctionProfile.getChildProfiles(); it2.hasNext();) {
                    FunctionProfile child = (FunctionProfile) it2.next();
                    double d1 = 0.0;
                    double d2 = 0.0;
                    double d3 = 0.0;
                    for (Iterator<FunctionProfile> it3 = ppFunctionProfile.getFunctionProfile().getChildProfileCallPathIterator(child); it3.hasNext();) {
                        FunctionProfile callPath = (FunctionProfile) it3.next();
                        d1 = d1 + callPath.getExclusive(ppTrial.getDefaultMetric().getID());
                        d2 = d2 + callPath.getInclusive(ppTrial.getDefaultMetric().getID());
                        d3 = d3 + callPath.getNumCalls();
                    }
                    callPathDrawObject = new CallPathDrawObject(child.getFunction(), false, true, false);
                    callPathDrawObject.setExclusiveValue(d1);
                    callPathDrawObject.setInclusiveValue(d2);
                    callPathDrawObject.setNumberOfCallsFromCallPathObjects(d3);
                    callPathDrawObject.setNumberOfCalls(child.getNumCalls());
                    drawObjectsComplete.add(callPathDrawObject);
                }
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
            }
        }

    }

    private void createDrawObjects() {
        drawObjects = new Vector<CallPathDrawObject>();
        Vector<CallPathDrawObject> holdingPattern = new Vector<CallPathDrawObject>();
        boolean adding = false;
        int state = -1;
        int size = -1;
        if (window.showCollapsedView()) {
            for (Enumeration<CallPathDrawObject> e = drawObjectsComplete.elements(); e.hasMoreElements();) {
                CallPathDrawObject callPathDrawObject = e.nextElement();
                if (callPathDrawObject.isSpacer())
                    state = 0;
                else if (callPathDrawObject.isParent()) {
                    if (adding)
                        state = 1;
                    else
                        state = 2;
                } else if (callPathDrawObject.isChild()) {
                    if (adding)
                        state = 3;
                    else
                        state = 4;
                } else {
                    if (adding)
                        state = 5;
                    else
                        state = 6;
                }

                switch (state) {
                case 0:
                    drawObjects.add(callPathDrawObject);
                    break;
                case 1:
                    adding = false;
                    holdingPattern.add(callPathDrawObject);
                    break;
                case 2:
                    holdingPattern.add(callPathDrawObject);
                    break;
                case 3:
                    drawObjects.add(callPathDrawObject);
                    break;
                case 5:
                    //Transfer holdingPattern elements to
                    // drawObjects, then add this function
                    //to drawObjects.
                    size = holdingPattern.size();
                    for (int i = 0; i < size; i++)
                        drawObjects.add(holdingPattern.elementAt(i));
                    holdingPattern.clear();
                    drawObjects.add(callPathDrawObject);
                    //Now check to see if this object is expanded.
                    if (callPathDrawObject.isExpanded())
                        adding = true;
                    else
                        adding = false;
                    break;
                case 6:
                    if (callPathDrawObject.isExpanded()) {
                        //Transfer holdingPattern elements to
                        // drawObjects, then add this function
                        //to drawObjects.
                        size = holdingPattern.size();
                        for (int i = 0; i < size; i++)
                            drawObjects.add(holdingPattern.elementAt(i));
                        holdingPattern.clear();
                        adding = true;
                    } else {
                        holdingPattern.clear();
                    }
                    drawObjects.add(callPathDrawObject);
                    break;
                default:
                }
            }
        } else {
            drawObjects = drawObjectsComplete;
        }

    }

    private void setSearchLines() {
        if (searcher.getSearchLines() == null) {
            Vector<String> searchLines = new Vector<String>();
            for (int i = 0; i < drawObjects.size(); i++) {
                String line;
                CallPathDrawObject callPathDrawObject = drawObjects.elementAt(i);
                if (i == 1) {
                    line = normalHeader;
                } else if (i == 2) {
                    line = normalDashString;
                } else if (!callPathDrawObject.isParentChild() && !callPathDrawObject.isSpacer()) {

                    //Function function = callPathDrawObject.getFunction();

                    line = "--> " + UtilFncs.getOutputString(window.units(), callPathDrawObject.getExclusiveValue(), 11, ppTrial.getDefaultMetric().isTimeDenominator())
                            + "      " + UtilFncs.getOutputString(window.units(), callPathDrawObject.getInclusiveValue(), 11, ppTrial.getDefaultMetric().isTimeDenominator())
                            + "      " + UtilFncs.formatDouble(callPathDrawObject.getNumberOfCalls(), 7, false);

                    line = UtilFncs.pad(line, 58) + callPathDrawObject.getName();// + "[" + function.getID() + "]";

                } else if (callPathDrawObject.isSpacer()) {
                    line = " ";
                } else {

                    //Function function = callPathDrawObject.getFunction();

                    line = "    " + UtilFncs.getOutputString(window.units(), callPathDrawObject.getExclusiveValue(), 11, ppTrial.getDefaultMetric().isTimeDenominator())
                            + "      " + UtilFncs.getOutputString(window.units(), callPathDrawObject.getInclusiveValue(), 11, ppTrial.getDefaultMetric().isTimeDenominator())
                            + "      "
                            + UtilFncs.formatDouble(callPathDrawObject.getNumberOfCallsFromCallPathObjects(), 7, false) + "/"
                            + UtilFncs.formatDouble(callPathDrawObject.getNumberOfCalls(), 7, false);

                    line = UtilFncs.pad(line, 58) + callPathDrawObject.getName();// + "[" + function.getID() + "]";

                }
                searchLines.add(line);
            }
            searcher.setSearchLines(searchLines);
        }
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {

        int yCoord = 0;

        //In this window, a Monospaced font has to be used. This will
        // probably not be the same font as the rest of ParaProf.

        monoFont = new Font("Monospaced", ppTrial.getPreferencesWindow().getFontStyle(), ParaProf.preferencesWindow.getFontSize());
        fontMetrics = g2D.getFontMetrics(monoFont);
        //int maxFontAscent = fontMetrics.getMaxAscent();
        //int maxFontDescent = fontMetrics.getMaxDescent();

        g2D.setFont(monoFont);

        rowHeight = fontMetrics.getHeight();

        searcher.setXOffset(base);
        searcher.setG2d(g2D);
        searcher.setLineHeight(rowHeight);

        //TODO: rewrite this crap

        CallPathDrawObject callPathDrawObject = null;

        CallPathUtilFuncs.buildThreadRelations(ppTrial.getDataSource(), thread);

        //Populate drawObjectsComplete vector.
        //This should only happen once.
        if (drawObjectsComplete == null) {
            createDrawObjectsComplete();
        }

        //Populate drawObjects vector.
        if (drawObjects == null) {
            createDrawObjects();
            searcher.setSearchLines(null);
            setSearchLines();
        }

        //######
        //Set panel size.
        //######

        if (this.calculatePanelSize) {

            int maxNameLength = 0;
            yHeightNeeded = 0;
            for (Enumeration<CallPathDrawObject> e = drawObjects.elements(); e.hasMoreElements();) {
                callPathDrawObject = e.nextElement();
                yHeightNeeded = yHeightNeeded + rowHeight;

                if (!callPathDrawObject.isSpacer()) {
                    maxNameLength = Math.max(maxNameLength, callPathDrawObject.getName().length());
                }
            }

            int charWidth = fontMetrics.stringWidth("A");
            startPosition = fontMetrics.stringWidth("--> ") + base;

            excPos = base + (charWidth * 4);
            incPos = excPos + (charWidth * 17);
            callsPos1 = incPos + (charWidth * 17);
            namePos = callsPos1 + (charWidth * 20);

            xWidthNeeded = (maxNameLength * charWidth) + namePos + 30;

            boolean sizeChange = false;
            //Resize the panel if needed.
            if (xWidthNeeded > xPanelSize) {
                xPanelSize = xWidthNeeded + 10;
                sizeChange = true;
            }
            if (yHeightNeeded > yPanelSize) {
                yPanelSize = yHeightNeeded + 10;
                sizeChange = true;
            }
            if (sizeChange && toScreen) {
                revalidate();
            }
            this.calculatePanelSize = false;

        }
        //######
        //End - Set panel size.
        //######

        // determine which elements to draw (clipping)
        int[] clips = ParaProfUtils.computeClipping(g2D.getClipBounds(), window.getViewRect(), toScreen, fullWindow,
                drawObjects.size(), rowHeight, yCoord);
        int startElement = clips[0];
        int endElement = clips[1];
        yCoord = clips[2];

        g2D.setColor(Color.black);

        if (drawHeader) {
            JScrollPane sp = window.getScrollPane();
            sp.getColumnHeader().paintAll(g2D);
            g2D.translate(0, sp.getColumnHeader().getHeight());
        }

        yCoord = yCoord + rowHeight;
        //######
        //End - Draw the header if required.
        //######
        for (int i = startElement; i <= endElement; i++) {
            searcher.drawHighlights(g2D, base, yCoord, i);
            g2D.setColor(Color.black);

            callPathDrawObject = drawObjects.elementAt(i);
            if (i == 1) {
                String header = normalHeader;
                g2D.drawString(header, base, yCoord);
            } else if (i == 2) {
                String dashString = normalDashString;
                g2D.drawString(dashString, base, yCoord);
            } else if (!callPathDrawObject.isParentChild() && !callPathDrawObject.isSpacer()) {

                String stats = "--> " + UtilFncs.getOutputString(window.units(), callPathDrawObject.getExclusiveValue(), 11, ppTrial.getDefaultMetric().isTimeDenominator())
                        + "      " + UtilFncs.getOutputString(window.units(), callPathDrawObject.getInclusiveValue(), 11, ppTrial.getDefaultMetric().isTimeDenominator())
                        + "      " + UtilFncs.formatDouble(callPathDrawObject.getNumberOfCalls(), 7, false);
                g2D.drawString(stats, base, yCoord);

                Function function = callPathDrawObject.getFunction();
                if (ppTrial.getHighlightedFunction() == function) {
                    g2D.setColor(Color.red);
                }

                g2D.drawString(callPathDrawObject.getName(), namePos, yCoord); // + "[" + function.getID() + "]", namePos, yCoord);

            } else if (callPathDrawObject.isSpacer()) {} else {

                String stats = "    " + UtilFncs.getOutputString(window.units(), callPathDrawObject.getExclusiveValue(), 11, ppTrial.getDefaultMetric().isTimeDenominator())
                        + "      " + UtilFncs.getOutputString(window.units(), callPathDrawObject.getInclusiveValue(), 11, ppTrial.getDefaultMetric().isTimeDenominator())
                        + "      " + UtilFncs.formatDouble(callPathDrawObject.getNumberOfCallsFromCallPathObjects(), 7, false)
                        + "/" + UtilFncs.formatDouble(callPathDrawObject.getNumberOfCalls(), 7, false);

                //g2D.drawString(stats, base, yCoord);
                Function function = callPathDrawObject.getFunction();

                if (ppTrial.getHighlightedFunction() == function) {
                    g2D.setColor(Color.red);
                }

                String functionString = callPathDrawObject.getName();// + "[" + function.getID() + "]";

                stats = UtilFncs.pad(stats, 58) + functionString;

                g2D.drawString(stats, base, yCoord);

                //g2D.drawString(functionString, namePos, yCoord);

            }
            yCoord = yCoord + rowHeight;

        }

    }

    public void mouseClicked(MouseEvent evt) {
        try {
            //Get the location of the mouse.
            int xCoord = evt.getX();
            int yCoord = evt.getY();

            //Calculate which CallPathDrawObject was clicked on.
            int index = (yCoord - 1) / (rowHeight);

            if (index < drawObjects.size()) {
                final CallPathDrawObject callPathDrawObject = drawObjects.elementAt(index);
                if (!callPathDrawObject.isSpacer()) {
                    if (ParaProfUtils.rightClick(evt)) {
                        JPopupMenu popup = ParaProfUtils.createFunctionClickPopUp(ppTrial, callPathDrawObject.getFunction(),
                                thread, this);

                        popup.add(new JSeparator());

                        JMenuItem menuItem = new JMenuItem("Goto Function");
                        menuItem.addActionListener(new ActionListener() {
                            public void actionPerformed(ActionEvent evt) {
                                try {

                                    Function function = callPathDrawObject.getFunction();
                                    int size = drawObjects.size();
                                    for (int i = 0; i < size; i++) {
                                        CallPathDrawObject callPathDrawObject2 = drawObjects.elementAt(i);
                                        if ((callPathDrawObject2.getFunction() == function)
                                                && (!callPathDrawObject2.isParentChild())) {
                                            Dimension dimension = window.getViewportSize();
                                            window.setVerticalScrollBarPosition((i * rowHeight)
                                                    - ((int) dimension.getHeight() / 2));
                                            ppTrial.setHighlightedFunction(function);
                                            return;
                                        }
                                    }
                                } catch (Exception e) {
                                    ParaProfUtils.handleException(e);
                                }
                            }

                        });

                        popup.add(menuItem);
                        popup.show(this, evt.getX(), evt.getY());
                    } else {
                        //Check to see if the click occured to the left of
                        // startPosition.
                        if (xCoord < startPosition) {
                            if (!callPathDrawObject.isParentChild()) {
                                if (callPathDrawObject.isExpanded())
                                    callPathDrawObject.setExpanded(false);
                                else
                                    callPathDrawObject.setExpanded(true);
                            }
                            drawObjects = null;
                        }
                        ppTrial.toggleHighlightedFunction(callPathDrawObject.getFunction());
                    }
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void mousePressed(MouseEvent evt) {}

    public void mouseReleased(MouseEvent evt) {}

    public void mouseEntered(MouseEvent evt) {}

    public void mouseExited(MouseEvent evt) {}

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        Dimension d = null;
        if (fullScreen) {
            d = this.getSize();
        } else {
            d = window.getSize();
        }
        d.setSize(d.getWidth(), d.getHeight() + lastHeaderEndPosition);
        return d;
    }

    public void resetAllDrawObjects() {
        if (drawObjectsComplete != null) {
            drawObjectsComplete.clear();
        }
        drawObjectsComplete = null;
        drawObjects.clear();
        drawObjects = null;
        searcher.setSearchLines(null);
        calculatePanelSize = true;
    }

    public Dimension getPreferredSize() {
        return new Dimension(xPanelSize, (yPanelSize + 10));
    }

    public Searcher getSearcher() {
        return searcher;
    }

}