package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import javax.swing.*;
import java.awt.geom.*;
import edu.uoregon.tau.dms.dss.*;
import java.awt.font.*;
import java.text.*;

/**
 * CallPathTextWindowPanel: This is the panel for the CallPathTextWindow
 *   
 * <P>CVS $Id: CallPathTextWindowPanel.java,v 1.11 2005/01/04 01:16:26 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.11 $
 * @see		CallPathDrawObject
 * @see		CallPathTextWindow
 * 
 * TODO:    1) Add printing support. 
 *          2) Need to do quite a bit of work in the renderIt function, such as
 *             adding clipping support, and bringing it more inline with the rest of the
 *             system.
 *          3) (Alan) Actually, renderIt needs to be completely rewritten
 */
public class CallPathTextWindowPanel extends JPanel implements ActionListener, MouseListener, Printable,
        ParaProfImageInterface {

    public CallPathTextWindowPanel(ParaProfTrial trial, edu.uoregon.tau.dms.dss.Thread thread,
            CallPathTextWindow cPTWindow, int windowType) {

        this.thread = thread;
        this.trial = trial;
        this.window = cPTWindow;
        this.windowType = windowType;

        setBackground(Color.white);

        addMouseListener(this);

        //Add items to the popu menu.
        JMenuItem jMenuItem = new JMenuItem("Show Function Details");
        jMenuItem.addActionListener(this);
        popup.add(jMenuItem);

        jMenuItem = new JMenuItem("Find Function");
        jMenuItem.addActionListener(this);
        popup.add(jMenuItem);

        jMenuItem = new JMenuItem("Change Function Color");
        jMenuItem.addActionListener(this);
        popup.add(jMenuItem);

        jMenuItem = new JMenuItem("Reset to Generic Color");
        jMenuItem.addActionListener(this);
        popup.add(jMenuItem);

    }

    public void paintComponent(Graphics g) {
        try {
            super.paintComponent(g);
            renderIt((Graphics2D) g, true, false, false);
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
            renderIt((Graphics2D) g, false, true, false);

            return Printable.PAGE_EXISTS;
        } catch (Exception e) {
            new ParaProfErrorDialog(e);
            return NO_SUCH_PAGE;
        }
    }

    
    
    
    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {

        int defaultNumberPrecision = ParaProf.defaultNumberPrecision;
        int yCoord = 0;

        //In this window, a Monospaced font has to be used. This will
        // probably not be the same font as the rest of ParaProf. As a result, some extra work will
        // have to be done to calculate spacing.
        
        int fontSize = trial.getPreferences().getBarHeight();
        spacing = trial.getPreferences().getBarSpacing();

        //Create font.
        monoFont = new Font("Monospaced", trial.getPreferences().getFontStyle(), fontSize);
        //Compute the font metrics.
        fmMonoFont = g2D.getFontMetrics(monoFont);
        maxFontAscent = fmMonoFont.getMaxAscent();
        maxFontDescent = fmMonoFont.getMaxDescent();
        g2D.setFont(monoFont);

        if (spacing <= (maxFontAscent + maxFontDescent)) {
            spacing = spacing + 1;
        }

        //TODO: rewrite this crap

        if (windowType == 0) {
            Iterator l1 = null;
            Iterator l2 = null;
            Iterator l3 = null;
            TrialData gm = trial.getTrialData();

            String s = null;
            Vector functionList = null;
            FunctionProfile gtde = null;
            PPFunctionProfile smwtde = null;
            CallPathDrawObject callPathDrawObject = null;
            double d1 = 0.0;
            double d2 = 0.0;
            double d3 = 0;

            //######
            //Populate drawObjectsComplete vector.
            //This should only happen once.
            //######
            if (drawObjectsComplete == null) {
                drawObjectsComplete = new Vector();
                //Add five spacer objects representing the column headings.
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));

                l1 = window.getDataIterator();
                while (l1.hasNext()) {
                    smwtde = (PPFunctionProfile) l1.next();
                    //Don't draw callpath mapping objects.
                    if (!(smwtde.isCallPathObject())) {
                        l2 = smwtde.getParents();
                        while (l2.hasNext()) {
                            Function parent = (Function) l2.next();
                            l3 = smwtde.getParentCallPathIterator(parent);
                            d1 = 0.0;
                            d2 = 0.0;
                            d3 = 0;
                            while (l3.hasNext()) {
                                Function parentCallPathID = (Function) l3.next();
                                d1 = d1 + parentCallPathID.getMeanExclusive(trial.getSelectedMetricID());
                                d2 = d2 + parentCallPathID.getMeanInclusive(trial.getSelectedMetricID());
                                d3 = d3 + parentCallPathID.getMeanNumCalls();
                            }
                            callPathDrawObject = new CallPathDrawObject(parent, true, false, false);
                            callPathDrawObject.setExclusiveValue(d1);
                            callPathDrawObject.setInclusiveValue(d2);
                            callPathDrawObject.setNumberOfCallsFromCallPathObjects(d3);
                            callPathDrawObject.setNumberOfCalls(smwtde.getNumberOfCalls());
                            drawObjectsComplete.add(callPathDrawObject);
                        }
                        callPathDrawObject = new CallPathDrawObject(smwtde.getFunction(), false, false, false);
                        callPathDrawObject.setExclusiveValue(smwtde.getExclusiveValue());
                        callPathDrawObject.setInclusiveValue(smwtde.getInclusiveValue());
                        callPathDrawObject.setNumberOfCalls(smwtde.getNumberOfCalls());
                        drawObjectsComplete.add(callPathDrawObject);
                        l2 = smwtde.getChildren();
                        while (l2.hasNext()) {
                            Function child = (Function) l2.next();
                            l3 = smwtde.getChildCallPathIterator(child);
                            d1 = 0.0;
                            d2 = 0.0;
                            d3 = 0;
                            while (l3.hasNext()) {
                                Function childCallPathID = (Function) l3.next();
                                d1 = d1 + childCallPathID.getMeanExclusive(trial.getSelectedMetricID());
                                d2 = d2 + childCallPathID.getMeanInclusive(trial.getSelectedMetricID());
                                d3 = d3 + childCallPathID.getMeanNumCalls();
                            }
                            callPathDrawObject = new CallPathDrawObject(child, false, true, false);
                            callPathDrawObject.setExclusiveValue(d1);
                            callPathDrawObject.setInclusiveValue(d2);
                            callPathDrawObject.setNumberOfCallsFromCallPathObjects(d3);
                            callPathDrawObject.setNumberOfCalls(child.getMeanNumCalls());
                            drawObjectsComplete.add(callPathDrawObject);
                        }
                        drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                        drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                    }
                }
            }
            //######
            //End - Populate drawObjectsComplete vector.
            //######

            //######
            //Populate drawObjects vector.
            //######
            if (drawObjects == null) {
                drawObjects = new Vector();
                Vector holdingPattern = new Vector();
                boolean adding = false;
                int state = -1;
                int size = -1;
                if (window.showCollapsedView()) {
                    for (Enumeration e = drawObjectsComplete.elements(); e.hasMoreElements();) {
                        callPathDrawObject = (CallPathDrawObject) e.nextElement();
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
                } else
                    drawObjects = drawObjectsComplete;
            }
            //######
            //End - Populate drawObjects vector.
            //######

            //######
            //Set panel size.
            //######

            if (this.calculatePanelSize()) {
                for (Enumeration e = drawObjects.elements(); e.hasMoreElements();) {
                    callPathDrawObject = (CallPathDrawObject) e.nextElement();
                    yHeightNeeded = yHeightNeeded + (spacing);
                    max = setMax(max, callPathDrawObject.getExclusiveValue(),
                            callPathDrawObject.getInclusiveValue());

                    if (!callPathDrawObject.isSpacer()) {
                        length = fmMonoFont.stringWidth(callPathDrawObject.getName()) + 10;
                        if (xWidthNeeded < length)
                            xWidthNeeded = length;
                    }
                }

                base = 20;
                startPosition = fmMonoFont.stringWidth("--> ") + base;
                stringWidth = (fmMonoFont.stringWidth(UtilFncs.getOutputString(window.units(), max,
                        defaultNumberPrecision))) + 30;
                check = fmMonoFont.stringWidth("Exclusive");
                if (stringWidth < check)
                    stringWidth = check + 35;
                numCallsWidth = (fmMonoFont.stringWidth(Double.toString(gm.getMaxMeanNumberOfCalls()))) + 30;
                check = fmMonoFont.stringWidth("Calls/Tot.Calls");
                if (numCallsWidth < check)
                    numCallsWidth = check + 35;
                excPos = startPosition;
                incPos = excPos + stringWidth;
                callsPos1 = incPos + stringWidth;
                callsPos2 = callsPos1 + numCallsWidth;
                namePos = callsPos2 + numCallsWidth;
                //Add this to the positon of the name plus a little extra.
                xWidthNeeded = xWidthNeeded + namePos + 20;

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
                if (sizeChange && toScreen)
                    revalidate();
                this.setCalculatePanelSize(false);
            }
            //######
            //End - Set panel size.
            //######

            int yBeg = 0;
            int yEnd = 0;
            int startElement = 0;
            int endElement = 0;
            Rectangle clipRect = null;
            Rectangle viewRect = null;

            if (!fullWindow) {
                if (toScreen) {
                    clipRect = g2D.getClipBounds();
                    yBeg = (int) clipRect.getY();
                    yEnd = (int) (yBeg + clipRect.getHeight());
                } else {
                    viewRect = window.getViewRect();
                    yBeg = (int) viewRect.getY();
                    yEnd = (int) (yBeg + viewRect.getHeight());
                }

                startElement = ((yBeg - yCoord) / spacing) - 1;
                endElement = ((yEnd - yCoord) / spacing) + 1;

                if (startElement < 0)
                    startElement = 0;

                if (endElement < 0)
                    endElement = 0;

                if (startElement > (drawObjects.size() - 1))
                    startElement = (drawObjects.size() - 1);

                if (endElement > (drawObjects.size() - 1))
                    endElement = (drawObjects.size() - 1);

                if (toScreen)
                    yCoord = yCoord + (startElement * spacing);
            } else {
                startElement = 0;
                endElement = ((drawObjects.size()) - 1);
            }

            /*
             * //At this point we can determine the size this panel will
             * //require. If we need to resize, don't do any more drawing,
             * //just call revalidate. Make sure we check the instruction
             * value as we only want to //revalidate if we are drawing to
             * the screen. if(resizePanel(fmFont, barXCoord, list,
             * startElement, endElement) && instruction==0){
             * this.revalidate(); return; }
             */

            g2D.setColor(Color.black);
            //Draw the header if required.
            if (drawHeader) {
                yCoord = yCoord + (spacing);
                String headerString = window.getHeaderString();
                //Need to split the string up into its separate lines.
                StringTokenizer st = new StringTokenizer(headerString, "'\n'");
                while (st.hasMoreTokens()) {
                    g2D.drawString(st.nextToken(), 15, yCoord);
                    yCoord = yCoord + (spacing);
                }
                lastHeaderEndPosition = yCoord;
            }
         
            
            for (int i = startElement; i <= endElement; i++) {
                callPathDrawObject = (CallPathDrawObject) drawObjects.elementAt(i);
                if (i == 1) {
                    g2D.drawString("Exclusive", excPos, yCoord);
                    g2D.drawString("Inclusive", incPos, yCoord);
                    g2D.drawString("Calls/Tot.Calls", callsPos1, yCoord);
                    g2D.drawString("Name[id]", namePos, yCoord);
                    yCoord = yCoord + spacing;
                } else if (i == 2) {
                    g2D.drawString(
                            "--------------------------------------------------------------------------------",
                            excPos, yCoord);
                    yCoord = yCoord + spacing;
                } else if (!callPathDrawObject.isParentChild() && !callPathDrawObject.isSpacer()) {
                    g2D.drawString("--> ", base, yCoord);
                    g2D.drawString(UtilFncs.getOutputString(window.units(),
                            callPathDrawObject.getExclusiveValue(), ParaProf.defaultNumberPrecision), excPos,
                            yCoord);
                    g2D.drawString(UtilFncs.getOutputString(window.units(),
                            callPathDrawObject.getInclusiveValue(), ParaProf.defaultNumberPrecision), incPos,
                            yCoord);
                    g2D.drawString(Double.toString(callPathDrawObject.getNumberOfCalls()), callsPos1, yCoord);
                    Function function = callPathDrawObject.getFunction();
                    if (trial.getColorChooser().getHighlightedFunction() == function) {
                        g2D.setColor(Color.red);
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]", namePos,
                                yCoord);
                        g2D.setColor(Color.black);
                    } else
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]", namePos,
                                yCoord);
                    yCoord = yCoord + (spacing);
                } else if (callPathDrawObject.isSpacer())
                    yCoord = yCoord + spacing;
                else {
                    g2D.drawString(UtilFncs.getOutputString(window.units(),
                            callPathDrawObject.getExclusiveValue(), ParaProf.defaultNumberPrecision), excPos,
                            yCoord);
                    g2D.drawString(UtilFncs.getOutputString(window.units(),
                            callPathDrawObject.getInclusiveValue(), ParaProf.defaultNumberPrecision), incPos,
                            yCoord);
                    g2D.drawString(callPathDrawObject.getNumberOfCallsFromCallPathObjects() + "/"
                            + callPathDrawObject.getNumberOfCalls(), callsPos1, yCoord);
                    Function function = callPathDrawObject.getFunction();
                    if (trial.getColorChooser().getHighlightedFunction() == function) {
                        g2D.setColor(Color.red);
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]", namePos,
                                yCoord);
                        g2D.setColor(Color.black);
                    } else
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]", namePos,
                                yCoord);
                    yCoord = yCoord + (spacing);
                }
            }
        } else if (windowType == 1) {
            Iterator l1 = null;
            Iterator l2 = null;
            Iterator l3 = null;
            TrialData gm = trial.getTrialData();
            String s = null;
            Vector functionList = null;
            FunctionProfile gtde = null;
            PPFunctionProfile smwtde = null;
            CallPathDrawObject callPathDrawObject = null;
            double d1 = 0.0;
            double d2 = 0.0;
            double d3 = 0;

           
            functionList = thread.getFunctionList();

            //######
            //Populate drawObjectsComplete vector.
            //This should only happen once.
            //######
            if (drawObjectsComplete == null) {
                drawObjectsComplete = new Vector();
                //Add five spacer objects representing the column headings.
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));

                l1 = window.getDataIterator();
                while (l1.hasNext()) {
                    smwtde = (PPFunctionProfile) l1.next();
                    //Don't draw callpath mapping objects.
                    if (!(smwtde.isCallPathObject())) {
                        l2 = smwtde.getParents();
                        while (l2.hasNext()) {
                            Function parent = (Function) l2.next();
                            l3 = smwtde.getParentCallPathIterator(parent);
                            d1 = 0.0;
                            d2 = 0.0;
                            d3 = 0;
                            while (l3.hasNext()) {
                                Function callPath = (Function) l3.next();
                                gtde = (FunctionProfile) functionList.elementAt(callPath.getID());
                                d1 = d1 + gtde.getExclusive(trial.getSelectedMetricID());
                                d2 = d2 + gtde.getInclusive(trial.getSelectedMetricID());
                                d3 = d3 + gtde.getNumCalls();
                            }
                            callPathDrawObject = new CallPathDrawObject(parent, true, false, false);
                            callPathDrawObject.setExclusiveValue(d1);
                            callPathDrawObject.setInclusiveValue(d2);
                            callPathDrawObject.setNumberOfCallsFromCallPathObjects(d3);
                            callPathDrawObject.setNumberOfCalls(smwtde.getNumberOfCalls());
                            drawObjectsComplete.add(callPathDrawObject);
                        }
                        callPathDrawObject = new CallPathDrawObject(smwtde.getFunction(), false, false, false);
                        callPathDrawObject.setExclusiveValue(smwtde.getExclusiveValue());
                        callPathDrawObject.setInclusiveValue(smwtde.getInclusiveValue());
                        callPathDrawObject.setNumberOfCalls(smwtde.getNumberOfCalls());
                        drawObjectsComplete.add(callPathDrawObject);
                        l2 = smwtde.getChildren();
                        while (l2.hasNext()) {
                            Function child = (Function) l2.next();
                            l3 = smwtde.getChildCallPathIterator(child);
                            d1 = 0.0;
                            d2 = 0.0;
                            d3 = 0;
                            while (l3.hasNext()) {
                                Function callPath = (Function) l3.next();
                                gtde = (FunctionProfile) functionList.elementAt(callPath.getID());
                                d1 = d1 + gtde.getExclusive(trial.getSelectedMetricID());
                                d2 = d2 + gtde.getInclusive(trial.getSelectedMetricID());
                                d3 = d3 + gtde.getNumCalls();
                            }
                            callPathDrawObject = new CallPathDrawObject(child, false, true, false);
                            callPathDrawObject.setExclusiveValue(d1);
                            callPathDrawObject.setInclusiveValue(d2);
                            callPathDrawObject.setNumberOfCallsFromCallPathObjects(d3);
                            callPathDrawObject.setNumberOfCalls(thread.getFunctionProfile(child).getNumCalls());
                            drawObjectsComplete.add(callPathDrawObject);
                        }
                        drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                        drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                    }
                }
            }
            //######
            //End - Populate drawObjectsComplete vector.
            //######

            //######
            //Populate drawObjects vector.
            //######
            if (drawObjects == null) {
                drawObjects = new Vector();
                Vector holdingPattern = new Vector();
                boolean adding = false;
                int state = -1;
                int size = -1;
                if (window.showCollapsedView()) {
                    for (Enumeration e = drawObjectsComplete.elements(); e.hasMoreElements();) {
                        callPathDrawObject = (CallPathDrawObject) e.nextElement();
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
                } else
                    drawObjects = drawObjectsComplete;
            }
            //######
            //End - Populate drawObjects vector.
            //######

            //######
            //Set panel size.
            //######

            if (this.calculatePanelSize()) {
                for (Enumeration e = drawObjects.elements(); e.hasMoreElements();) {
                    callPathDrawObject = (CallPathDrawObject) e.nextElement();
                    yHeightNeeded = yHeightNeeded + (spacing);

                    max = setMax(max, fmMonoFont.stringWidth(UtilFncs.getOutputString(window.units(),
                            callPathDrawObject.getExclusiveValue(), defaultNumberPrecision)),
                            fmMonoFont.stringWidth(UtilFncs.getOutputString(window.units(),
                                    callPathDrawObject.getInclusiveValue(), defaultNumberPrecision)));

                    //			max =
                    // setMax(max,callPathDrawObject.getExclusiveValue(),callPathDrawObject.getInclusiveValue());

                    if (!callPathDrawObject.isSpacer()) {
                        length = fmMonoFont.stringWidth(callPathDrawObject.getName()) + 10;
                        if (xWidthNeeded < length)
                            xWidthNeeded = length;
                    }
                }

                base = 20;
                startPosition = fmMonoFont.stringWidth("--> ") + base;
                stringWidth =
                 (fmMonoFont.stringWidth(UtilFncs.getOutputString(window.units(),max,defaultNumberPrecision)))+50;
                //stringWidth = (int) max + 10;

                check = fmMonoFont.stringWidth("Exclusive");
                if (stringWidth < check)
                    stringWidth = check + 25;
                numCallsWidth = (fmMonoFont.stringWidth(Integer.toString((int) thread.getMaxNumCalls()))) + 25;
                check = fmMonoFont.stringWidth("Calls/Tot.Calls");
                if (numCallsWidth < check)
                    numCallsWidth = check + 25;
                excPos = startPosition;
                incPos = excPos + stringWidth;
                callsPos1 = incPos + stringWidth;
                callsPos2 = callsPos1 + numCallsWidth;
                namePos = callsPos2 + numCallsWidth;
                //Add this to the positon of the name plus a little extra.
                xWidthNeeded = xWidthNeeded + namePos + 20;

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
                if (sizeChange && toScreen)
                    revalidate();
                this.setCalculatePanelSize(false);
            }
            //######
            //End - Set panel size.
            //######

            int yBeg = 0;
            int yEnd = 0;
            int startElement = 0;
            int endElement = 0;
            Rectangle clipRect = null;
            Rectangle viewRect = null;

            if (!fullWindow) {
                if (toScreen) {
                    clipRect = g2D.getClipBounds();
                    yBeg = (int) clipRect.getY();
                    yEnd = (int) (yBeg + clipRect.getHeight());
                } else {
                    viewRect = window.getViewRect();
                    yBeg = (int) viewRect.getY();
                    yEnd = (int) (yBeg + viewRect.getHeight());
                }

                startElement = ((yBeg - yCoord) / spacing) - 1;
                endElement = ((yEnd - yCoord) / spacing) + 1;

                if (startElement < 0)
                    startElement = 0;

                if (endElement < 0)
                    endElement = 0;

                if (startElement > (drawObjects.size() - 1))
                    startElement = (drawObjects.size() - 1);

                if (endElement > (drawObjects.size() - 1))
                    endElement = (drawObjects.size() - 1);

                if (toScreen)
                    yCoord = yCoord + (startElement * spacing);
            } else {
                startElement = 0;
                endElement = ((drawObjects.size()) - 1);
            }

            /*
             * //At this point we can determine the size this panel will
             * //require. If we need to resize, don't do any more drawing,
             * //just call revalidate. Make sure we check the instruction
             * value as we only want to //revalidate if we are drawing to
             * the screen. if(resizePanel(fmFont, barXCoord, list,
             * startElement, endElement) && instruction==0){
             * this.revalidate(); return; }
             */

            g2D.setColor(Color.black);
            //######
            //Draw the header if required.
            //######
            if (drawHeader) {
                yCoord = yCoord + (spacing);
                String headerString = window.getHeaderString();
                //Need to split the string up into its separate lines.
                StringTokenizer st = new StringTokenizer(headerString, "'\n'");
                while (st.hasMoreTokens()) {
                    g2D.drawString(st.nextToken(), 15, yCoord);
                    yCoord = yCoord + (spacing);
                }
                lastHeaderEndPosition = yCoord;
            }
            //######
            //End - Draw the header if required.
            //######
            for (int i = startElement; i <= endElement; i++) {
                callPathDrawObject = (CallPathDrawObject) drawObjects.elementAt(i);
                if (i == 1) {
                    g2D.drawString("Exclusive", excPos, yCoord);
                    g2D.drawString("Inclusive", incPos, yCoord);
                    g2D.drawString("Calls/Tot.Calls", callsPos1, yCoord);
                    g2D.drawString("Name[id]", namePos, yCoord);
                    yCoord = yCoord + spacing;
                } else if (i == 2) {
                    g2D.drawString(
                            "--------------------------------------------------------------------------------",
                            excPos, yCoord);
                    yCoord = yCoord + spacing;
                } else if (!callPathDrawObject.isParentChild() && !callPathDrawObject.isSpacer()) {
                    g2D.drawString("--> ", base, yCoord);
                    g2D.drawString(UtilFncs.getOutputString(window.units(),
                            callPathDrawObject.getExclusiveValue(), ParaProf.defaultNumberPrecision), excPos,
                            yCoord);
                    g2D.drawString(UtilFncs.getOutputString(window.units(),
                            callPathDrawObject.getInclusiveValue(), ParaProf.defaultNumberPrecision), incPos,
                            yCoord);
                    g2D.drawString(Double.toString(callPathDrawObject.getNumberOfCalls()), callsPos1, yCoord);
                    Function function = callPathDrawObject.getFunction();
                    if (trial.getColorChooser().getHighlightedFunction() == function) {
                        g2D.setColor(Color.red);
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]", namePos,
                                yCoord);
                        g2D.setColor(Color.black);
                    } else
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]", namePos,
                                yCoord);
                    yCoord = yCoord + (spacing);
                } else if (callPathDrawObject.isSpacer())
                    yCoord = yCoord + spacing;
                else {
                    g2D.drawString(UtilFncs.getOutputString(window.units(),
                            callPathDrawObject.getExclusiveValue(), ParaProf.defaultNumberPrecision), excPos,
                            yCoord);
                    g2D.drawString(UtilFncs.getOutputString(window.units(),
                            callPathDrawObject.getInclusiveValue(), ParaProf.defaultNumberPrecision), incPos,
                            yCoord);
                    g2D.drawString(callPathDrawObject.getNumberOfCallsFromCallPathObjects() + "/"
                            + callPathDrawObject.getNumberOfCalls(), callsPos1, yCoord);
                    Function function = callPathDrawObject.getFunction();
                    if (trial.getColorChooser().getHighlightedFunction() == function) {
                        g2D.setColor(Color.red);
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]", namePos,
                                yCoord);
                        g2D.setColor(Color.black);
                    } else
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]", namePos,
                                yCoord);
                    yCoord = yCoord + (spacing);
                }
            }
        } else if (windowType == 2) {
            Iterator l1 = null;
            Iterator l2 = null;
            Iterator l3 = null;
            TrialData gm = trial.getTrialData();
            CallPathDrawObject callPathDrawObject = null;
            String s = null;

            //######
            //Populate drawObjectsComplete vector.
            //This should only happen once.
            //######

            if (drawObjectsComplete == null) {
                drawObjectsComplete = new Vector(); //Add five spacer
                // objects representing
                // the column headings.
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));

                l1 = window.getDataIterator();
                while (l1.hasNext()) {
                    Function function = (Function) l1.next(); //Don't draw
                    // callpath
                    // mapping objects.
                    if (!(function.isCallPathObject())) {
                        l2 = function.getParents();
                        while (l2.hasNext()) {
                            Function parent = (Function) l2.next();
                            callPathDrawObject = new CallPathDrawObject(parent, true, false, false);
                            drawObjectsComplete.add(callPathDrawObject);
                        }
                        callPathDrawObject = new CallPathDrawObject(function, false, false, false);
                        drawObjectsComplete.add(callPathDrawObject);
                        l2 = function.getChildren();
                        while (l2.hasNext()) {
                            Function child = (Function) l2.next();
                            callPathDrawObject = new CallPathDrawObject(child, false, true, false);
                            drawObjectsComplete.add(callPathDrawObject);
                        }
                        drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                        drawObjectsComplete.add(new CallPathDrawObject(null, false, false, true));
                    }
                }
            }

            //######
            //End - Populate drawObjectsComplete vector.
            //######
            //######
            //Populate drawObjects vector.
            //######
            if (drawObjects == null) {
                drawObjects = new Vector();
                Vector holdingPattern = new Vector();
                boolean adding = false;
                int state = -1;
                int size = -1;
                if (window.showCollapsedView()) {
                    for (Enumeration e = drawObjectsComplete.elements(); e.hasMoreElements();) {
                        callPathDrawObject = (CallPathDrawObject) e.nextElement();
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
                } else
                    drawObjects = drawObjectsComplete;
            }
            //######
            //End - Populate drawObjects vector.
            //######

            //######
            //Set panel size.
            //######
            if (this.calculatePanelSize()) {
                for (Enumeration e = drawObjects.elements(); e.hasMoreElements();) {
                    callPathDrawObject = (CallPathDrawObject) e.nextElement();
                    yHeightNeeded = yHeightNeeded + (spacing);
                    if (!callPathDrawObject.isSpacer()) {
                        length = fmMonoFont.stringWidth(callPathDrawObject.getName()) + 10;
                        if (xWidthNeeded < length)
                            xWidthNeeded = length;
                    }
                }

                base = 20;
                startPosition = fmMonoFont.stringWidth("--> ") + base;

                xWidthNeeded = xWidthNeeded + 20;

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
                if (sizeChange && toScreen)
                    revalidate();
                this.setCalculatePanelSize(false);
            }
            //######
            //End - Set panel size.
            //######

            int yBeg = 0;
            int yEnd = 0;
            int startElement = 0;
            int endElement = 0;
            Rectangle clipRect = null;
            Rectangle viewRect = null;

            if (!fullWindow) {
                if (toScreen) {
                    clipRect = g2D.getClipBounds();
                    yBeg = (int) clipRect.getY();
                    yEnd = (int) (yBeg + clipRect.getHeight());
                } else {
                    viewRect = window.getViewRect();
                    yBeg = (int) viewRect.getY();
                    yEnd = (int) (yBeg + viewRect.getHeight());
                }
                startElement = ((yBeg - yCoord) / spacing) - 1;
                endElement = ((yEnd - yCoord) / spacing) + 1;

                if (startElement < 0)
                    startElement = 0;

                if (endElement < 0)
                    endElement = 0;

                if (startElement > (drawObjects.size() - 1))
                    startElement = (drawObjects.size() - 1);

                if (endElement > (drawObjects.size() - 1))
                    endElement = (drawObjects.size() - 1);

                if (toScreen)
                    yCoord = yCoord + (startElement * spacing);
            } else {
                startElement = 0;
                endElement = ((drawObjects.size()) - 1);
            }

            g2D.setColor(Color.black);
            //######
            //Draw the header if required.
            //######
            if (drawHeader) {
                FontRenderContext frc = g2D.getFontRenderContext();
                Insets insets = this.getInsets();
                yCoord = yCoord + (spacing);
                String headerString = window.getHeaderString();
                //Need to split the string up into its separate lines.
                StringTokenizer st = new StringTokenizer(headerString, "'\n'");
                while (st.hasMoreTokens()) {
                    AttributedString as = new AttributedString(st.nextToken());
                    as.addAttribute(TextAttribute.FONT, monoFont);
                    AttributedCharacterIterator aci = as.getIterator();
                    LineBreakMeasurer lbm = new LineBreakMeasurer(aci, frc);
                    float wrappingWidth = this.getSize().width - insets.left - insets.right;
                    float x = insets.left;
                    float y = insets.right;
                    while (lbm.getPosition() < aci.getEndIndex()) {
                        TextLayout textLayout = lbm.nextLayout(wrappingWidth);
                        yCoord += spacing;
                        textLayout.draw(g2D, x, yCoord);
                        x = insets.left;
                    }
                }
                lastHeaderEndPosition = yCoord;
            }
            //######
            //End - Draw the header if required.
            //######
            for (int i = startElement; i <= endElement; i++) {
                callPathDrawObject = (CallPathDrawObject) drawObjects.elementAt(i);
                if (i == 1) {
                    g2D.drawString("Name[id]", startPosition, yCoord);
                    yCoord = yCoord + spacing;
                } else if (i == 2) {
                    g2D.drawString(
                            "--------------------------------------------------------------------------------",
                            startPosition, yCoord);
                    yCoord = yCoord + spacing;
                } else if (!callPathDrawObject.isParentChild() && !callPathDrawObject.isSpacer()) {
                    g2D.drawString("--> ", base, yCoord);

                    Function function = callPathDrawObject.getFunction();
                    if (trial.getColorChooser().getHighlightedFunction() == function) {
                        g2D.setColor(Color.red);
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]",
                                startPosition, yCoord);
                        g2D.setColor(Color.black);
                    } else
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]",
                                startPosition, yCoord);
                    yCoord = yCoord + (spacing);
                } else if (callPathDrawObject.isSpacer())
                    yCoord = yCoord + spacing;
                else {

                    Function function = callPathDrawObject.getFunction();
                    if (trial.getColorChooser().getHighlightedFunction() == function) {
                        g2D.setColor(Color.red);
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]",
                                startPosition, yCoord);
                        g2D.setColor(Color.black);
                    } else
                        g2D.drawString(callPathDrawObject.getName() + "[" + function.getID() + "]",
                                startPosition, yCoord);
                    yCoord = yCoord + (spacing);
                }
            }
        }
    }

    private double setMax(double max, double d1, double d2) {
        if (max < d1)
            max = d1;
        if (max < d2)
            max = d2;
        return max;
    }


    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            CallPathDrawObject callPathDrawObject = null;

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Show Function Details")) {
                    if (clickedOnObject instanceof CallPathDrawObject) {
                        callPathDrawObject = (CallPathDrawObject) clickedOnObject;
                        //Bring up an expanded data window for this mapping,
                        // and set this mapping as highlighted.
                        trial.getColorChooser().setHighlightedFunction(callPathDrawObject.getFunction());
                        FunctionDataWindow functionDataWindow = new FunctionDataWindow(trial,
                                callPathDrawObject.getFunction(), trial.getStaticMainWindow().getDataSorter());
                        trial.getSystemEvents().addObserver(functionDataWindow);
                        functionDataWindow.show();
                    }
                } else if (arg.equals("Find Function")) {
                    if (clickedOnObject instanceof CallPathDrawObject) {
                        Function function = ((CallPathDrawObject) clickedOnObject).getFunction();
                        int size = drawObjects.size();
                        for (int i = 0; i < size; i++) {
                            callPathDrawObject = (CallPathDrawObject) drawObjects.elementAt(i);
                            if ((callPathDrawObject.getFunction() == function)
                                    && (!callPathDrawObject.isParentChild())) {
                                Dimension dimension = window.getViewportSize();
                                window.setVerticalScrollBarPosition((i * (trial.getPreferences().getBarSpacing()))
                                        - ((int) dimension.getHeight() / 2));
                                trial.getColorChooser().setHighlightedFunction(function);
                                return;
                            }
                        }
                    }
                } else if (arg.equals("Change Function Color")) {
                    if (clickedOnObject instanceof CallPathDrawObject) {
                        Function function = ((CallPathDrawObject) clickedOnObject).getFunction();
                        Color color = function.getColor();
                        color = JColorChooser.showDialog(this, "Please select a new color", color);
                        if (color != null) {
                            function.setSpecificColor(color);
                            function.setColorFlag(true);
                            trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                        }
                    }
                } else if (arg.equals("Reset to Generic Color")) {
                    if (clickedOnObject instanceof CallPathDrawObject) {
                        Function function = ((CallPathDrawObject) clickedOnObject).getFunction();
                        function.setColorFlag(false);
                        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                    }
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void mouseClicked(MouseEvent evt) {
        try {
            //Get the location of the mouse.
            int xCoord = evt.getX();
            int yCoord = evt.getY();

            //Get the number of times clicked.
            int clickCount = evt.getClickCount();

            CallPathDrawObject callPathDrawObject = null;

            //Calculate which CallPathDrawObject was clicked on.
            int index = (yCoord - 1) / (trial.getPreferences().getBarSpacing()) + 1;

            if (index < drawObjects.size()) {
                callPathDrawObject = (CallPathDrawObject) drawObjects.elementAt(index);
                if (!callPathDrawObject.isSpacer()) {
                    if ((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0) {
                        clickedOnObject = callPathDrawObject;
                        popup.show(this, evt.getX(), evt.getY());
                        return;
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
                        trial.getColorChooser().toggleHighlightedFunction(callPathDrawObject.getFunction());
                    }
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void mousePressed(MouseEvent evt) {
    }

    public void mouseReleased(MouseEvent evt) {
    }

    public void mouseEntered(MouseEvent evt) {
    }

    public void mouseExited(MouseEvent evt) {
    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        Dimension d = null;
        if (fullScreen)
            d = this.getSize();
        else
            d = window.getSize();
        d.setSize(d.getWidth(), d.getHeight() + lastHeaderEndPosition);
        return d;
    }


    public void resetAllDrawObjects() {
        drawObjectsComplete.clear();
        drawObjectsComplete = null;
        drawObjects.clear();
        drawObjects = null;
    }

    private void setCalculatePanelSize(boolean calculatePanelSize) {
        this.calculatePanelSize = calculatePanelSize;
    }

    private boolean calculatePanelSize() {
        return calculatePanelSize;
    };

    public Dimension getPreferredSize() {
        return new Dimension(xPanelSize, (yPanelSize + 10));
    }

    
    
    //Instance data.
    int xPanelSize = 800;
    int yPanelSize = 600;
    boolean calculatePanelSize = true;

    edu.uoregon.tau.dms.dss.Thread thread;
    
    private ParaProfTrial trial = null;
    CallPathTextWindow window = null;
    int windowType = 0; //0: mean data,1: function data, 2: global relations.
    Font monoFont = null;
    FontMetrics fmMonoFont = null;

    //Some drawing details.
    Vector drawObjectsComplete = null;
    Vector drawObjects = null;
    int startLocation = 0;
    int maxFontAscent = 0;
    int maxFontDescent = 0;
    int spacing = 0;

    int check = 0;
    int base = 0;
    int startPosition = 0;
    int stringWidth = 0;
    int numCallsWidth = 0;
    int excPos = 0;
    int incPos = 0;
    int callsPos1 = 0;
    int callsPos2 = 0;
    int namePos = 0;
    double max = 0.0;
    int yHeightNeeded = 0;
    int xWidthNeeded = 0;
    int length = 0;

    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    private int lastHeaderEndPosition = 0;
}