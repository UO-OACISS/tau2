/*
 * 
 * ThreadDataWindowPanel.java
 * 
 * Title: ParaProf Author: Robert Bell Description:
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import javax.swing.*;
import java.awt.geom.*;
import java.text.*;
import java.awt.font.*;
import edu.uoregon.tau.dms.dss.*;

public class ThreadDataWindowPanel extends JPanel implements ActionListener, MouseListener,
        Printable, ParaProfImageInterface {

    public ThreadDataWindowPanel() {

        try {
            setSize(new java.awt.Dimension(xPanelSize, yPanelSize));

            //Schedule a repaint of this panel.
            this.repaint();
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "TDWP01");
        }

    }

    public ThreadDataWindowPanel(ParaProfTrial trial, int nodeID, int contextID, int threadID,
            ThreadDataWindow tDWindow, boolean debug) {
        try {
            setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
            setBackground(Color.white);

            //Add this object as a mouse listener.
            addMouseListener(this);

            this.nodeID = nodeID;
            this.contextID = contextID;
            this.threadID = threadID;
            this.trial = trial;
            this.tDWindow = tDWindow;
            this.debug = debug;
            barLength = baseBarLength;

            if (nodeID == -1) {
                thread = trial.getDataSource().getMeanData();
            } else {
                thread = trial.getNCT().getThread(nodeID, contextID, threadID);
            }

            //if (windowType == 1)
            //thread = trial.getNCT().getThread(nodeID, contextID, threadID);

            //######
            //Add items to the popu menu.
            //######
            JMenuItem mappingDetailsItem = new JMenuItem("Show Function Details");
            mappingDetailsItem.addActionListener(this);
            popup.add(mappingDetailsItem);

            JMenuItem changeColorItem = new JMenuItem("Change Function Color");
            changeColorItem.addActionListener(this);
            popup.add(changeColorItem);

            JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
            maskMappingItem.addActionListener(this);
            popup.add(maskMappingItem);
            //######
            //End - Add items to the popu menu.
            //######

            //Schedule a repaint of this panel.
            this.repaint();
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "TDWP02");
        }
    }

    public void paintComponent(Graphics g) {
        try {
            super.paintComponent(g);
            renderIt((Graphics2D) g, true, false, false);
        } catch (Exception e) {
            System.out.println(e);
            UtilFncs.systemError(e, null, "TDWP03");
        }
    }

    public int print(Graphics g, PageFormat pageFormat, int page) {
        if (page >= 1) {
            return NO_SUCH_PAGE;
        }
        
        double pageWidth = pageFormat.getImageableWidth();
        double pageHeight = pageFormat.getImageableHeight();
        int cols = (int) (xPanelSize / pageWidth) + 1;
        int rows = (int) (yPanelSize / pageHeight) + 1;
        double xScale = pageWidth / xPanelSize;
        double yScale = pageHeight / yPanelSize;
        double scale = Math.min(xScale, yScale);
        
        double tx = 0.0;
        double ty = 0.0;
        if (xScale > scale) {
            tx = 0.5 * (xScale - scale) * xPanelSize;
        } else {
            ty = 0.5 * (yScale - scale) * yPanelSize;
        }

        Graphics2D g2 = (Graphics2D) g;

        g2.translate((int) pageFormat.getImageableX(),
                (int) pageFormat.getImageableY());
        g2.translate(tx, ty);
        g2.scale(scale, scale);

        renderIt(g2, false, true, false);

        return Printable.PAGE_EXISTS;

    }

    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {

        try {
            list = tDWindow.getData();

            //######
            //Some declarations.
            //######
            double value = 0.0;
            double maxValue = 0.0;
            int stringWidth = 0;
            int yCoord = 0;
            int barXCoord = barLength + textOffset;
            PPFunctionProfile ppFunctionProfile = null;
            //######
            //End - Some declarations.
            //######

            //With group support present, it is possible that the number of
            // mappings in
            //our data list is zero. This can occur when the user's selected
            // groups to display are
            //not present on this thread ... for example. If so, just return.
            if ((list.size()) == 0)
                return;

            //To make sure the bar details are set, this
            //method must be called.
            trial.getPreferences().setBarDetails(g2D);

            //Now safe to grab spacing and bar heights.
            barSpacing = trial.getPreferences().getBarSpacing();
            barHeight = trial.getPreferences().getBarHeight();

            //Obtain the font and its metrics.
            Font font = new Font(trial.getPreferences().getParaProfFont(),
                    trial.getPreferences().getFontStyle(), barHeight);
            g2D.setFont(font);
            FontMetrics fmFont = g2D.getFontMetrics(font);

            //######
            //Set max values.
            //######

            switch (tDWindow.getValueType()) {
            case 2:
                if (tDWindow.isPercent())
                    maxValue = thread.getMaxExclusivePercent(trial.getSelectedMetricID());
                else
                    maxValue = thread.getMaxExclusive(trial.getSelectedMetricID());
                break;
            case 4:
                if (tDWindow.isPercent())
                    maxValue = thread.getMaxInclusivePercent(trial.getSelectedMetricID());
                else
                    maxValue = thread.getMaxInclusive(trial.getSelectedMetricID());
                break;
            case 6:
                maxValue = thread.getMaxNumCalls();
                break;
            case 8:
                maxValue = thread.getMaxNumSubr();
                break;
            case 10:
                maxValue = thread.getMaxInclusivePerCall(trial.getSelectedMetricID());
                break;
            default:
                UtilFncs.systemError(null, null, "Unexpected type - MDWP value: "
                        + tDWindow.getValueType());
            }


            if (tDWindow.isPercent()) {
                stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(0, maxValue, 6) + "%");
                barXCoord = barXCoord + stringWidth;
            } else {
                stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(tDWindow.units(),
                        maxValue, ParaProf.defaultNumberPrecision));
                barXCoord = barXCoord + stringWidth;
            }
            //######
            //End - Set max values.
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
                    viewRect = tDWindow.getViewRect();
                    yBeg = (int) viewRect.getY();
                    yEnd = (int) (yBeg + viewRect.getHeight());
                }
                startElement = ((yBeg - yCoord) / barSpacing) - 1;
                endElement = ((yEnd - yCoord) / barSpacing) + 1;

                if (startElement < 0)
                    startElement = 0;

                if (endElement < 0)
                    endElement = 0;

                if (startElement > (list.size() - 1))
                    startElement = (list.size() - 1);

                if (endElement > (list.size() - 1))
                    endElement = (list.size() - 1);

                if (toScreen)
                    yCoord = yCoord + (startElement * barSpacing);
            } else {
                startElement = 0;
                endElement = ((list.size()) - 1);
            }

            //At this point we can determine the size this panel will
            //require. If we need to resize, don't do any more drawing,
            //just call revalidate. Make sure we check the instruction value as
            // we only want to
            //revalidate if we are drawing to the screen.
            if (resizePanel(fmFont, barXCoord, list, startElement, endElement) && toScreen) {
                this.revalidate();
                return;
            }

            //######
            //Draw the header if required.
            //######
            if (drawHeader) {
                FontRenderContext frc = g2D.getFontRenderContext();
                Insets insets = this.getInsets();
                yCoord = yCoord + (barSpacing);
                String headerString = tDWindow.getHeaderString();
                //Need to split the string up into its separate lines.
                StringTokenizer st = new StringTokenizer(headerString, "'\n'");
                while (st.hasMoreTokens()) {
                    AttributedString as = new AttributedString(st.nextToken());
                    as.addAttribute(TextAttribute.FONT, font);
                    AttributedCharacterIterator aci = as.getIterator();
                    LineBreakMeasurer lbm = new LineBreakMeasurer(aci, frc);
                    float wrappingWidth = this.getSize().width - insets.left - insets.right;
                    float x = insets.left;
                    float y = insets.right;
                    while (lbm.getPosition() < aci.getEndIndex()) {
                        TextLayout textLayout = lbm.nextLayout(wrappingWidth);
                        yCoord += barSpacing;
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
                ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);
                    switch (tDWindow.getValueType()) {
                    case 2:
                        if (tDWindow.isPercent()) {
                            value = ppFunctionProfile.getExclusivePercentValue();
                        } else {
                            value = ppFunctionProfile.getExclusiveValue();
                        }
                        break;
                    case 4:
                        if (tDWindow.isPercent()) {
                            value = ppFunctionProfile.getInclusivePercentValue();
                        } else {
                            value = ppFunctionProfile.getInclusiveValue();
                        }
                        break;
                    case 6:
                        value = ppFunctionProfile.getNumberOfCalls();
                        break;
                    case 8:
                        value = ppFunctionProfile.getNumberOfSubRoutines();
                        break;
                    case 10:
                        value = ppFunctionProfile.getInclusivePerCall();
                        break;
                    default:
                        UtilFncs.systemError(null, null, "Unexpected type - MDWP value: "
                                + tDWindow.getValueType());
                    }

                yCoord = yCoord + (barSpacing);
                drawBar(g2D, fmFont, value, maxValue, barXCoord, yCoord, barHeight,
                        ppFunctionProfile, toScreen);
            }
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "TDWP04");
        }
    }

    private void drawBar(Graphics2D g2D, FontMetrics fmFont, double value, double maxValue,
            int barXCoord, int yCoord, int barHeight, PPFunctionProfile ppFunctionProfile, boolean toScreen
            ) {
        int xLength = 0;
        double d = 0.0;
        String s = null;
        int stringWidth = 0;
        int stringStart = 0;

        Function f = ppFunctionProfile.getFunction();
        String mappingName = ppFunctionProfile.getFunctionName();
        boolean groupMember = ppFunctionProfile.isGroupMember(trial.getColorChooser().getHighlightedGroup());

        d = (value / maxValue);
        xLength = (int) (d * barLength);
        if (xLength == 0)
            xLength = 1;

        if ((xLength > 2) && (barHeight > 2)) {
            g2D.setColor(ppFunctionProfile.getColor());
            g2D.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1,
                    barHeight - 1);

            if (f == (trial.getColorChooser().getHighlightedFunction())) {
                g2D.setColor(trial.getColorChooser().getHighlightColor());
                g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                g2D.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2,
                        barHeight - 2);
            } else if (groupMember) {
                g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
                g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                g2D.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2,
                        barHeight - 2);
            } else {
                g2D.setColor(Color.black);
                g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            }
        } else {
            if (f == (trial.getColorChooser().getHighlightedFunction()))
                g2D.setColor(trial.getColorChooser().getHighlightColor());
            else if (groupMember)
                g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
            else {
                g2D.setColor(ppFunctionProfile.getColor());
            }
            g2D.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
        }

        //Draw the value next to the bar.
        g2D.setColor(Color.black);
        //Do not want to put a percent sign after the bar if we are not
        // exclusive or inclusive.
        if (tDWindow.isPercent() && tDWindow.getValueType() <= 4) {
            //s = (UtilFncs.adjustDoublePresision(value, ParaProf.defaultNumberPrecision)) + "%";
        //s = (UtilFncs.adjustDoublePresision(value, 4)) + "%";
        s = UtilFncs.getOutputString(0, value, 6) + "%";

        
        } else
            s = UtilFncs.getOutputString(tDWindow.units(), value, ParaProf.defaultNumberPrecision);
        stringWidth = fmFont.stringWidth(s);
        //Now draw the percent value to the left of the bar.
        stringStart = barXCoord - xLength - stringWidth - 5;
        g2D.drawString(s, stringStart, yCoord);

        //Now draw the mapping to the right of the bar.
        g2D.drawString(mappingName, (barXCoord + 5), yCoord);

        //Grab the width of the mappingName.
        stringWidth = fmFont.stringWidth(mappingName);
        //Update the drawing coordinates if we are drawing to the screen.
        if (toScreen)
            ppFunctionProfile.setDrawCoords(stringStart, (barXCoord + 5 + stringWidth),
                    (yCoord - barHeight), yCoord);
    }

    public void changeInMultiples() {
        computeBarLength();
        this.repaint();
    }

    public void computeBarLength() {
        try {
            double sliderValue = (double) tDWindow.getSliderValue();
            double sliderMultiple = tDWindow.getSliderMultiple();
            barLength = baseBarLength * ((int) (sliderValue * sliderMultiple));
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "MDWP06");
        }
    }

    //####################################
    //Interface code.
    //####################################

    //######
    //ActionListener.
    //######
    public void actionPerformed(ActionEvent evt) {

        try {
            Object EventSrc = evt.getSource();
            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Show Function Details")) {
                    if (clickedOnObject instanceof Function) {
                        Function function = (Function) clickedOnObject;
                        //Bring up an expanded data window for this mapping,
                        // and set this mapping as highlighted.
                        trial.getColorChooser().setHighlightedFunction(function);
                        FunctionDataWindow functionDataWindow = new FunctionDataWindow(trial,
                                function, tDWindow.getDataSorter(), this.debug());
                        trial.getSystemEvents().addObserver(functionDataWindow);
                        functionDataWindow.show();
                    }
                } else if (arg.equals("Change Function Color")) {
                    //Get the clicked on object.
                    if (clickedOnObject instanceof Function) {
                        Function function = (Function) clickedOnObject;
                        Color color = function.getColor();
                        color = JColorChooser.showDialog(this, "Please select a new color", color);
                        if (color != null) {
                            function.setSpecificColor(color);
                            function.setColorFlag(true);
                            trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                        }
                    }
                } else if (arg.equals("Reset to Generic Color")) {
                    //Get the clicked on object.
                    if (clickedOnObject instanceof Function) {
                        Function function = (Function) clickedOnObject;
                        function.setColorFlag(false);
                        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                    }
                }
            }
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "TDWP04");
        }
    }

    //######
    //End - ActionListener.
    //######

    //######
    //MouseListener
    //######
    public void mouseClicked(MouseEvent evt) {
        try {
            //Get the location of the mouse.
            int xCoord = evt.getX();
            int yCoord = evt.getY();

            //Get the number of times clicked.
            int clickCount = evt.getClickCount();

            PPFunctionProfile ppFunctionProfile = null;

            //Calculate which PPFunctionProfile was clicked on.
            int index = (yCoord) / (trial.getPreferences().getBarSpacing());

            if (list != null && index < list.size()) {
                ppFunctionProfile = (PPFunctionProfile) list.elementAt(index);
                if ((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0) {
                    clickedOnObject = ppFunctionProfile.getFunction();
                    popup.show(this, evt.getX(), evt.getY());
                    return;
                } else {
                    trial.getColorChooser().toggleHighlightedFunction(
                            ppFunctionProfile.getFunction());
                }
            }
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "TDWP05");
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

    //######
    //End - MouseListener
    //######

    //######
    //ParaProfImageInterface
    //######
    public Dimension getImageSize(boolean fullScreen, boolean header) {
        Dimension d = null;
        if (fullScreen)
            d = this.getSize();
        else
            d = tDWindow.getSize();
        d.setSize(d.getWidth(), d.getHeight() + lastHeaderEndPosition);
        return d;
    }

    //######
    //End - ParaProfImageInterface
    //######

    //####################################
    //End - Interface code.
    //####################################

    //This method sets both xPanelSize and yPanelSize.
    private boolean resizePanel(FontMetrics fmFont, int barXCoord, Vector list, int startElement,
            int endElement) {
        boolean resized = false;
        try {
            int newXPanelSize = 0;
            int newYPanelSize = 0;
            int width = 0;
            int height = 0;
            PPFunctionProfile ppFunctionProfile = null;

            for (int i = startElement; i <= endElement; i++) {
                ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);
                //As a temporary fix, at 500 pixels to the end.
                width = barXCoord + 5
                        + (fmFont.stringWidth(ppFunctionProfile.getFunctionName())) + 500;
                if (width > newXPanelSize)
                    newXPanelSize = width;
            }

            newYPanelSize = barSpacing + ((list.size() + 1) * barSpacing);

            if ((newYPanelSize != yPanelSize) || (newXPanelSize != xPanelSize)) {
                yPanelSize = newYPanelSize;
                xPanelSize = newXPanelSize;
                this.setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
                resized = false;
            }
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "MDWP07");
        }
        return resized;
    }

    public Dimension getPreferredSize() {
        return new Dimension(xPanelSize, (yPanelSize + 10));
    }

    public void setDebug(boolean debug) {
        this.debug = debug;
    }

    public boolean debug() {
        return debug;
    }

    //####################################
    //Instance data.
    //####################################
    private int xPanelSize = 640;
    private int yPanelSize = 480;

    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 250;
    private int barLength = 0;
    private int textOffset = 60;

    private int maxXLength = 0;

    private ParaProfTrial trial = null;
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private ThreadDataWindow tDWindow = null;
    private edu.uoregon.tau.dms.dss.Thread thread = null;
    private Vector list = null;

    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    private int lastHeaderEndPosition = 0;

    private boolean debug = false; //Off by default.
    //####################################
    //Instance data.
    //####################################
}