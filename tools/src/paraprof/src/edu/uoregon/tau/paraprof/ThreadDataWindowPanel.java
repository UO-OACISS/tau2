/*
 * 
 * ThreadDataWindowPanel.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
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

public class ThreadDataWindowPanel extends JPanel implements ActionListener, MouseListener, Printable,
        ParaProfImageInterface {

    public ThreadDataWindowPanel(ParaProfTrial trial, int nodeID, int contextID, int threadID,
            ThreadDataWindow tDWindow) {
        setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
        setBackground(Color.white);

        //Add this object as a mouse listener.
        addMouseListener(this);

        this.trial = trial;
        this.window = tDWindow;
        barLength = baseBarLength;

        if (nodeID == -1) {
            thread = trial.getDataSource().getMeanData();
        } else {
            thread = trial.getDataSource().getThread(nodeID, contextID, threadID);
        }

        //if (windowType == 1)
        //thread = trial.getNCT().getThread(nodeID, contextID, threadID);

        //Add items to the popu menu.
        JMenuItem functionDetailsItem = new JMenuItem("Show Function Details");
        functionDetailsItem.addActionListener(this);
        popup.add(functionDetailsItem);

        JMenuItem changeColorItem = new JMenuItem("Change Function Color");
        changeColorItem.addActionListener(this);
        popup.add(changeColorItem);

        JMenuItem resetColorItem = new JMenuItem("Reset to Generic Color");
        resetColorItem.addActionListener(this);
        popup.add(resetColorItem);

        //Schedule a repaint of this panel.
        this.repaint();
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

    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader)
            throws ParaProfException {
        list = window.getData();

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

        // With group support present, it is possible that the number of
        // functions in our data list is zero. This can occur when the user's selected
        // groups to display are not present on this thread ... for example. If so, just return.

        if ((list.size()) == 0)
            return;

        //To make sure the bar details are set, this method must be called.
        trial.getPreferences().setBarDetails(g2D);

        //Now safe to grab spacing and bar heights.
        barSpacing = trial.getPreferences().getBarSpacing();
        barHeight = trial.getPreferences().getBarHeight();

        //Obtain the font and its metrics.
        Font font = new Font(trial.getPreferences().getParaProfFont(), trial.getPreferences().getFontStyle(),
                barHeight);
        g2D.setFont(font);
        FontMetrics fmFont = g2D.getFontMetrics(font);

        //######
        //Set max values.
        //######

        //maxValue = ParaProfUtils.getMaxThreadValue(thread, window.getValueType(), window.isPercent(), trial);

        maxValue = window.getPPThread().getMaxValue(window.getValueType(), window.isPercent());

        
        if (window.isPercent()) {
            stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(0, maxValue, 6) + "%");
            barXCoord = barXCoord + stringWidth;
        } else {
            stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(window.units(), maxValue,
                    ParaProf.defaultNumberPrecision));
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
                viewRect = window.getViewRect();
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

        //Draw the header if required.
        if (drawHeader) {
            FontRenderContext frc = g2D.getFontRenderContext();
            Insets insets = this.getInsets();
            yCoord = yCoord + (barSpacing);
            String headerString = window.getHeaderString();
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

        // Iterate through each function and draw it's bar
        for (int i = startElement; i <= endElement; i++) {
            ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);

            value = ParaProfUtils.getValue(ppFunctionProfile, window.getValueType(), window.isPercent());

            yCoord = yCoord + (barSpacing);
            drawBar(g2D, fmFont, value, maxValue, barXCoord, yCoord, barHeight, ppFunctionProfile, toScreen);
        }
    }

    private void drawBar(Graphics2D g2D, FontMetrics fmFont, double value, double maxValue, int barXCoord,
            int yCoord, int barHeight, PPFunctionProfile ppFunctionProfile, boolean toScreen) {
        int xLength = 0;
        double d = 0.0;
        String s = null;
        int stringWidth = 0;
        int stringStart = 0;

        Function f = ppFunctionProfile.getFunction();
        String functionName = ppFunctionProfile.getFunctionName();
        boolean groupMember = ppFunctionProfile.isGroupMember(trial.getColorChooser().getHighlightedGroup());

        d = (value / maxValue);
        xLength = (int) (d * barLength);
        if (xLength == 0)
            xLength = 1;

        if ((xLength > 2) && (barHeight > 2)) {
            g2D.setColor(ppFunctionProfile.getColor());
            g2D.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);

            if (f == (trial.getColorChooser().getHighlightedFunction())) {
                g2D.setColor(trial.getColorChooser().getHighlightColor());
                g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                g2D.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            } else if (groupMember) {
                g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
                g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                g2D.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
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
        if (window.isPercent() && window.getValueType() <= 4) {
            //s = (UtilFncs.adjustDoublePresision(value, ParaProf.defaultNumberPrecision)) + "%";
            //s = (UtilFncs.adjustDoublePresision(value, 4)) + "%";
            s = UtilFncs.getOutputString(0, value, 6) + "%";

        } else
            s = UtilFncs.getOutputString(window.units(), value, ParaProf.defaultNumberPrecision);
        stringWidth = fmFont.stringWidth(s);
        //Now draw the percent value to the left of the bar.
        stringStart = barXCoord - xLength - stringWidth - 5;
        g2D.drawString(s, stringStart, yCoord);

        g2D.drawString(functionName, (barXCoord + 5), yCoord);

        stringWidth = fmFont.stringWidth(functionName);
        //Update the drawing coordinates if we are drawing to the screen.
        if (toScreen)
            ppFunctionProfile.setDrawCoords(stringStart, (barXCoord + 5 + stringWidth), (yCoord - barHeight),
                    yCoord);
    }

    public void changeInMultiples() {
        computeBarLength();
        this.repaint();
    }

    public void computeBarLength() {
        double sliderValue = (double) window.getSliderValue();
        double sliderMultiple = window.getSliderMultiple();
        barLength = baseBarLength * ((int) (sliderValue * sliderMultiple));
    }

    public void actionPerformed(ActionEvent evt) {

        try {
            Object EventSrc = evt.getSource();
            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Show Function Details")) {
                    if (clickedOnObject instanceof Function) {
                        Function function = (Function) clickedOnObject;
                        trial.getColorChooser().setHighlightedFunction(function);
                        FunctionDataWindow functionDataWindow = new FunctionDataWindow(trial, function,
                                window.getDataSorter());
                        trial.getSystemEvents().addObserver(functionDataWindow);
                        functionDataWindow.show();
                    }
                } else if (arg.equals("Change Function Color")) {
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
                    if (clickedOnObject instanceof Function) {
                        Function function = (Function) clickedOnObject;
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
                    trial.getColorChooser().toggleHighlightedFunction(ppFunctionProfile.getFunction());
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

    //This method sets both xPanelSize and yPanelSize.
    private boolean resizePanel(FontMetrics fmFont, int barXCoord, Vector list, int startElement, int endElement) {
        boolean resized = false;
        int newXPanelSize = 0;
        int newYPanelSize = 0;
        int width = 0;
        int height = 0;
        PPFunctionProfile ppFunctionProfile = null;

        for (int i = startElement; i <= endElement; i++) {
            ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);
            //As a temporary fix, at 500 pixels to the end.
            width = barXCoord + 5 + (fmFont.stringWidth(ppFunctionProfile.getFunctionName())) + 500;
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
        return resized;
    }

    public Dimension getPreferredSize() {
        return new Dimension(xPanelSize, (yPanelSize + 10));
    }

    //Instance data.
    private int xPanelSize = 640;
    private int yPanelSize = 480;

    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 250;
    private int barLength = 0;
    private int textOffset = 60;

    private ParaProfTrial trial = null;
    private ThreadDataWindow window = null;
    private edu.uoregon.tau.dms.dss.Thread thread = null;
    private Vector list = null;

    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    private int lastHeaderEndPosition = 0;

}