/**
 * HistogramWindowPanel
 * This is the panel for the HistogramWindow.
 *  
 * <P>CVS $Id: HistogramWindowPanel.java,v 1.3 2004/12/24 00:25:08 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.3 $
 * @see		HistogramWindow
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.print.*;
import java.awt.geom.*;
//import javax.print.*;
import edu.uoregon.tau.dms.dss.*;

public class HistogramWindowPanel extends JPanel implements ActionListener, MouseListener, PopupMenuListener,
        Printable, ParaProfImageInterface {

    public HistogramWindowPanel() {
        try {
            //Set the default tool tip for this panel.
            this.setToolTipText("Incorrect Constructor!!!");
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SMWP01");
        }
    }

    public HistogramWindowPanel(ParaProfTrial trial, HistogramWindow bWindow, Function function) {
        try {
            //Set the default tool tip for this panel.
            this.setToolTipText("ParaProf bar graph draw window!");
            setBackground(Color.white);

            this.trial = trial;
            this.bWindow = bWindow;
            this.function = function;

            //Add this object as a mouse listener.
            addMouseListener(this);

            barXStart = 100;

            //Add items to the first popup menu.
            JMenuItem mappingDetailsItem = new JMenuItem("Show Function Details");
            mappingDetailsItem.addActionListener(this);
            popup.add(mappingDetailsItem);

            JMenuItem changeColorItem = new JMenuItem("Change Function Color");
            changeColorItem.addActionListener(this);
            popup.add(changeColorItem);

            JMenuItem maskMappingItem = new JMenuItem("Reset to Generic Color");
            maskMappingItem.addActionListener(this);
            popup.add(maskMappingItem);

            JMenuItem highlightMappingItem = new JMenuItem("Highlight this Function");
            highlightMappingItem.addActionListener(this);
            popup.add(highlightMappingItem);

            JMenuItem unHighlightMappingItem = new JMenuItem("Un-Highlight this Function");
            unHighlightMappingItem.addActionListener(this);
            popup.add(unHighlightMappingItem);

            //Add items to the second popup menu.
            popup2.addPopupMenuListener(this);

            JMenuItem tSWItem = new JMenuItem("Show Statistics Window");
            tSWItem.addActionListener(this);
            popup2.add(tSWItem);

            if (trial.userEventsPresent()) {
                tUESWItem = new JMenuItem("Show User Event Statistics Window");
                tUESWItem.addActionListener(this);
            }

            popup2.add(tUESWItem);
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SMWP02");
        }
    }

    public String getToolTipText(MouseEvent evt) {
        return null;
    }

    public void actionPerformed(ActionEvent evt) {
    }

    public void mouseClicked(MouseEvent evt) {
    }

    public void mousePressed(MouseEvent evt) {
    }

    public void mouseReleased(MouseEvent evt) {
    }

    public void mouseEntered(MouseEvent evt) {
    }

    public void mouseExited(MouseEvent evt) {
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

        g2.translate((int) pageFormat.getImageableX(), (int) pageFormat.getImageableY());
        g2.translate(tx, ty);
        g2.scale(scale, scale);

        renderIt(g2, false, true, false);

        return Printable.PAGE_EXISTS;
    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        return this.getSize();
    }

    private double getMaxValue(Function function) {
        double maxValue = 0;
        switch (bWindow.getValueType()) {
        case 2:
            maxValue = function.getMaxExclusive(trial.getSelectedMetricID());
            break;
        case 4:
            maxValue = function.getMaxInclusive(trial.getSelectedMetricID());
            break;
        case 6:
            maxValue = function.getMaxNumCalls();
            break;
        case 8:
            maxValue = function.getMaxNumSubr();
            break;
        case 10:
            maxValue = function.getMaxInclusivePerCall(trial.getSelectedMetricID());
            break;
        default:
            UtilFncs.systemError(null, null, "Unexpected type - MDWP value: " + bWindow.getValueType());
        }
        return maxValue;
    }

    private double getValue(PPFunctionProfile ppFunctionProfile) {
        double value = 0;
        switch (bWindow.getValueType()) {
        case 2:
            value = ppFunctionProfile.getExclusiveValue();
            break;
        case 4:
            value = ppFunctionProfile.getInclusiveValue();
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
            UtilFncs.systemError(null, null, "Unexpected type - MDWP value: " + bWindow.getValueType());
        }
        return value;
    }

    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        try {

            list = bWindow.getData();

            //Check to see if selected groups only are being displayed.
            TrialData td = trial.getTrialData();

            //**********
            //Other initializations.
            highlighted = false;
            xCoord = yCoord = 0;
            //End - Other initializations.
            //**********

            //**********
            //Do the standard font and spacing stuff.
            if (!(trial.getPreferences().areBarDetailsSet())) {

                //Create font.
                Font font = new Font(trial.getPreferences().getParaProfFont(),
                        trial.getPreferences().getFontStyle(), 12);
                g2D.setFont(font);
                FontMetrics fmFont = g2D.getFontMetrics(font);

                //Set up the bar details.

                //Compute the font metrics.
                int maxFontAscent = fmFont.getAscent();
                int maxFontDescent = fmFont.getMaxDescent();

                int tmpInt = maxFontAscent + maxFontDescent;

                trial.getPreferences().setBarDetails(g2D);

                // trial.getPreferences().setSliders(maxFontAscent, (tmpInt +
                // 5));
            }
            //End - Do the standard font and spacing stuff.
            //**********

            //Set local spacing and bar heights.
            int barSpacing = trial.getPreferences().getBarSpacing();
            int barHeight = trial.getPreferences().getBarHeight();

            //Create font.
            Font font = new Font(trial.getPreferences().getParaProfFont(),
                    trial.getPreferences().getFontStyle(), barHeight);
            g2D.setFont(font);
            FontMetrics fmFont = g2D.getFontMetrics(font);

            //**********
            //Calculating the starting positions of drawing.
            String tmpString2 = new String("n,c,t 99,99,99");
            int stringWidth = fmFont.stringWidth(tmpString2);
            barXStart = stringWidth + 15;
            int tmpXWidthCalc = barXStart + defaultBarLength;
            int barXCoord = barXStart;
            yCoord = yCoord + (barSpacing);
            //End - Calculating the starting positions of drawing.
            //**********

            Rectangle clipRect = g2D.getClipBounds();

            int yBeg = (int) clipRect.getY();
            int yEnd = (int) (yBeg + clipRect.getHeight());
            //Because tooltip redraw can louse things up. Add an extra one to draw.

            yEnd = yEnd + barSpacing;

            yCoord = yCoord + (barSpacing);

            //Set the drawing color to the text color ... in this case,
            // black.
            g2D.setColor(Color.black);

            g2D.drawLine(35, 430, 35, 30);
            g2D.drawLine(35, 430, 585, 430);

            double maxValue = 0;
            double minValue = 0;
            boolean start = true;
            PPFunctionProfile ppFunctionProfile = null;

            for (Enumeration e1 = list.elements(); e1.hasMoreElements();) {

                ppFunctionProfile = (PPFunctionProfile) e1.nextElement();

                if (ppFunctionProfile.getFunction() == function) {
                    double tmpDataValue = getValue(ppFunctionProfile);
                    if (start) {
                        minValue = tmpDataValue;
                        start = false;
                    }
                    if (tmpDataValue > maxValue)
                        maxValue = tmpDataValue;
                    if (tmpDataValue < minValue)
                        minValue = tmpDataValue;
                }
            }

            double increment = maxValue / 10;

            for (int i = 0; i < 10; i++) {
                g2D.drawLine(30, 30 + i * 40, 35, 30 + i * 40);
                g2D.drawString("" + (10 * (10 - i)), 5, 33 + i * 40);
            }

            for (int i = 1; i < 11; i++) {
                g2D.drawLine(35 + i * 55, 430, 35 + i * 55, 435);
            }

            
            
            
            g2D.drawString("Min Value = " + UtilFncs.getOutputString(bWindow.units(), minValue, 6), 35, 450);
            g2D.drawString("Max Value = " + UtilFncs.getOutputString(bWindow.units(), maxValue, 6), 552, 450);

            xPanelSize = 552 + fmFont.stringWidth("Max Value = " + maxValue);

            int[] intArray = new int[10];

            for (int i = 0; i < 10; i++) {
                intArray[i] = 0;
            }

            int count = 0;

            int numBins = 10;

            double binWidth = (maxValue - minValue) / numBins;

            for (Enumeration e1 = list.elements(); e1.hasMoreElements();) {
                ppFunctionProfile = (PPFunctionProfile) e1.nextElement();
                if (ppFunctionProfile.getFunction() == function) {
                    double tmpDataValue = getValue(ppFunctionProfile);
                    for (int j = 0; j < 10; j++) {
                        if (tmpDataValue <= (minValue + (binWidth * (j + 1)))) {
                            intArray[j]++;
                            count++;
                            break;
                        }
                    }
                }
            }

            g2D.setColor(Color.red);

            int num = count;
            for (int i = 0; i < 10; i++) {
                if (intArray[i] != 0) {
                    double tmp1 = intArray[i];

                    double per = (tmp1 / num) * 100;
                    int result = (int) per;
                    g2D.fillRect(38 + i * 55, 430 - (result * 4), 49, result * 4);
                }
            }

            boolean sizeChange = false;
            //Resize the panel if needed.
            //if (tmpXWidthCalc > 600) {
            //    xPanelSize = tmpXWidthCalc + 1;
            //    sizeChange = true;
            // }

            // hmm
            yCoord = 450;
            if (yCoord > 300) {
                yPanelSize = yCoord + 1;
                sizeChange = true;
            }

            if (sizeChange)
                revalidate();
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SMWP06");
        }
    }

    public void popupMenuWillBecomeVisible(PopupMenuEvent evt) {
        try {
            if (trial.userEventsPresent()) {
                tUESWItem.setEnabled(true);
            } else {
                tUESWItem.setEnabled(false);
            }
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SMW03");
        }
    }

    public void popupMenuWillBecomeInvisible(PopupMenuEvent evt) {
    }

    public void popupMenuCanceled(PopupMenuEvent evt) {
    }

    public Dimension getPreferredSize() {
        return new Dimension(xPanelSize + 10, (yPanelSize + 10));
    }

    //####################################
    //Instance data.
    //####################################
    private ParaProfTrial trial = null;
    HistogramWindow bWindow = null;
    int xPanelSize = 600;
    int yPanelSize = 400;

    Function function = null;

    private Vector list = null;

    //**********
    //Popup menu definitions.
    private JPopupMenu popup = new JPopupMenu();
    private JPopupMenu popup2 = new JPopupMenu();

    JMenuItem tUESWItem = new JMenuItem("Show Total User Event Statistics Windows");

    //**********

    //**********
    //Some place holder definitions - used for cycling through data lists.
    Vector contextList = null;
    Vector threadList = null;
    Vector threadDataList = null;
    PPThread ppThread = null;
    PPFunctionProfile ppFunctionProfile = null;
    //End - Place holder definitions.
    //**********

    //**********
    //Other useful variables for getToolTipText, mouseEvents, and
    // paintComponent.
    int xCoord = -1;
    int yCoord = -1;
    Object clickedOnObject = null;
    //End - Other useful variables for getToolTipText, mouseEvents, and
    // paintComponent.
    //**********

    //**********
    //Some misc stuff for the paintComponent function.
    String counterName = null;
    private int defaultBarLength = 500;
    String tmpString = null;
    double tmpSum = -1;
    double tmpDataValue = -1;
    Color tmpColor = null;
    boolean highlighted = false;
    int barXStart = -1;
    int numberOfColors = 0;
    //End - Some misc stuff for the paintComponent function.
    //**********

}