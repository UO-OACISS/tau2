/**
 * HistogramWindowPanel
 * This is the panel for the HistogramWindow.
 *  
 * <P>CVS $Id: HistogramWindowPanel.java,v 1.4 2004/12/29 00:09:48 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.4 $
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

public class HistogramWindowPanel extends JPanel implements Printable, ParaProfImageInterface {

    public HistogramWindowPanel(ParaProfTrial trial, HistogramWindow window, Function function) {
        //Set the default tool tip for this panel.
        this.setToolTipText("ParaProf bar graph draw window!");
        setBackground(Color.white);

        this.trial = trial;
        this.window = window;
        this.function = function;

        //Add this object as a mouse listener.
        //addMouseListener(this);
        barXStart = 100;
    }

    public String getToolTipText(MouseEvent evt) {
        return null;
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

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        return this.getSize();
    }

   

    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) throws ParaProfException {
    
            list = window.getData();

            //Check to see if selected groups only are being displayed.
            TrialData td = trial.getTrialData();

            //**********
            //Other initializations.
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
                    double tmpDataValue = ParaProfUtils.getValue(ppFunctionProfile, window.getValueType(), false);
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

            g2D.drawString("Min Value = " + UtilFncs.getOutputString(window.units(), minValue, 6), 35, 450);
            g2D.drawString("Max Value = " + UtilFncs.getOutputString(window.units(), maxValue, 6), 552, 450);

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
                    double tmpDataValue = ParaProfUtils.getValue(ppFunctionProfile, window.getValueType(), false);
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
    }


    public Dimension getPreferredSize() {
        return new Dimension(xPanelSize + 10, (yPanelSize + 10));
    }

    //Instance data.
    private ParaProfTrial trial = null;
    HistogramWindow window = null;
    int xPanelSize = 600;
    int yPanelSize = 400;

    Function function = null;

    private Vector list = null;


    int xCoord = -1;
    int yCoord = -1;

    private int defaultBarLength = 500;
    int barXStart = -1;
}