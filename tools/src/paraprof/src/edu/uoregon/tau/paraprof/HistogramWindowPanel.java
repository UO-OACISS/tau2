package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.font.*;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.print.*;
import java.awt.geom.*;
//import javax.print.*;
import edu.uoregon.tau.dms.dss.*;
import java.text.*;

/**
 * HistogramWindowPanel
 * This is the panel for the HistogramWindow.
 *  
 * <P>CVS $Id: HistogramWindowPanel.java,v 1.10 2005/03/08 01:11:18 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.10 $
 * @see		HistogramWindow
 */
public class HistogramWindowPanel extends JPanel implements Printable, ParaProfImageInterface {

    public HistogramWindowPanel(ParaProfTrial ppTrial, HistogramWindow window) {
        this.ppTrial = ppTrial;
        this.window = window;

        this.setToolTipText("ParaProf histogram window!");
        setBackground(Color.white);
        this.xPanelSize = 600;
        this.yPanelSize = 400;
    }

    public String getToolTipText(MouseEvent evt) {
        try {
            int x = evt.getX();

            x -= xOffset;

            if (x < 0 || x > 550)
                return null;

            int rectWidth = 550 / window.getNumBins();

            int bin = x / rectWidth;

            if (bin < 0 || bin > window.getNumBins() - 1)
                return null;

            String minString = UtilFncs.getOutputString(window.units(), minValue + (bin * binWidth), 5);
            String maxString = UtilFncs.getOutputString(window.units(), minValue + ((bin + 1) * binWidth), 5);

            return "<html>Number of threads: " + bins[bin] + "<br>Range minimum: " + minString
                    + "<br>Range maximum: " + maxString + "</html>";
        } catch (Exception e) {
            // it's just a tooltip
            return null;
        }
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
        Dimension d = null;
        d = this.getSize();

        int yOffset = 0;

        //Draw the header if required.
        if (header) {
            d.setSize(d.getWidth(), d.getHeight() + lastHeaderEndPosition);
        } else {
            d.setSize(d.getWidth(), d.getHeight());
        }

        return d;
    }

    private void processData() {
        data = window.getData();

        maxValue = 0;
        minValue = 0;
        PPFunctionProfile ppFunctionProfile = null;

        int numThreads = 0;

        boolean start = true;
        for (Enumeration e1 = data.elements(); e1.hasMoreElements();) {
            ppFunctionProfile = (PPFunctionProfile) e1.nextElement();

                numThreads++;
                double tmpValue = ppFunctionProfile.getValue(); 
                if (start) {
                    minValue = tmpValue;
                    start = false;
                }
                maxValue = Math.max(maxValue, tmpValue);
                minValue = Math.min(minValue, tmpValue);
        }

        int numBins = window.getNumBins();

        double increment = (double) maxValue / numBins;
        binWidth = ((double) maxValue - minValue) / numBins;

        // allocate and clear the bins
        bins = new int[numBins];
        for (int i = 0; i < numBins; i++) {
            bins[i] = 0;
        }

        int count = 0;

        // fill the bins
        for (Enumeration e1 = data.elements(); e1.hasMoreElements();) {
            ppFunctionProfile = (PPFunctionProfile) e1.nextElement();
                double tmpDataValue = ppFunctionProfile.getValue();
                for (int j = 0; j < numBins; j++) {
                    if (tmpDataValue <= (minValue + (binWidth * (j + 1)))) {
                        bins[j]++;
                        count++;
                        break;
                    }
                }
        }

        // find the max number of threads in any bin
        maxInAnyBin = 0;
        for (int i = 0; i < numBins; i++) {
            maxInAnyBin = Math.max(maxInAnyBin, bins[i]);
        }

    }

    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {

        processData();

        Font font = new Font(ppTrial.getPreferencesWindow().getParaProfFont(),
                ppTrial.getPreferencesWindow().getFontStyle(), ppTrial.getPreferencesWindow().getFontSize());
        g2D.setFont(font);
        FontMetrics fontMetrics = g2D.getFontMetrics(font);

        int yOffset = 0;

        // draw the header if requested (only in the event of print with "prepend header")
        if (drawHeader) {
            FontRenderContext frc = g2D.getFontRenderContext();
            Insets insets = this.getInsets();
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
                    yOffset += ppTrial.getPreferencesWindow().getBarSpacing();
                    textLayout.draw(g2D, x, yOffset);
                    x = insets.left;
                }
            }
            lastHeaderEndPosition = yOffset;
            yOffset = yOffset + ppTrial.getPreferencesWindow().getBarSpacing();
        }

        int maxFontAscent = fontMetrics.getMaxAscent();
        int maxFontDescent = fontMetrics.getMaxDescent();

        int rectWidth = 550 / bins.length;

        g2D.setColor(Color.black);

        g2D.drawString("# Threads", 3, yOffset + maxFontAscent);

        yOffset = yOffset + (maxFontAscent * 2) + 3;

        // determine the x offset by looking at the maximum width that the y-scale strings will be
        xOffset = 0;
        for (int i = 0; i < 10; i++) {
            double height = ((10 - i)) * (double) maxInAnyBin / 10;
            String heightString = UtilFncs.formatDouble(height, 4);
            int stringWidth = fontMetrics.stringWidth(heightString);
            xOffset = Math.max(xOffset, stringWidth);
        }
        xOffset += 25;

        for (int i = 0; i < 10; i++) {
            double height = ((10 - i)) * (double) maxInAnyBin / 10;

            String heightString = UtilFncs.formatDouble(height, 4);
            int stringWidth = fontMetrics.stringWidth(heightString);

            g2D.drawLine(xOffset - 5, yOffset + i * 40, xOffset, yOffset + i * 40);
            //g2D.drawString("" + (10 * (10 - i)), 5, 33 + i * 40);
            g2D.drawString(heightString, 15, yOffset + 5 + i * 40);
        }

        int spacing = (rectWidth / 10) / 2;

        int endOfChart = rectWidth * bins.length + xOffset + 4 - spacing;

        g2D.drawLine(xOffset, 400 + yOffset, xOffset, yOffset);
        g2D.drawLine(xOffset, 400 + yOffset, endOfChart, 400 + yOffset);

        for (int i = 1; i < bins.length + 1; i++) {
            g2D.drawLine(xOffset + 4 + i * rectWidth - spacing, 400 + yOffset, xOffset + 4 + i * rectWidth
                    - spacing, 405 + yOffset);
        }

        String maxString = "Max Value = " + UtilFncs.getOutputString(window.units(), maxValue, 5);
        int maxStringWidth = fontMetrics.stringWidth(maxString);

        g2D.drawString("Min Value = " + UtilFncs.getOutputString(window.units(), minValue, 5), xOffset,
                420 + yOffset);
        g2D.drawString(maxString, endOfChart - maxStringWidth, 420 + yOffset);

        xPanelSize = endOfChart + 10;

        for (int i = 0; i < bins.length; i++) {
            if (bins[i] != 0) {
                double tmp1 = bins[i];

                double percent = (tmp1 / maxInAnyBin) * 100;
                int result = (int) percent;

                if (result < 1)
                    result = 1;

                int drawWidth = rectWidth - (rectWidth / 10);

                if (rectWidth < 10)
                    drawWidth = rectWidth;

                if (drawWidth < 1)
                    drawWidth = 1;

                g2D.setColor(Color.red);
                g2D.fillRect((xOffset + 4) + i * rectWidth, 400 + yOffset - (result * 4), drawWidth, result * 4);

                g2D.setColor(Color.black);
                g2D.drawRect((xOffset + 4) + i * rectWidth, 400 + yOffset - (result * 4), drawWidth, result * 4);
            }
        }

        int tmpXPanelSize = Math.max(xPanelSize, lastHeaderEndPosition);
        int tmpYPanelSize = 420 + yOffset;
        boolean sizeChange = false;
        if (tmpXPanelSize != xPanelSize || tmpYPanelSize != yPanelSize) {
            xPanelSize = tmpXPanelSize;
            yPanelSize = tmpYPanelSize;
            revalidate();
        }
    }

    public Dimension getPreferredSize() {
        return new Dimension(xPanelSize + 10, yPanelSize + 10);
    }

    // instance data
    private ParaProfTrial ppTrial = null;
    private HistogramWindow window = null;
    private Vector data = null;

    private int xPanelSize;
    private int yPanelSize;

    // need to keep this stuff around for the tooltip computation
    private int[] bins;
    private int maxInAnyBin;
    private double maxValue;
    private double minValue;
    private int xOffset;
    private double binWidth;

    // need to keep this around for getImageSize
    private int lastHeaderEndPosition = 0;
}