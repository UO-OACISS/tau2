/*
 * 
 * ThreadDataWindowPanel.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell, Alan Morris 
 * Description:
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.ClipboardOwner;
import java.awt.datatransfer.Transferable;
import java.awt.event.*;
import java.awt.font.*;
import java.awt.geom.AffineTransform;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.text.AttributedCharacterIterator;
import java.text.AttributedString;
import java.util.StringTokenizer;
import java.util.Vector;

import javax.swing.*;

import edu.uoregon.tau.dms.dss.Function;
import edu.uoregon.tau.dms.dss.UtilFncs;
import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.interfaces.ImageExport;
import edu.uoregon.tau.paraprof.interfaces.Searchable;

public class ThreadDataWindowPanel extends JPanel implements Printable, ImageExport, MouseListener {

    //Instance data.
    private int xPanelSize = 640;
    private int yPanelSize = 480;

    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 250;
    private int barLength = 0;
    private int textOffset = 60;

    private int maxRightSideStringPixelWidth = 0;

    private ParaProfTrial ppTrial = null;
    private ThreadDataWindow window = null;
    private edu.uoregon.tau.dms.dss.Thread thread = null;
    private Vector list = new Vector();

    private int lastHeaderEndPosition = 0;
    private Searcher searcher;

    
    public ThreadDataWindowPanel(ParaProfTrial ppTrial, edu.uoregon.tau.dms.dss.Thread thread, ThreadDataWindow tDWindow) {
        this.thread = thread;
        setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
        setBackground(Color.white);

        searcher = new Searcher(this, tDWindow);
        addMouseListener(searcher);
        addMouseMotionListener(searcher);

        addMouseListener(this);

        this.ppTrial = ppTrial;
        this.window = tDWindow;
        barLength = baseBarLength;

        setAutoscrolls(true);


        //Schedule a repaint of this panel.
        repaint();
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
            new ParaProfErrorDialog(e);
            return NO_SUCH_PAGE;
        }
    }

    public void resetStringSize() {
        maxRightSideStringPixelWidth = 0;
        searcher.setSearchLines(null);
    }

    private void setSearchLines() {
        if (searcher.getSearchLines() == null && list != null) {
            Vector searchLines = new Vector();
            for (int i = 0; i < list.size(); i++) {
                PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);
                searchLines.add(ppFunctionProfile.getFunctionName());
            }
            searcher.setSearchLines(searchLines);
        }
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        list = window.getData();

        //Some declarations.
        int barXCoord = barLength + textOffset;
        PPFunctionProfile ppFunctionProfile = null;

        // With group support present, it is possible that the number of
        // functions in our data list is zero. This can occur when the user's selected
        // groups to display are not present on this thread ... for example. If so, just return.

        if ((list.size()) == 0)
            return;

        //To make sure the bar details are set, this method must be called.
        ppTrial.getPreferencesWindow().setBarDetails(g2D);

        //Now safe to grab spacing and bar heights.
        barSpacing = ppTrial.getPreferencesWindow().getBarSpacing();
        barHeight = ppTrial.getPreferencesWindow().getBarHeight();

        searcher.setLineHeight(barSpacing);

        //Obtain the font and its metrics.
        Font font = new Font(ppTrial.getPreferencesWindow().getParaProfFont(),
                ppTrial.getPreferencesWindow().getFontStyle(), barHeight);
        g2D.setFont(font);
        FontMetrics fmFont = g2D.getFontMetrics(font);

        searcher.setMaxDescent(fmFont.getMaxDescent());

        if (maxRightSideStringPixelWidth == 0) {
            for (int i = 0; i < list.size(); i++) {
                ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);
                maxRightSideStringPixelWidth = Math.max(maxRightSideStringPixelWidth,
                        fmFont.stringWidth(ppFunctionProfile.getFunctionName() + 5));
            }
        }

        setSearchLines();

        //######
        //Set max values.
        //######
        double maxValue = window.getDataSorter().getValueType().getThreadMaxValue(window.getPPThread().getThread(),
                window.getDataSorter().getSelectedMetricID());

        // this is crap, this is not the correct way to determine the largest string, the largest
        // value item is not necessarily the longest as a string (variable width font)
        if (window.isPercent()) {
            int stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(0, maxValue, 6) + "%");
            barXCoord = barXCoord + stringWidth;
        } else {
            int stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(window.units(), maxValue,
                    ParaProf.defaultNumberPrecision));
            barXCoord = barXCoord + stringWidth;
        }
        //######
        //End - Set max values.
        //######


        int yCoord = 0;

        //determine which elements to draw (clipping)
        int[] clips = ParaProfUtils.computeClipping(g2D.getClipBounds(), window.getViewRect(), toScreen, fullWindow,
                list.size(), barSpacing, yCoord);
        int startElement = clips[0];
        int endElement = clips[1];
        yCoord = clips[2];

        //At this point we can determine the size this panel will
        //require. If we need to resize, don't do any more drawing,
        //just call revalidate. Make sure we check the instruction value as
        // we only want to revalidate if we are drawing to the screen.
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

        searcher.setVisibleLines(startElement, endElement);
        searcher.setG2d(g2D);
        searcher.setXOffset(barXCoord + 5);

        // Iterate through each function and draw it's bar
        for (int i = startElement; i <= endElement; i++) {
            ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);

            //value = ParaProfUtils.getValue(ppFunctionProfile, window.getValueType(), window.isPercent());
            double value = ppFunctionProfile.getValue();

            yCoord = yCoord + (barSpacing);
            drawBar(g2D, fmFont, maxValue, barXCoord, yCoord, barHeight, ppFunctionProfile, i, toScreen);
        }

    }

    private void drawBar(Graphics2D g2D, FontMetrics fmFont, double maxValue, int barXCoord, int yCoord, int barHeight,
            PPFunctionProfile ppFunctionProfile, int line, boolean toScreen) {

        double value = ppFunctionProfile.getValue();
        Function function = ppFunctionProfile.getFunction();
        String functionName = ppFunctionProfile.getFunctionName();

        String originalFunctionName = functionName;
        boolean groupMember = ppFunctionProfile.isGroupMember(ppTrial.getHighlightedGroup());

        double d = (value / maxValue);
        int xLength = (int) (d * barLength);
        if (xLength == 0)
            xLength = 1;

        if ((xLength > 2) && (barHeight > 2)) {
            g2D.setColor(ppFunctionProfile.getColor());
            g2D.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);

            if (function == (ppTrial.getHighlightedFunction())) {
                g2D.setColor(ppTrial.getColorChooser().getHighlightColor());
                g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                g2D.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            } else if (groupMember) {
                g2D.setColor(ppTrial.getColorChooser().getGroupHighlightColor());
                g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                g2D.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            } else {
                g2D.setColor(Color.black);
                g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            }
        } else {
            if (function == (ppTrial.getHighlightedFunction()))
                g2D.setColor(ppTrial.getColorChooser().getHighlightColor());
            else if (groupMember)
                g2D.setColor(ppTrial.getColorChooser().getGroupHighlightColor());
            else {
                g2D.setColor(ppFunctionProfile.getColor());
            }
            g2D.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
        }

        //Draw the value next to the bar.
        g2D.setColor(Color.black);
        //Do not want to put a percent sign after the bar if we are not doing exclusive or inclusive.

        String leftString;

        if (window.getDataSorter().getValueType() == ValueType.EXCLUSIVE_PERCENT
                || window.getDataSorter().getValueType() == ValueType.INCLUSIVE_PERCENT) {
            leftString = UtilFncs.getOutputString(0, value, 6) + "%";

        } else {
            leftString = UtilFncs.getOutputString(window.units(), value, ParaProf.defaultNumberPrecision);
        }

        int stringWidth = fmFont.stringWidth(leftString);
        int stringStart = barXCoord - xLength - stringWidth - 5;

        g2D.drawString(leftString, stringStart, yCoord);

        int x = (barXCoord + 5);
        int y = yCoord;

        searcher.drawHighlights(g2D, x, y, line);

        // now draw the actual text
        g2D.setPaint(Color.black);
        g2D.drawString(functionName, x, y);
    }

    public void setBarLength(int barLength) {
        this.barLength = Math.max(1, barLength);
        this.repaint();
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
            int index = (yCoord) / (ppTrial.getPreferencesWindow().getBarSpacing());

            if (list != null && index < list.size()) {
                ppFunctionProfile = (PPFunctionProfile) list.elementAt(index);
                if (ParaProfUtils.rightClick(evt)) {

                    JPopupMenu popup = ParaProfUtils.createFunctionClickPopUp(ppTrial, ppFunctionProfile.getFunction(),
                            this);
                    popup.show(this, evt.getX(), evt.getY());
                    return;
                } else { // not right click
                    ppTrial.toggleHighlightedFunction(ppFunctionProfile.getFunction());
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
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
            maxRightSideStringPixelWidth = Math.max(maxRightSideStringPixelWidth,
                    fmFont.stringWidth(ppFunctionProfile.getFunctionName() + 5));
        }

        width = barXCoord + 5 + maxRightSideStringPixelWidth;
        newXPanelSize = Math.max(newXPanelSize, width);

        newYPanelSize = barSpacing + ((list.size() - 1) * barSpacing);

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

    public Searcher getSearcher() {
        return searcher;
    }

    public void mouseReleased(MouseEvent evt) {
    }

    public void mouseEntered(MouseEvent evt) {
    }

    public void mouseExited(MouseEvent evt) {
    }

    public void mouseMoved(MouseEvent e) {
    }

    public void mouseDragged(MouseEvent evt) {
    }

    public void mousePressed(MouseEvent e) {
    }

}