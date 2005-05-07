/*
 * 
 * StaticMainWindowPanel.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description: 
 * Things to do: 1)Add printing support.
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.geom.*;
import java.text.*;
import java.awt.font.*;
import edu.uoregon.tau.dms.dss.*;

public class FullDataWindowPanel extends JPanel implements MouseListener, Printable, ParaProfImageInterface {

    
    private ParaProfTrial ppTrial = null;
    private FullDataWindow window = null;
    private Vector list = new Vector();

    //Drawing information.
    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 500;
    private int barLength = 0;
    private int textOffset = 60;
    private int barXCoord = 100;
    private int lastHeaderEndPosition = 0;

    //Panel information.
    private int xPanelSize = 0;
    private int yPanelSize = 0;

    
    
    public FullDataWindowPanel(ParaProfTrial ppTrial, FullDataWindow window) {
        this.setToolTipText("ParaProf Full Data Window");
        setBackground(Color.white);

        addMouseListener(this);

        this.ppTrial = ppTrial;
        this.window = window;

        barLength = baseBarLength;
    }

    public String getToolTipText(MouseEvent evt) {

        try {
            //Get the location of the mouse.
            int xCoord = evt.getX();
            int yCoord = evt.getY();

            PPThread ppThread = null;

            int index = 0;

            if (ppTrial.getPreferencesWindow().getBarSpacing() != 0) {
                //Calculate which PPFunctionProfile was clicked on.
                index = (yCoord) / (ppTrial.getPreferencesWindow().getBarSpacing());
            }

            if (index == 0) { // mean
                if (xCoord < barXCoord) {
                    if (ParaProf.helpWindow.isShowing()) {
                        //Clear the window first.
                        ParaProf.helpWindow.clearText();

                        //Now send the help info.
                        ParaProf.helpWindow.writeText("You are to the left of the mean bar.");
                        ParaProf.helpWindow.writeText("");
                        ParaProf.helpWindow.writeText("Using either the right or left mouse buttons, click once"
                                + " to display more options about the"
                                + " mean values for the functionProfiles in the system.");
                    }
                    //Return a string indicating that clicking before the
                    // display bar
                    //will cause thread data to be displayed.
                    return "Left or right click for more options";
                } else {
                    ppThread = (PPThread) list.get(index);

                    Iterator l = ppThread.getFunctionListIterator();
                    while (l.hasNext()) {
                        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) l.next();
                        if (xCoord <= ppFunctionProfile.getXEnd() && xCoord >= ppFunctionProfile.getXBeg()) {
                            if (ParaProf.helpWindow.isShowing()) {
                                //Clear the window first.
                                ParaProf.helpWindow.clearText();

                                //Now send the help info.
                                ParaProf.helpWindow.writeText("Your mouse is over the mean draw bar!");
                                ParaProf.helpWindow.writeText("");
                                ParaProf.helpWindow.writeText("Current function name is: "
                                        + ppFunctionProfile.getFunctionName());
                                ParaProf.helpWindow.writeText("");
                                ParaProf.helpWindow.writeText("The mean draw bars give a visual representation of the"
                                        + " mean values for the functionProfiles which have run in the system."
                                        + "  The funtions are assigned a color from the current"
                                        + " ParaProf color set.  The colors are cycled through when the"
                                        + " number of funtions exceeds the number of available"
                                        + " colors. In the preferences section, you can add more colors."
                                        + "  Use the right and left mouse buttons " + "to give additional information.");
                            }
                            //Return the name of the function in the current
                            // thread data object.
                            return ppFunctionProfile.getFunctionName();
                        }
                    }
                    //If in here, and at this position, it means that the mouse
                    // is not over
                    //a bar. However, we might be over the misc. function
                    // section. Check for this.
                    if (xCoord <= ppThread.getMiscXEnd() && xCoord >= ppThread.getMiscXBeg()) {
                        //Output data to the help window if it is showing.
                        if (ParaProf.helpWindow.isShowing()) {
                            //Clear the window fisrt.
                            ParaProf.helpWindow.clearText();

                            //Now send the help info.
                            ParaProf.helpWindow.writeText("Your mouse is over the misc. function section!");
                            ParaProf.helpWindow.writeText("");
                            ParaProf.helpWindow.writeText("These are functionProfiles which have a non zero value,"
                                    + " but whose screen representation is less than a pixel.");
                            ParaProf.helpWindow.writeText("");
                            ParaProf.helpWindow.writeText("To view these functionProfiles, right or left click to the left of"
                                    + " this bar to bring up windows which will show more detailed information.");
                        }

                        return "Misc function section ... see help window for details";
                    }
                }
            } else if (index < list.size()) {
                ppThread = (PPThread) list.get(index);
                if (xCoord < barXCoord) {
                    if (ParaProf.helpWindow.isShowing()) {
                        //Clear the window fisrt.
                        ParaProf.helpWindow.clearText();

                        //Now send the help info.
                        ParaProf.helpWindow.writeText("n,c,t stands for: Node, Context and Thread.");
                        ParaProf.helpWindow.writeText("");
                        ParaProf.helpWindow.writeText("Using either the right or left mouse buttons, click once"
                                + " to display more options for this" + " thread.");
                    }

                    //Return a string indicating that clicking before the
                    // display bar
                    //will cause thread data to be displayed.
                    return "Left or right click for more options";
                } else {
                    Iterator l = (Iterator) ppThread.getFunctionListIterator();
                    while (l.hasNext()) {
                        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) l.next();
                        if (xCoord <= ppFunctionProfile.getXEnd() && xCoord >= ppFunctionProfile.getXBeg()) {
                            if (ParaProf.helpWindow.isShowing()) {
                                ParaProf.helpWindow.clearText();
                                ParaProf.helpWindow.writeText("Your mouse is over one of the thread draw bars!");
                                ParaProf.helpWindow.writeText("");
                                ParaProf.helpWindow.writeText("Current function name is: "
                                        + ppFunctionProfile.getFunctionName());
                                ParaProf.helpWindow.writeText("");
                                ParaProf.helpWindow.writeText("The thread draw bars give a visual representation"
                                        + " functionProfiles which have run on this thread."
                                        + "  The funtions are assigned a color from the current"
                                        + " color set.  The colors are cycled through when the"
                                        + " number of functionProfiles exceeds the number of available" + " colors."
                                        + "  Use the right and left mouse buttons " + "to give additional information.");
                            }
                            //Return the name of the function in the current
                            // thread data object.
                            return ppFunctionProfile.getFunctionName();
                        }
                    }
                    //If in here, and at this position, it means that the mouse
                    // is not over a bar. However, we might be over the misc. function
                    // section. Check for this.
                    if (xCoord <= ppThread.getMiscXEnd() && xCoord >= ppThread.getMiscXBeg()) {
                        //Output data to the help window if it is showing.
                        if (ParaProf.helpWindow.isShowing()) {
                            //Clear the window fisrt.
                            ParaProf.helpWindow.clearText();

                            //Now send the help info.
                            ParaProf.helpWindow.writeText("Your mouse is over the misc. function section!");
                            ParaProf.helpWindow.writeText("");
                            ParaProf.helpWindow.writeText("These are functionProfiles which have a non zero value,"
                                    + " but whose screen representation is less than a pixel.");
                            ParaProf.helpWindow.writeText("");
                            ParaProf.helpWindow.writeText("To view these functionProfiles, right or left click to the left of"
                                    + " this bar to bring up windows which will show more detailed information.");
                        }

                        return "Misc function section ... see help window for details";
                    }
                }
            }
        } catch (Exception e) {
            // do nothing, it's just a tooltip
        }

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

    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {

        list = window.getData();

        int yCoord = 0;
        PPThread ppThread = null;

        //To make sure the bar details are set, this
        //method must be called.
        ppTrial.getPreferencesWindow().setBarDetails(g2D);

        //Now safe to grab spacing and bar heights.
        barSpacing = ppTrial.getPreferencesWindow().getBarSpacing();
        barHeight = ppTrial.getPreferencesWindow().getBarHeight();

        //Create font.
        Font font = new Font(ppTrial.getPreferencesWindow().getParaProfFont(),
                ppTrial.getPreferencesWindow().getFontStyle(), barHeight);
        g2D.setFont(font);
        FontMetrics fmFont = g2D.getFontMetrics(font);

        //######
        //Calculating the starting positions of drawing.
        int[] maxNCT = ppTrial.getMaxNCTNumbers();
        barXCoord = (fmFont.stringWidth("n,c,t " + maxNCT[0] + "," + maxNCT[1] + "," + maxNCT[2])) + 15;
        //End - Calculating the starting positions of drawing.
        //######

        
        //determine which elements to draw (clipping)
        int[] clips = ParaProfUtils.computeClipping(g2D.getClipBounds(), window.getViewRect(), toScreen, fullWindow, list.size(), barSpacing, yCoord);
        int startElement = clips[0];
        int endElement = clips[1];
        yCoord = clips[2];
        
        
        //######
        //Draw the header if required.
        //######
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
        //######
        //End - Draw the header if required.
        //######

        int maxBarWidth = 0;

        // Draw the mean bar
        //   yCoord = yCoord + (barSpacing); //Still need to update the yCoord
        // even if the mean bar is not
        // drawn.

        // Draw the thread bar
        if (list != null) {
            for (int i = startElement; i <= endElement; i++) {
                ppThread = (PPThread) list.elementAt(i);
                yCoord = yCoord + (barSpacing);

                String barString;
                if (i == 0) {
                    barString = "mean";
                } else {
                    barString = "n,c,t " + (ppThread.getNodeID()) + "," + (ppThread.getContextID()) + ","
                            + (ppThread.getThreadID());
                }

                int width = drawBar(g2D, fmFont, barString, ppThread, barXCoord, yCoord, barHeight, toScreen);

                if (width > maxBarWidth)
                    maxBarWidth = width;

            }
        }

        if (resizePanel(fmFont, maxBarWidth + 5) && toScreen) {
            this.revalidate();
            return;
        }

    }

    private int drawStackedBar(Graphics2D g2D, FontMetrics fmFont, String text, PPThread ppThread, int barXCoord,
            int yCoord, int barHeight, boolean toScreen) {

        DssIterator l = null;
        Group selectedGroup = ppTrial.getHighlightedGroup();
        boolean highlighted = false;

        int barXEnd = (int) ((double) barXCoord + barLength);

        g2D.setColor(Color.black);
        g2D.drawString(text, (barXCoord - (fmFont.stringWidth(text)) - 5), yCoord);

        //Calculate the sum for this thread.
        double sum = 0.00;

        l = (DssIterator) ppThread.getFunctionListIterator();

        while (l.hasNext()) {
            PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) l.next();

            if (ppFunctionProfile.getExclusiveValue() > 0) {
                sum += ppFunctionProfile.getExclusiveValue();
            }
        }

        l.reset();
        double valueSum = 0;
        int lengthDrawn = 0;

        while (l.hasNext()) {

            PPFunctionProfile ppFunctionProfile;

            ppFunctionProfile = (PPFunctionProfile) l.next();

            double value = 0.00;
            value = ppFunctionProfile.getExclusiveValue();

            int xLength = 0;

            if (window.getNormalizeBars()) {
                // normalize the bars so that they're all the same length, so we
                // find percentages
                // in each thread instead of against the maximum
                xLength = (int) ((value / sum) * barLength);
            } else {
                xLength = (int) (((value + valueSum) / window.getDataSorter().getMaxExclusiveSum()) * barLength)
                        - lengthDrawn;
            }

            if (xLength > 2) { // only draw if there is something to draw
                valueSum += value;
                lengthDrawn += xLength;

                if (barHeight > 2) {
                    g2D.setColor(ppFunctionProfile.getColor());
                    g2D.fillRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);

                    if ((ppFunctionProfile.getFunction()) == (ppTrial.getHighlightedFunction())) {
                        highlighted = true;
                        g2D.setColor(ppTrial.getColorChooser().getHighlightColor());
                        g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
                        g2D.drawRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                    } else if ((ppFunctionProfile.isGroupMember(ppTrial.getHighlightedGroup()))) {
                        highlighted = true;
                        g2D.setColor(ppTrial.getColorChooser().getGroupHighlightColor());
                        g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
                        g2D.drawRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                    } else {
                        g2D.setColor(Color.black);
                        if (highlighted) {
                            //Manually draw in the lines for consistancy.
                            g2D.drawLine(barXCoord + 1, (yCoord - barHeight), barXCoord + 1 + xLength,
                                    (yCoord - barHeight));
                            g2D.drawLine(barXCoord + 1, yCoord, barXCoord + 1 + xLength, yCoord);
                            g2D.drawLine(barXCoord + 1 + xLength, (yCoord - barHeight), barXCoord + 1 + xLength, yCoord);
                            highlighted = false;
                        } else
                            g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
                    }

                    //Set the draw coords.
                    if (toScreen)
                        ppFunctionProfile.setDrawCoords(barXCoord, (barXCoord + xLength), (yCoord - barHeight), yCoord);

                    //Update barXCoord.
                    barXCoord = (barXCoord + xLength);
                } else {

                    // barHeight less than 2!

                    //Now set the color values for drawing!
                    //Get the appropriate color.
                    if ((ppFunctionProfile.getFunction()) == (ppTrial.getHighlightedFunction()))
                        g2D.setColor(ppTrial.getColorChooser().getHighlightColor());
                    else if ((ppFunctionProfile.isGroupMember(ppTrial.getHighlightedGroup())))
                        g2D.setColor(ppTrial.getColorChooser().getGroupHighlightColor());
                    else
                        g2D.setColor(ppFunctionProfile.getColor());

                    g2D.fillRect(barXCoord, (yCoord - barHeight), xLength, barHeight);

                    //Set the draw coords.
                    if (toScreen)
                        ppFunctionProfile.setDrawCoords(barXCoord, (barXCoord + xLength), (yCoord - barHeight), yCoord);

                    //Update barXCoord.
                    barXCoord = (barXCoord + xLength);
                }

            } else {

                //Still want to set the draw coords for this function, were it
                // to be non zero.
                //This aids in mouse click and tool tip events.
                if (toScreen)
                    ppFunctionProfile.setDrawCoords(barXCoord, barXCoord, (yCoord - barHeight), yCoord);
            }

        }

        //We have reached the end of the cycle for this thread. However, we
        // might be less
        //than the max length of the bar. Therefore, fill in the rest of the
        // bar with the
        //misc. function color.

        if (window.getNormalizeBars()) {

            if (barXCoord < barXEnd) {
                g2D.setColor(ppTrial.getColorChooser().getMiscFunctionColor());
                g2D.fillRect(barXCoord, (yCoord - barHeight), (barXEnd - barXCoord), barHeight);
                g2D.setColor(Color.black);
                g2D.drawRect(barXCoord, (yCoord - barHeight), (barXEnd - barXCoord), barHeight);

                ppThread.setMiscCoords(barXCoord, barXEnd, (yCoord - barHeight), yCoord);

                barXCoord = barXEnd;
            }
        } else {

            double diff = sum - valueSum;

            //int thisLength = (int)
            // ((sum/window.getSMWData().maxExclusiveSum)*barLength);

            //int diffLength = (int)
            // ((diff/window.getSMWData().maxExclusiveSum)*barLength);

            int diffLength = (int) (((diff + valueSum) / window.getDataSorter().getMaxExclusiveSum()) * barLength)
                    - lengthDrawn;

            g2D.setColor(ppTrial.getColorChooser().getMiscFunctionColor());
            g2D.fillRect(barXCoord, (yCoord - barHeight), diffLength, barHeight);
            g2D.setColor(Color.black);
            g2D.drawRect(barXCoord, (yCoord - barHeight), diffLength, barHeight);

            ppThread.setMiscCoords(barXCoord, barXCoord + diffLength, (yCoord - barHeight), yCoord);

            barXCoord += diffLength;

        }

        return barXCoord;
    }

    private int drawBar(Graphics2D g2D, FontMetrics fmFont, String text, PPThread ppThread, int barXCoord, int yCoord,
            int barHeight, boolean toScreen) {

        if (window.getStackBars())
            return drawStackedBar(g2D, fmFont, text, ppThread, barXCoord, yCoord, barHeight, toScreen);

        Iterator l = null;
        Group selectedGroup = ppTrial.getHighlightedGroup();
        boolean highlighted = false;

        int barXEnd = (int) ((double) barXCoord + barLength);

        g2D.setColor(Color.black);
        g2D.drawString(text, (barXCoord - (fmFont.stringWidth(text)) - 5), yCoord);

        //Calculate the sum for this thread.
        double sum = 0.00;

        Iterator allFuncs;

        l = ppThread.getFunctionListIterator();

        allFuncs = ((PPThread) list.get(0)).getFunctionListIterator();

        while (l.hasNext()) {
            PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) l.next();
            sum += ppFunctionProfile.getExclusiveValue();
        }

        l = ppThread.getFunctionListIterator();

        double valueSum = 0;
        int lengthDrawn = 0;

        PPFunctionProfile ppFunctionProfile;

        if (!l.hasNext()) {
            return 0;
        }

        ppFunctionProfile = (PPFunctionProfile) l.next();

        while (allFuncs.hasNext()) {
            PPFunctionProfile meanPPFunctionProfile = (PPFunctionProfile) allFuncs.next();

            Function function;
            double maxForThisFunc;
            int maxBarLength;

            function = meanPPFunctionProfile.getFunction();
            //maxForThisFunc = function.getMaxExclusive(meanPPFunctionProfile.getTrial().getSelectedMetricID());
            maxForThisFunc = window.getDataSorter().getMaxExclusives()[function.getID()];

            //maxForThisFunc wifunction.getMaxExclusive(meanPPFunctionProfile.getTrial().getSelectedMetricID());

            maxBarLength = (int) (((maxForThisFunc / window.getDataSorter().getMaxExclusiveSum()) * barLength));

            if (function != ppFunctionProfile.getFunction()) {
                // This thread didn't call this function, so just put some space
                // Note that we only do this if some other function is going to
                // show it (maxBarLength > 2)
                if (maxBarLength > 2)
                    barXCoord += maxBarLength + 5;
            } else {

                double value = 0.00;
                value = ppFunctionProfile.getExclusiveValue();

                int xLength = 0;

                if (window.getNormalizeBars()) {
                    // normalize the bars so that they're all the same length,
                    // so we find percentages
                    // in each thread instead of against the maximum
                    xLength = (int) ((value / sum) * barLength);
                } else {

                    xLength = (int) ((value / window.getDataSorter().getMaxExclusiveSum()) * barLength);
                    if (xLength < 1)
                        xLength = 1;
                }

                if (xLength > 2 || maxBarLength > 2) { // only draw if there is
                    // something to draw
                    valueSum += value;
                    lengthDrawn += xLength;

                    if (barHeight > 2) {
                        g2D.setColor(ppFunctionProfile.getColor());
                        g2D.fillRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);

                        if ((ppFunctionProfile.getFunction()) == (ppTrial.getHighlightedFunction())) {
                            highlighted = true;
                            g2D.setColor(ppTrial.getColorChooser().getHighlightColor());
                            g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
                            g2D.drawRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                        } else if ((ppFunctionProfile.isGroupMember(ppTrial.getHighlightedGroup()))) {
                            highlighted = true;
                            g2D.setColor(ppTrial.getColorChooser().getGroupHighlightColor());
                            g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
                            g2D.drawRect(barXCoord + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
                        } else {
                            g2D.setColor(Color.black);
                            g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
                        }

                        //Set the draw coords.
                        if (toScreen)
                            ppFunctionProfile.setDrawCoords(barXCoord, (barXCoord + xLength), (yCoord - barHeight),
                                    yCoord);

                        //Update barXCoord.
                        barXCoord = (barXCoord + xLength);
                    } else {

                        // barHeight less than 2!

                        //Now set the color values for drawing!
                        //Get the appropriate color.
                        if ((ppFunctionProfile.getFunction()) == (ppTrial.getHighlightedFunction()))
                            g2D.setColor(ppTrial.getColorChooser().getHighlightColor());
                        else if ((ppFunctionProfile.isGroupMember(ppTrial.getHighlightedGroup())))
                            g2D.setColor(ppTrial.getColorChooser().getGroupHighlightColor());
                        else
                            g2D.setColor(ppFunctionProfile.getColor());

                        g2D.fillRect(barXCoord, (yCoord - barHeight), xLength, barHeight);
                        g2D.setColor(Color.black);
                        g2D.drawRect(barXCoord, (yCoord - barHeight), xLength, barHeight);

                        //Set the draw coords.
                        if (toScreen)
                            ppFunctionProfile.setDrawCoords(barXCoord, (barXCoord + xLength), (yCoord - barHeight),
                                    yCoord);

                        //Update barXCoord.
                        barXCoord = (barXCoord + xLength);
                    }

                } else {

                    //Still want to set the draw coords for this function, were
                    // it to be none zero.
                    //This aids in mouse click and tool tip events.
                    if (toScreen)
                        ppFunctionProfile.setDrawCoords(barXCoord, barXCoord, (yCoord - barHeight), yCoord);
                }

                // skip some space

                if (maxBarLength > 2)
                    barXCoord += maxBarLength - xLength + 5;

                if (l.hasNext()) {
                    ppFunctionProfile = (PPFunctionProfile) l.next();
                }
            }
        }

        //We have reached the end of the cycle for this thread. However, we
        // might be less
        //than the max length of the bar. Therefore, fill in the rest of the
        // bar with the
        //misc. function color.

        double diff = sum - valueSum;

        if (diff > 0) {
            //int thisLength = (int)
            // ((sum/window.getSMWData().maxExclusiveSum)*barLength);

            //int diffLength = (int)
            // ((diff/window.getSMWData().maxExclusiveSum)*barLength);

            int diffLength = (int) (((diff + valueSum) / window.getDataSorter().getMaxExclusiveSum()) * barLength)
                    - lengthDrawn;

            g2D.setColor(ppTrial.getColorChooser().getMiscFunctionColor());
            g2D.fillRect(barXCoord, (yCoord - barHeight), diffLength, barHeight);
            g2D.setColor(Color.black);
            g2D.drawRect(barXCoord, (yCoord - barHeight), diffLength, barHeight);

            ppThread.setMiscCoords(barXCoord, barXCoord + diffLength, (yCoord - barHeight), yCoord);

            barXCoord += diffLength;
        }

        // 	g2D.setColor(Color.red);
        // 	g2D.fillRect(barXCoord, (yCoord - barHeight), 200, barHeight);
        // 	g2D.setColor(Color.black);
        // 	g2D.drawRect(barXCoord, (yCoord - barHeight), 200, barHeight);

        return barXCoord;
    }

    public void mouseClicked(MouseEvent evt) {
        try {
            //Get the location of the mouse.
            int xCoord = evt.getX();
            int yCoord = evt.getY();

            //Get the number of times clicked.
            int clickCount = evt.getClickCount();

            PPThread ppThread = null;

            // figure out which row was hit
            int index = (yCoord) / (ppTrial.getPreferencesWindow().getBarSpacing());

            if (index >= list.size()) {
                // un-highlight
                ppTrial.setHighlightedFunction(null);
                return;
            }

            ppThread = (PPThread) list.elementAt(index);

            if ((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0) { // Right
                // Click

                if (xCoord < barXCoord) { // user clicked on the N,C,T
                    ParaProfUtils.handleThreadClick(ppTrial, ppThread.getThread(), this, evt);
                } else {

                    Iterator l = ppThread.getFunctionListIterator();
                    while (l.hasNext()) {
                        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) l.next();

                        if (xCoord <= ppFunctionProfile.getXEnd() && xCoord >= ppFunctionProfile.getXBeg()) {
                            JPopupMenu popup = ParaProfUtils.createFunctionClickPopUp(ppTrial, ppFunctionProfile.getFunction(), this);
                            popup.show(this, evt.getX(), evt.getY());
                        }
                    }
                }
            } else {
                // Left Click

                if (xCoord < barXCoord) { // user clicked on N,C,T
                    ThreadDataWindow threadDataWindow = new ThreadDataWindow(ppTrial, ppThread.getNodeID(),
                            ppThread.getContextID(), ppThread.getThreadID());
                    ppTrial.getSystemEvents().addObserver(threadDataWindow);
                    threadDataWindow.show();
                } else {
                    //Find the appropriate PPFunctionProfile.
                    DssIterator l = (DssIterator) ppThread.getFunctionListIterator();
                    while (l.hasNext()) {
                        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) l.next();
                        if (xCoord <= ppFunctionProfile.getXEnd() && xCoord >= ppFunctionProfile.getXBeg()) {
                            ppTrial.setHighlightedFunction(ppFunctionProfile.getFunction());
                            //Now display the FunctionDataWindow for this
                            // function.
                            FunctionDataWindow functionDataWindow = new FunctionDataWindow(ppTrial,
                                    ppFunctionProfile.getFunction());
                            ppTrial.getSystemEvents().addObserver(functionDataWindow);
                            functionDataWindow.show();
                            return;
                        }
                    }
                    ppTrial.setHighlightedFunction(null);
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


    public void setBarLength(int barLength) {
        this.barLength = Math.max(1, barLength);
        this.repaint();
    }

    //This method sets both xPanelSize and yPanelSize.
    private boolean resizePanel(FontMetrics fmFont, int width) {
        boolean resized = false;
        int newYPanelSize = (((window.getData()).size()) + 1) * barSpacing + 10;
        int newXPanelSize = width;
        if ((newYPanelSize != yPanelSize) || (newXPanelSize != xPanelSize)) {
            yPanelSize = newYPanelSize;
            xPanelSize = newXPanelSize;
            this.setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
            resized = false;
        }
        return resized;
    }

    public Dimension getPreferredSize() {
        return new Dimension(xPanelSize, yPanelSize);
    }

}