/**
 * FunctionDataWindowPanel
 * This is the panel for the FunctionDataWindow.
 *  
 * <P>CVS $Id: FunctionDataWindowPanel.java,v 1.5 2005/01/03 20:40:33 amorris Exp $</P>
 * @author	Robert Bell, Alan Morris
 * @version	$Revision: 1.5 $
 * @see		FunctionDataWindow
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

public class FunctionDataWindowPanel extends JPanel implements ActionListener, MouseListener, Printable,
        ParaProfImageInterface {

    public FunctionDataWindowPanel(ParaProfTrial trial, Function function, FunctionDataWindow window) {

        this.ppTrial = trial;
        this.window = window;
        this.function = function;
        barLength = baseBarLength;

        //Want the background to be white.
        setBackground(Color.white);

        //Add this object as a mouse listener.
        addMouseListener(this);

        //Add items to the first popup menu.
        JMenuItem jMenuItem = new JMenuItem("Show Mean Statistics Window");
        jMenuItem.addActionListener(this);
        popup1.add(jMenuItem);

        jMenuItem = new JMenuItem("Show Mean Call Path Thread Relations");
        jMenuItem.addActionListener(this);
        popup1.add(jMenuItem);
        jMenuItem = new JMenuItem("Show Mean Call Graph");
        jMenuItem.addActionListener(this);
        popup1.add(jMenuItem);

        //Add items to the seccond popup menu.
        jMenuItem = new JMenuItem("Show Statistics Window");
        jMenuItem.addActionListener(this);
        popup2.add(jMenuItem);

        if (trial.userEventsPresent()) {
            jMenuItem = new JMenuItem("Show User Event Statistics Window");
            jMenuItem.addActionListener(this);
            popup2.add(jMenuItem);
        }

        jMenuItem = new JMenuItem("Show Call Path Thread Relations");
        jMenuItem.addActionListener(this);
        popup2.add(jMenuItem);
        jMenuItem = new JMenuItem("Show Thread Call Graph");
        jMenuItem.addActionListener(this);
        popup2.add(jMenuItem);

        //Add items to the third popup menu.
        jMenuItem = new JMenuItem("Change Function Color");
        jMenuItem.addActionListener(this);
        popup3.add(jMenuItem);

        jMenuItem = new JMenuItem("Reset to Generic Color");
        jMenuItem.addActionListener(this);
        popup3.add(jMenuItem);

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

  

    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) throws ParaProfException {

        list = window.getData();

        int stringWidth = 0;
        int yCoord = 0;
        barXCoord = barLength + textOffset;

        //To make sure the bar details are set, this
        //method must be called.
        ppTrial.getPreferences().setBarDetails(g2D);

        //Now safe to grab spacing and bar heights.
        barSpacing = ppTrial.getPreferences().getBarSpacing();
        barHeight = ppTrial.getPreferences().getBarHeight();

        //Obtain the font and its metrics.
        Font font = new Font(ppTrial.getPreferences().getParaProfFont(), ppTrial.getPreferences().getFontStyle(),
                barHeight);
        g2D.setFont(font);
        FontMetrics fmFont = g2D.getFontMetrics(font);

        //Get the max value for this function
        double maxValue = ParaProfUtils.getMaxValue(function, window.getValueType(), window.isPercent(), ppTrial);

        // too bad these next few lines are bullshit 
        // (you can't determine the max width by looking at the max value)  1.0E99 > 43.34534, but is thinner
        if (window.isPercent()) {
            stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(0, maxValue, 6) + "%");
            barXCoord = barXCoord + stringWidth;
        } else {
            stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(window.units(), maxValue,
                    ParaProf.defaultNumberPrecision));
            barXCoord = barXCoord + stringWidth;
        }

        // At this point we can determine the size this panel will require. 
        // If we need to resize, don't do any more drawing, just call revalidate.
        if (resizePanel(fmFont, barXCoord) && toScreen) {
            this.revalidate();
            return;
        }

        int yBeg = 0;
        int yEnd = 0;
        int startElement = 0;
        int endElement = 0;

        if (!fullWindow) {
            if (toScreen) {
                Rectangle clipRect = g2D.getClipBounds();
                yBeg = (int) clipRect.getY();
                yEnd = (int) (yBeg + clipRect.getHeight());
            } else {
                Rectangle viewRect = window.getViewRect();
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

        //Check for group membership.
        boolean groupMember = function.isGroupMember(ppTrial.getColorChooser().getHighlightedGroup());

        //Draw the header if required.
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
                    yCoord += barSpacing;
                    textLayout.draw(g2D, x, yCoord);
                    x = insets.left;
                }
            }
            yCoord = yCoord + (barSpacing);
            lastHeaderEndPosition = yCoord;
        }

        // Iterate through and draw each thread's values
        for (int i = startElement; i <= endElement; i++) {
            PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);
            double value = ParaProfUtils.getValue(ppFunctionProfile, window.getValueType(), window.isPercent());

            yCoord = yCoord + (barSpacing);

            String barString;
            if (ppFunctionProfile.getNodeID() == -1) {
                barString = "mean";
            } else {
                barString = "n,c,t " + (ppFunctionProfile.getNodeID()) + ","
                        + (ppFunctionProfile.getContextID()) + "," + (ppFunctionProfile.getThreadID());
            }

            drawBar(g2D, fmFont, value, maxValue, barString, barXCoord, yCoord, barHeight, groupMember);
        }
    }

    private void drawBar(Graphics2D g2D, FontMetrics fmFont, double value, double maxValue, String text,
            int barXCoord, int yCoord, int barHeight, boolean groupMember) {
        int xLength = 0;
        double d = 0.0;
        String s = null;
        int stringWidth = 0;
        int stringStart = 0;
        d = (value / maxValue);
        xLength = (int) (d * barLength);
        if (xLength == 0)
            xLength = 1;

        if ((xLength > 2) && (barHeight > 2)) {
            g2D.setColor(function.getColor());
            g2D.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);

            if (function == (ppTrial.getColorChooser().getHighlightedFunction())) {
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
            if (function == (ppTrial.getColorChooser().getHighlightedFunction()))
                g2D.setColor(ppTrial.getColorChooser().getHighlightColor());
            else if ((function.isGroupMember(ppTrial.getColorChooser().getHighlightedGroup())))
                g2D.setColor(ppTrial.getColorChooser().getGroupHighlightColor());
            else {
                g2D.setColor(function.getColor());
            }
            g2D.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
        }

        //Draw the value next to the bar.
        g2D.setColor(Color.black);
        //Do not want to put a percent sign after the bar if we are not
        // exclusive or inclusive.
        if ((window.isPercent()) && ((window.getValueType()) <= 4)) {

            //s = (UtilFncs.adjustDoublePresision(value, 4)) + "%";
            s = UtilFncs.getOutputString(0, value, 6) + "%";

        } else
            s = UtilFncs.getOutputString(window.units(), value, ParaProf.defaultNumberPrecision);
        stringWidth = fmFont.stringWidth(s);
        //Now draw the percent value to the left of the bar.
        stringStart = barXCoord - xLength - stringWidth - 5;
        g2D.drawString(s, stringStart, yCoord);
        g2D.drawString(text, (barXCoord + 5), yCoord);
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();
            PPFunctionProfile ppFunctionProfile = null;

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Show Mean Statistics Window")) {
                    StatWindow statWindow = new StatWindow(ppTrial, -1, -1, -1, window.getDataSorter(), false);
                    ppTrial.getSystemEvents().addObserver(statWindow);
                    statWindow.show();
                } else if (arg.equals("Show Mean User Event Statistics Window")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        StatWindow statWindow = new StatWindow(ppTrial, ppFunctionProfile.getNodeID(),
                                ppFunctionProfile.getContextID(), ppFunctionProfile.getThreadID(),
                                window.getDataSorter(), true);
                        ppTrial.getSystemEvents().addObserver(statWindow);
                        statWindow.show();
                    }
                } else if (arg.equals("Show Mean Call Path Thread Relations")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        CallPathUtilFuncs.trimCallPathData(ppTrial.getTrialData(), ppTrial.getNCT().getThread(
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID()));
                        CallPathTextWindow callPathTextWindow = new CallPathTextWindow(ppTrial,
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID(), window.getDataSorter(), 1);
                        ppTrial.getSystemEvents().addObserver(callPathTextWindow);
                        callPathTextWindow.show();
                    }
                } else if (arg.equals("Show Mean Call Graph")) {
                    PPThread ppThread = (PPThread) clickedOnObject;

                    CallGraphWindow tmpRef = new CallGraphWindow(ppTrial, -1, -1, -1, window.getDataSorter());
                    ppTrial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();

                } else if (arg.equals("Show Statistics Window")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        StatWindow statWindow = new StatWindow(ppTrial, ppFunctionProfile.getNodeID(),
                                ppFunctionProfile.getContextID(), ppFunctionProfile.getThreadID(),
                                window.getDataSorter(), false);
                        ppTrial.getSystemEvents().addObserver(statWindow);
                        statWindow.show();
                    }
                } else if (arg.equals("Show User Event Statistics Window")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        StatWindow statWindow = new StatWindow(ppTrial, ppFunctionProfile.getNodeID(),
                                ppFunctionProfile.getContextID(), ppFunctionProfile.getThreadID(),
                                window.getDataSorter(), true);
                        ppTrial.getSystemEvents().addObserver(statWindow);
                        statWindow.show();
                    }
                } else if (arg.equals("Show Call Path Thread Relations")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        CallPathUtilFuncs.trimCallPathData(ppTrial.getTrialData(), ppTrial.getNCT().getThread(
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID()));
                        CallPathTextWindow callPathTextWindow = new CallPathTextWindow(ppTrial,
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID(), window.getDataSorter(), 1);
                        ppTrial.getSystemEvents().addObserver(callPathTextWindow);
                        callPathTextWindow.show();
                    }

                } else if (arg.equals("Show Thread Call Graph")) {
                    PPThread ppThread = (PPThread) clickedOnObject;
                    CallPathUtilFuncs.trimCallPathData(ppTrial.getTrialData(), ppTrial.getNCT().getThread(
                            ppThread.getNodeID(), ppThread.getContextID(), ppThread.getThreadID()));

                    CallGraphWindow tmpRef = new CallGraphWindow(ppTrial, ppThread.getNodeID(),
                            ppThread.getContextID(), ppThread.getThreadID(), window.getDataSorter());
                    ppTrial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();

                } else if (arg.equals("Change Function Color")) {
                    Color tmpCol = function.getColor();

                    JColorChooser tmpJColorChooser = new JColorChooser();
                    tmpCol = JColorChooser.showDialog(this, "Please select a new color", tmpCol);
                    if (tmpCol != null) {
                        function.setSpecificColor(tmpCol);
                        function.setColorFlag(true);
                        ppTrial.getSystemEvents().updateRegisteredObjects("colorEvent");
                    }
                } else if (arg.equals("Reset to Generic Color")) {
                    function.setColorFlag(false);
                    ppTrial.getSystemEvents().updateRegisteredObjects("colorEvent");
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
            int index = (yCoord) / (ppTrial.getPreferences().getBarSpacing());

            if (list != null && index < list.size()) {
                ppFunctionProfile = (PPFunctionProfile) list.elementAt(index);
                if ((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0) {
                    clickedOnObject = ppFunctionProfile;
                    if (xCoord > barXCoord) { //barXCoord should have been set during the last render.
                        if (index == 0)
                            popup1.show(this, evt.getX(), evt.getY());
                        else
                            popup2.show(this, evt.getX(), evt.getY());
                    } else
                        popup3.show(this, evt.getX(), evt.getY());
                } else {
                    if (xCoord > barXCoord) { //barXCoord should have been set during the last render.
                        ThreadDataWindow threadDataWindow = new ThreadDataWindow(ppTrial,
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID(), window.getDataSorter());
                        ppTrial.getSystemEvents().addObserver(threadDataWindow);
                        threadDataWindow.show();
                    } else {
                        ppTrial.getColorChooser().toggleHighlightedFunction(function);
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
        
        if (header) {
            d.setSize(d.getWidth(), d.getHeight() + lastHeaderEndPosition);
        } else {
            d.setSize(d.getWidth(), d.getHeight());
        }
        
        return d;
    }

    public void changeInMultiples() {
        computeBarLength();
        this.repaint();
    }

    public void computeBarLength() {
        double sliderValue = (double) window.getSliderValue();
        double sliderMultiple = window.getSliderMultiple();
        barLength = (int) (baseBarLength * ((double) (sliderValue * sliderMultiple)));
    }

    //This method sets both xPanelSize and yPanelSize.
    private boolean resizePanel(FontMetrics fmFont, int barXCoord) {
        boolean resized = false;
        int newYPanelSize = ((window.getData().size()) + 1) * barSpacing + 10;
        int[] nct = ppTrial.getMaxNCTNumbers();
        String nctString = "n,c,t " + nct[0] + "," + nct[1] + "," + nct[2];

        int newXPanelSize = barXCoord + 5 + (fmFont.stringWidth(nctString)) + 25;
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

    

    //Instance data.
    private ParaProfTrial ppTrial = null;
    private FunctionDataWindow window = null;
    private Vector list = null;
    private Function function = null;

    //Drawing information.
    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 250;
    private int barLength = 0;
    private int textOffset = 60;
    private int barXCoord = 0;
    private int lastHeaderEndPosition = 0;

    //Panel information.
    int xPanelSize = 0;
    int yPanelSize = 0;

    //Popup menu stuff.
    private JPopupMenu popup1 = new JPopupMenu();
    private JPopupMenu popup2 = new JPopupMenu();
    private JPopupMenu popup3 = new JPopupMenu();
    private Object clickedOnObject = null;
}