/*
 * 
 * FunctionDataWindowPanel.java
 * 
 * Title: ParaProf Author: Robert Bell Description:
 * 
 * Things to do:
 *  
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

public class FunctionDataWindowPanel extends JPanel implements ActionListener, MouseListener,
        Printable, ParaProfImageInterface {
    public FunctionDataWindowPanel() {
        try {
            setSize(new java.awt.Dimension(xPanelSize, yPanelSize));

            //Schedule a repaint of this panel.
            this.repaint();
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "MDWP01");
        }
    }

    public FunctionDataWindowPanel(ParaProfTrial trial, Function function,
            FunctionDataWindow mDWindow, boolean debug) {
        try {

            this.trial = trial;
            this.mDWindow = mDWindow;
            this.debug = debug;
            this.function = function;
            barLength = baseBarLength;

            //Want the background to be white.
            setBackground(Color.white);

            //Add this object as a mouse listener.
            addMouseListener(this);

            //######
            //Add items to the first popup menu.
            //######
            JMenuItem jMenuItem = new JMenuItem("Show Mean Statistics Window");
            jMenuItem.addActionListener(this);
            popup1.add(jMenuItem);

            jMenuItem = new JMenuItem("Show Mean Call Path Thread Relations");
            jMenuItem.addActionListener(this);
            popup1.add(jMenuItem);
            jMenuItem = new JMenuItem("Show Mean Call Graph");
            jMenuItem.addActionListener(this);
            popup1.add(jMenuItem);
            //######
            //End - Add items to the first popup menu.
            //######

            //######
            //Add items to the seccond popup menu.
            //######
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
            //######
            //End - Add items to the second popup menu.
            //######

            //######
            //Add items to the third popup menu.
            //######
            jMenuItem = new JMenuItem("Change Function Color");
            jMenuItem.addActionListener(this);
            popup3.add(jMenuItem);

            jMenuItem = new JMenuItem("Reset to Generic Color");
            jMenuItem.addActionListener(this);
            popup3.add(jMenuItem);
            //######
            //End - Add items to the third popup menu.
            //######
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "MDWP02");
        }

    }

    public void paintComponent(Graphics g) {
        try {
            super.paintComponent(g);
            renderIt((Graphics2D) g, 0, false);
        } catch (Exception e) {
            System.out.println(e);
            UtilFncs.systemError(e, null, "TDWP03");
        }
    }

    public int print(Graphics g, PageFormat pf, int page) {

        if (pf.getOrientation() == PageFormat.PORTRAIT)
            System.out.println("PORTRAIT");
        else if (pf.getOrientation() == PageFormat.LANDSCAPE)
            System.out.println("LANDSCAPE");

        if (page >= 3)
            return Printable.NO_SUCH_PAGE;
        Graphics2D g2 = (Graphics2D) g;
        g2.translate(pf.getImageableX(), pf.getImageableY());
        g2.draw(new Rectangle2D.Double(0, 0, pf.getImageableWidth(), pf.getImageableHeight()));

        renderIt(g2, 2, false);

        return Printable.PAGE_EXISTS;
    }

    public void renderIt(Graphics2D g2D, int instruction, boolean header) {
        try {
            if (this.debug()) {
                System.out.println("####################################");
                System.out.println("FunctionDataWindowPanel.renderIt(...)");
                System.out.println("####################################");
            }

            list = mDWindow.getData();

            //######
            //Some declarations.
            //######
            double value = 0.0;
            double maxValue = 0.0;
            int stringWidth = 0;
            int yCoord = 0;
            barXCoord = barLength + textOffset;
            PPFunctionProfile ppFunctionProfile = null;
            //######
            //Some declarations.
            //######

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

            //***
            //Set the max and mean values for this function.
            //***
            switch (mDWindow.getValueType()) {
            case 2:
                if (mDWindow.isPercent()) {
                    maxValue = function.getMaxExclusivePercent(trial.getSelectedMetricID());
                } else {
                    maxValue = function.getMaxExclusive(trial.getSelectedMetricID());
                }
                break;
            case 4:
                if (mDWindow.isPercent()) {
                    maxValue = function.getMaxInclusivePercent(trial.getSelectedMetricID());
                } else {
                    maxValue = function.getMaxInclusive(trial.getSelectedMetricID());
                }
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
                UtilFncs.systemError(null, null, "Unexpected type - MDWP value: "
                        + mDWindow.getValueType());
            }

            if (mDWindow.isPercent()) {
                stringWidth = fmFont.stringWidth(UtilFncs.adjustDoublePresision(maxValue,
                        ParaProf.defaultNumberPrecision)
                        + "%");
                barXCoord = barXCoord + stringWidth;
            } else {
                stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(mDWindow.units(),
                        maxValue, ParaProf.defaultNumberPrecision));
                barXCoord = barXCoord + stringWidth;
            }
            //***
            //End - Set the max and mean values for this mapping.
            //***

            //At this point we can determine the size this panel will
            //require. If we need to resize, don't do any more drawing,
            //just call revalidate.
            if (resizePanel(fmFont, barXCoord) && instruction == 0) {
                this.revalidate();
                return;
            }

            int yBeg = 0;
            int yEnd = 0;
            int startElement = 0;
            int endElement = 0;
            Rectangle clipRect = null;
            Rectangle viewRect = null;

            if (instruction == 0 || instruction == 1) {
                if (instruction == 0) {
                    clipRect = g2D.getClipBounds();
                    yBeg = (int) clipRect.getY();
                    yEnd = (int) (yBeg + clipRect.getHeight());
                    /*
                     * System.out.println("Clipping Rectangle: xBeg,xEnd:
                     * "+clipRect.getX()+","+((clipRect.getX())+(clipRect.getWidth()))+ "
                     * yBeg,yEnd:
                     * "+clipRect.getY()+","+((clipRect.getY())+(clipRect.getHeight())));
                     */
                } else {
                    viewRect = mDWindow.getViewRect();
                    yBeg = (int) viewRect.getY();
                    yEnd = (int) (yBeg + viewRect.getHeight());
                    /*
                     * System.out.println("Viewing Rectangle: xBeg,xEnd:
                     * "+viewRect.getX()+","+((viewRect.getX())+(viewRect.getWidth()))+ "
                     * yBeg,yEnd:
                     * "+viewRect.getY()+","+((viewRect.getY())+(viewRect.getHeight())));
                     */
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

                if (instruction == 0)
                    yCoord = yCoord + (startElement * barSpacing);
            } else if (instruction == 2 || instruction == 3) {
                startElement = 0;
                endElement = ((list.size()) - 1);
            }

            //Check for group membership.
            boolean groupMember = function.isGroupMember(trial.getColorChooser().getHighlightedGroup());

            //######
            //Draw the header if required.
            //######
            if (header) {
                FontRenderContext frc = g2D.getFontRenderContext();
                Insets insets = this.getInsets();
                yCoord = yCoord + (barSpacing);
                String headerString = mDWindow.getHeaderString();
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

            //######
            //Draw thread information for this mapping.
            //######
            for (int i = startElement; i <= endElement; i++) {
                ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);
                switch (mDWindow.getValueType()) {
                case 2:
                    if (mDWindow.isPercent())
                        value = ppFunctionProfile.getExclusivePercentValue();
                    else
                        value = ppFunctionProfile.getExclusiveValue();
                    break;
                case 4:
                    if (mDWindow.isPercent())
                        value = ppFunctionProfile.getInclusivePercentValue();
                    else
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
                    UtilFncs.systemError(null, null, "Unexpected type - MDWP value: "
                            + mDWindow.getValueType());
                }

                yCoord = yCoord + (barSpacing);

                String barString;
                if (ppFunctionProfile.getNodeID() == -1) {
                    barString = "mean";
                } else {
                    barString = "n,c,t " + (ppFunctionProfile.getNodeID()) + ","
                            + (ppFunctionProfile.getContextID()) + ","
                            + (ppFunctionProfile.getThreadID());
                }

                drawBar(g2D, fmFont, value, maxValue, barString, barXCoord, yCoord, barHeight,
                        groupMember, instruction);
            }
            //######
            //End - Draw thread information for this mapping.
            //######
            if (this.debug()) {
                System.out.println("####################################");
                System.out.println("End - FunctionDataWindowPanel.renderIt(...)");
                System.out.println("####################################");
            }
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "MDWP03");
        }
    }

    private void drawBar(Graphics2D g2D, FontMetrics fmFont, double value, double maxValue,
            String text, int barXCoord, int yCoord, int barHeight, boolean groupMember,
            int instruction) {
        if (this.debug()) {
            System.out.println("####################################");
            System.out.println("FunctionDataWindowPanel.drawBar(...)");
            System.out.println("####################################");
        }
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
            g2D.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1,
                    barHeight - 1);

            if (function == (trial.getColorChooser().getHighlightedFunction())) {
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
            if (function == (trial.getColorChooser().getHighlightedFunction()))
                g2D.setColor(trial.getColorChooser().getHighlightColor());
            else if ((function.isGroupMember(trial.getColorChooser().getHighlightedGroup())))
                g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
            else {
                g2D.setColor(function.getColor());
            }
            g2D.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
        }

        //Draw the value next to the bar.
        g2D.setColor(Color.black);
        //Do not want to put a percent sign after the bar if we are not
        // exclusive or inclusive.
        if ((mDWindow.isPercent()) && ((mDWindow.getValueType()) <= 4))
            s = (UtilFncs.adjustDoublePresision(value, 4)) + "%";
        else
            s = UtilFncs.getOutputString(mDWindow.units(), value, ParaProf.defaultNumberPrecision);
        stringWidth = fmFont.stringWidth(s);
        //Now draw the percent value to the left of the bar.
        stringStart = barXCoord - xLength - stringWidth - 5;
        g2D.drawString(s, stringStart, yCoord);
        g2D.drawString(text, (barXCoord + 5), yCoord);
        if (this.debug()) {
            System.out.println("####################################");
            System.out.println("End - FunctionDataWindowPanel.drawBar(...)");
            System.out.println("####################################");
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
            PPFunctionProfile ppFunctionProfile = null;

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Show Mean Statistics Window")) {
                    StatWindow statWindow = new StatWindow(trial, -1, -1, -1,
                            mDWindow.getDataSorter(), false, this.debug());
                    trial.getSystemEvents().addObserver(statWindow);
                    statWindow.show();
                } else if (arg.equals("Show Mean User Event Statistics Window")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        StatWindow statWindow = new StatWindow(trial,
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID(), mDWindow.getDataSorter(), true,
                                this.debug());
                        trial.getSystemEvents().addObserver(statWindow);
                        statWindow.show();
                    }
                } else if (arg.equals("Show Mean Call Path Thread Relations")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        CallPathUtilFuncs.trimCallPathData(trial.getTrialData(),
                                trial.getNCT().getThread(ppFunctionProfile.getNodeID(),
                                        ppFunctionProfile.getContextID(),
                                        ppFunctionProfile.getThreadID()));
                        CallPathTextWindow callPathTextWindow = new CallPathTextWindow(trial,
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID(), mDWindow.getDataSorter(), 1,
                                this.debug());
                        trial.getSystemEvents().addObserver(callPathTextWindow);
                        callPathTextWindow.show();
                    }
                } else if (arg.equals("Show Mean Call Graph")) {
                    PPThread ppThread = (PPThread) clickedOnObject;

                    CallGraphWindow tmpRef = new CallGraphWindow(trial, -1, -1, -1, mDWindow.getDataSorter());
                    trial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();
    

                } else if (arg.equals("Show Statistics Window")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        StatWindow statWindow = new StatWindow(trial,
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID(), mDWindow.getDataSorter(), false,
                                this.debug());
                        trial.getSystemEvents().addObserver(statWindow);
                        statWindow.show();
                    }
                } else if (arg.equals("Show User Event Statistics Window")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        StatWindow statWindow = new StatWindow(trial,
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID(), mDWindow.getDataSorter(), true,
                                this.debug());
                        trial.getSystemEvents().addObserver(statWindow);
                        statWindow.show();
                    }
                } else if (arg.equals("Show Call Path Thread Relations")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        CallPathUtilFuncs.trimCallPathData(trial.getTrialData(),
                                trial.getNCT().getThread(ppFunctionProfile.getNodeID(),
                                        ppFunctionProfile.getContextID(),
                                        ppFunctionProfile.getThreadID()));
                        CallPathTextWindow callPathTextWindow = new CallPathTextWindow(trial,
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID(), mDWindow.getDataSorter(), 1,
                                this.debug());
                        trial.getSystemEvents().addObserver(callPathTextWindow);
                        callPathTextWindow.show();
                    }

                
                } else if (arg.equals("Show Thread Call Graph")) {
                    PPThread ppThread = (PPThread) clickedOnObject;
                    CallPathUtilFuncs.trimCallPathData(trial.getTrialData(),
                            trial.getNCT().getThread(ppThread.getNodeID(),
                                    ppThread.getContextID(), ppThread.getThreadID()));

                    CallGraphWindow tmpRef = new CallGraphWindow(trial, ppThread.getNodeID(), ppThread.getContextID(),
                            ppThread.getThreadID(), mDWindow.getDataSorter());
                    trial.getSystemEvents().addObserver(tmpRef);
                    tmpRef.show();

                
                } else if (arg.equals("Change Function Color")) {
                    Color tmpCol = function.getColor();

                    JColorChooser tmpJColorChooser = new JColorChooser();
                    tmpCol = JColorChooser.showDialog(this, "Please select a new color", tmpCol);
                    if (tmpCol != null) {
                        function.setSpecificColor(tmpCol);
                        function.setColorFlag(true);
                        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                    }
                } else if (arg.equals("Reset to Generic Color")) {
                    function.setColorFlag(false);
                    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                }
            }
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "MDWP04");
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
                    clickedOnObject = ppFunctionProfile;
                    if (xCoord > barXCoord) { //barXCoord should have been set
                        // during the last render.
                        if (index == 0)
                            popup1.show(this, evt.getX(), evt.getY());
                        else
                            popup2.show(this, evt.getX(), evt.getY());
                    } else
                        popup3.show(this, evt.getX(), evt.getY());
                } else {
                    if (xCoord > barXCoord) { //barXCoord should have been set
                        // during the last render.
                        ThreadDataWindow threadDataWindow = new ThreadDataWindow(trial,
                                ppFunctionProfile.getNodeID(), ppFunctionProfile.getContextID(),
                                ppFunctionProfile.getThreadID(), mDWindow.getDataSorter(),
                                this.debug());
                        trial.getSystemEvents().addObserver(threadDataWindow);
                        threadDataWindow.show();
                    } else {
                        trial.getColorChooser().toggleHighlightedFunction(function);
                    }
                }
            }
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "MDWP05");
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
            d = mDWindow.getSize();
        d.setSize(d.getWidth(), d.getHeight() + lastHeaderEndPosition);
        System.out.println("getImageSize x,y: " + d.getWidth() + "," + d.getHeight());
        return d;
    }

    //######
    //End - ParaProfImageInterface
    //######

    //####################################
    //End - Interface code.
    //####################################

    public void changeInMultiples() {
        computeBarLength();
        this.repaint();
    }

    public void computeBarLength() {
        try {
            double sliderValue = (double) mDWindow.getSliderValue();
            double sliderMultiple = mDWindow.getSliderMultiple();
            barLength = (int) (baseBarLength * ((double) (sliderValue * sliderMultiple)));
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "MDWP06");
        }
    }

    //This method sets both xPanelSize and yPanelSize.
    private boolean resizePanel(FontMetrics fmFont, int barXCoord) {
        boolean resized = false;
        try {
            int newYPanelSize = ((mDWindow.getData().size()) + 6) * barSpacing + 10;
            int[] nct = trial.getMaxNCTNumbers();
            String nctString = "n,c,t " + nct[0] + "," + nct[1] + "," + nct[2];
            
            int newXPanelSize = barXCoord + 5 + (fmFont.stringWidth(nctString)) + 25;
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
        return new Dimension(xPanelSize, yPanelSize);
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
    private ParaProfTrial trial = null;
    private FunctionDataWindow mDWindow = null;
    private Vector list = null;
    private Function function = null;

    //######
    //Drawing information.
    //######
    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 250;
    private int barLength = 0;
    private int textOffset = 60;
    private int barXCoord = 0;
    private int lastHeaderEndPosition = 0;
    //######
    //End - Drawing information.
    //######

    //######
    //Panel information.
    //######
    int xPanelSize = 0;
    int yPanelSize = 0;
    //######
    //End - Panel information.
    //######

    //######
    //Popup menu stuff.
    //######
    private JPopupMenu popup1 = new JPopupMenu();
    private JPopupMenu popup2 = new JPopupMenu();
    private JPopupMenu popup3 = new JPopupMenu();
    private Object clickedOnObject = null;
    //######
    //End - Popup menu stuff.
    //######

    private boolean debug = false; //Off by default.
    //####################################
    //Instance data.
    //####################################
}