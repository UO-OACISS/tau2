/*
 * 
 * StatWindowPanel.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.text.*;
import java.awt.font.*;
import java.awt.font.TextAttribute;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.awt.geom.*;
import java.awt.print.*;
import edu.uoregon.tau.dms.dss.*;

public class StatWindowPanel extends JPanel implements ActionListener, MouseListener, Printable,
        ParaProfImageInterface {

    public StatWindowPanel(ParaProfTrial pptrial, int nodeID, int contextID, int threadID, StatWindow window,
            boolean userEventWindow) {

        setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
        setBackground(Color.white);

        trial = pptrial;
        this.window = window;
        this.userEventWindow = userEventWindow;

        //Add this object as a mouse listener.
        addMouseListener(this);

        //Add items to the popup menu.
        if (userEventWindow) {
            JMenuItem userEventDetailsItem = new JMenuItem("Show User Event Details");
            userEventDetailsItem.addActionListener(this);
            popup.add(userEventDetailsItem);

            JMenuItem changeColorItem = new JMenuItem("Change User Event Color");
            changeColorItem.addActionListener(this);
            popup.add(changeColorItem);


        } else {
            JMenuItem functionDetailsItem = new JMenuItem("Show Function Details");
            functionDetailsItem.addActionListener(this);
            popup.add(functionDetailsItem);

            JMenuItem jMenuItem = new JMenuItem("Show Function Histogram");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            JMenuItem changeColorItem = new JMenuItem("Change Function Color");
            changeColorItem.addActionListener(this);
            popup.add(changeColorItem);

        }

        JMenuItem maskColorItem = new JMenuItem("Reset to Generic Color");
        maskColorItem.addActionListener(this);
        popup.add(maskColorItem);

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

    public static String getUserEventStatStringHeading() {

        int w = 18;
        return UtilFncs.pad("NumSamples", w) + UtilFncs.pad("Max", w) + UtilFncs.pad("Min", w)
                + UtilFncs.pad("Mean", w) + UtilFncs.pad("Std. Dev", w);

    }

    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {

        list = window.getData();

        //With group support present, it is possible that the number of functions in
        //our data list is zero. If so, just return.
        if ((list.size()) == 0)
            return;

        //######
        //Some declarations.
        //######
        PPFunctionProfile ppFunctionProfile = null;
        PPUserEventProfile ppUserEventProfile = null;

        Color tmpColor;
        int yCoord = 0;
        String tmpString = null;
        String dashString = "";
        int tmpXWidthCalc = 0;
        //######
        //Some declarations.
        //######

        //In this window, a Monospaced font has to be used. This will
        // probably
        // not be the same
        //font as the rest of ParaProf. As a result, some extra work will
        // have
        // to be done to calculate
        //spacing.
        int fontSize = trial.getPreferences().getBarHeight();
        spacing = trial.getPreferences().getBarSpacing();
        //Create font.
        monoFont = new Font("Monospaced", trial.getPreferences().getFontStyle(), fontSize);
        //Compute the font metrics.
        fmMonoFont = g2D.getFontMetrics(monoFont);
        maxFontAscent = fmMonoFont.getMaxAscent();
        maxFontDescent = fmMonoFont.getMaxDescent();
        g2D.setFont(monoFont);
        FontRenderContext frc = g2D.getFontRenderContext();

        if (spacing <= (maxFontAscent + maxFontDescent)) {
            spacing = spacing + 1;
        }

        //######
        //Draw the header if required.
        //######
        if (drawHeader) {
            //FontRenderContext frc2 = g2D.getFontRenderContext();
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

        if (userEventWindow) {
            tmpString = StatWindowPanel.getUserEventStatStringHeading();

        } else {
            if (trial.isTimeMetric())
                tmpString = PPFunctionProfile.getStatStringHeading("Time");
            else
                tmpString = PPFunctionProfile.getStatStringHeading("Counts");
        }

        //Calculate the name position.
        int namePosition = fmMonoFont.stringWidth(tmpString) + 20; //Note
        // that
        // 20
        // is the begin
        // draw
        // position.

        //Now append "name" to the end of the string.
        tmpString = tmpString + "Name";
        int tmpInt = tmpString.length();

        for (int i = 0; i < tmpInt; i++) {
            dashString = dashString + "-";
        }

        g2D.setColor(Color.black);

        //Draw the first dashed string.
        yCoord = yCoord + spacing;
        g2D.drawString(dashString, 20, yCoord);
        yCoord = yCoord + spacing + 10;

        //Draw the heading.
        g2D.drawString(tmpString, 20, yCoord);
        yCoord = yCoord + spacing + 10;

        //Draw the second dashed string.
        g2D.drawString(dashString, 20, yCoord);

        if (toScreen)
            startLocation = yCoord;

        //Set up some panel dimensions.
        newYPanelSize = yCoord + ((list.size() + 1) * spacing);

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

            if (startElement > (list.size() - 1))
                startElement = (list.size() - 1);

            if (endElement > (list.size() - 1))
                endElement = (list.size() - 1);

            if (toScreen)
                yCoord = yCoord + (startElement * spacing);
        } else {
            startElement = 0;
            endElement = ((list.size()) - 1);
        }

        for (int i = startElement; i <= endElement; i++) {
            if (userEventWindow) {
                ppUserEventProfile = (PPUserEventProfile) list.elementAt(i);
                tmpString = ppUserEventProfile.getUserEventStatString(ParaProf.defaultNumberPrecision);
            } else {
                ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);
                tmpString = ppFunctionProfile.getStatString(window.units());
            }

            yCoord = yCoord + spacing;

            g2D.setColor(Color.black);

            int highLightColor = -1;
            if (userEventWindow) {
                UserEvent userEvent = trial.getColorChooser().getHighlightedUserEvent();
                if (userEvent != null)
                    highLightColor = userEvent.getID();
            } else {
                Function function = trial.getColorChooser().getHighlightedFunction();
                if (function != null)
                    highLightColor = function.getID();
            }
            if ((userEventWindow && ppUserEventProfile.getUserEvent().getID() == highLightColor)
                    || (!userEventWindow && ppFunctionProfile.getFunction().getID() == highLightColor)) {
                g2D.setColor(trial.getColorChooser().getHighlightColor());
                (new TextLayout(tmpString, monoFont, frc)).draw(g2D, 20, yCoord);

                if (userEventWindow) {
                    g2D.setColor(ppUserEventProfile.getColor());
                    (new TextLayout(ppUserEventProfile.getUserEventName(), monoFont, frc)).draw(g2D,
                            namePosition, yCoord);
                } else {
                    g2D.setColor(ppFunctionProfile.getColor());
                    (new TextLayout(ppFunctionProfile.getFunctionName(), monoFont, frc)).draw(g2D,
                            namePosition, yCoord);
                }

            } else if (!userEventWindow
                    && (ppFunctionProfile.isGroupMember(trial.getColorChooser().getHighlightedGroup()))) {
                g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
                (new TextLayout(tmpString, monoFont, frc)).draw(g2D, 20, yCoord);
                //g2D.drawString(tmpString, 20, yCoord);
                g2D.setColor(ppFunctionProfile.getColor());
                (new TextLayout(ppFunctionProfile.getFunctionName(), monoFont, frc)).draw(g2D, namePosition,
                        yCoord);
            } else {
                (new TextLayout(tmpString, monoFont, frc)).draw(g2D, 20, yCoord);
                //g2D.drawString(tmpString, 20, yCoord);
                if (userEventWindow) {
                    g2D.setColor(ppUserEventProfile.getColor());
                    (new TextLayout(ppUserEventProfile.getUserEventName(), monoFont, frc)).draw(g2D,
                            namePosition, yCoord);
                } else {
                    g2D.setColor(ppFunctionProfile.getColor());
                    (new TextLayout(ppFunctionProfile.getFunctionName(), monoFont, frc)).draw(g2D,
                            namePosition, yCoord);
                }

            }

            if (userEventWindow) {
                //Figure out how wide that string was for x coord reasons.
                if (tmpXWidthCalc < (20 + namePosition + fmMonoFont.stringWidth(ppUserEventProfile.getUserEventName()))) {
                    tmpXWidthCalc = (20 + namePosition + fmMonoFont.stringWidth(ppUserEventProfile.getUserEventName()));
                }
            } else {
                //Figure out how wide that string was for x coord reasons.
                if (tmpXWidthCalc < (20 + namePosition + fmMonoFont.stringWidth(ppFunctionProfile.getFunctionName()))) {
                    tmpXWidthCalc = (20 + namePosition + fmMonoFont.stringWidth(ppFunctionProfile.getFunctionName()));
                }
            }
        }
        //Resize the panel if needed.
        if ((newYPanelSize >= yPanelSize) || (tmpXWidthCalc >= xPanelSize)) {
            yPanelSize = newYPanelSize + 1;
            xPanelSize = tmpXWidthCalc + 1;
            revalidate();
        }
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            PPUserEventProfile ppUserEventProfile = null;

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Show Function Details")) {

                    if (clickedOnObject instanceof PPFunctionProfile) {
                        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        FunctionDataWindow tmpRef = new FunctionDataWindow(trial,
                                ppFunctionProfile.getFunction(), trial.getStaticMainWindow().getDataSorter());
                        trial.getSystemEvents().addObserver(tmpRef);
                        tmpRef.show();
                    }
                } else if (arg.equals("Show Function Histogram")) {
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        PPFunctionProfile ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        HistogramWindow hw = new HistogramWindow(trial, window.getDataSorter(),
                                ppFunctionProfile.getFunction());
                        trial.getSystemEvents().addObserver(hw);
                        hw.show();
                    }

                } else if (arg.equals("Show User Event Details")) {

                    if (clickedOnObject instanceof PPUserEventProfile) {
                        ppUserEventProfile = (PPUserEventProfile) clickedOnObject;
                        //Bring up an expanded data window for this user event and highlight it
                        trial.getColorChooser().setHighlightedUserEvent(ppUserEventProfile.getUserEvent());
                        UserEventWindow tmpRef = new UserEventWindow(trial, ppUserEventProfile.getUserEvent(),
                                trial.getStaticMainWindow().getDataSorter());
                        trial.getSystemEvents().addObserver(tmpRef);
                        tmpRef.show();
                    }
                } else if (arg.equals("Change Function Color")) {
                    Function function = null;
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        function = ((PPFunctionProfile) clickedOnObject).getFunction();

                        Color tmpCol = function.getColor();
                        JColorChooser tmpJColorChooser = new JColorChooser();
                        tmpCol = JColorChooser.showDialog(this, "Please select a new color", tmpCol);
                        if (tmpCol != null) {
                            function.setSpecificColor(tmpCol);
                            function.setColorFlag(true);

                            trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                        }
                    }
                } else if (arg.equals("Change User Event Color")) {
                    UserEvent userEvent = null;
                    if (clickedOnObject instanceof PPUserEventProfile)
                        userEvent = ((PPUserEventProfile) clickedOnObject).getUserEvent();

                    Color tmpCol = userEvent.getColor();
                    JColorChooser tmpJColorChooser = new JColorChooser();
                    tmpCol = JColorChooser.showDialog(this, "Please select a new color", tmpCol);
                    if (tmpCol != null) {
                        userEvent.setSpecificColor(tmpCol);
                        userEvent.setColorFlag(true);

                        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                    }
                } else if (arg.equals("Reset to Generic Color")) {

                    //Get the clicked on object.
                    if (clickedOnObject instanceof PPFunctionProfile) {
                        Function f = ((PPFunctionProfile) clickedOnObject).getFunction();
                        f.setColorFlag(false);

                    }

                    if (clickedOnObject instanceof PPUserEventProfile) {
                        UserEvent ue = ((PPUserEventProfile) clickedOnObject).getUserEvent();
                        ue.setColorFlag(false);
                    }
                    trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void mouseClicked(MouseEvent evt) {
        try {

            String tmpString = null;

            //Get the location of the mouse.
            //Get the location of the mouse.
            int xCoord = evt.getX();
            int yCoord = evt.getY();

            int fontSize = trial.getPreferences().getBarHeight();

            //Get the number of times clicked.
            int clickCount = evt.getClickCount();

            int tmpInt1 = yCoord - startLocation;
            int tmpInt2 = tmpInt1 / spacing;
            int tmpInt3 = (tmpInt2 + 1) * spacing;
            int tmpInt4 = tmpInt3 - maxFontAscent;

            if ((tmpInt1 >= tmpInt4) && (tmpInt1 <= tmpInt3)) {
                if (tmpInt2 < (list.size())) {
                    if (userEventWindow) {
                        PPUserEventProfile ppUserEventProfile = null;

                        ppUserEventProfile = (PPUserEventProfile) list.elementAt(tmpInt2);
                        if ((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0) {
                            clickedOnObject = ppUserEventProfile;
                            popup.show(this, evt.getX(), evt.getY());
                        } else {
                            trial.getColorChooser().toggleHighlightedUserEvent(
                                    ppUserEventProfile.getUserEvent());
                        }

                    } else {
                        PPFunctionProfile ppFunctionProfile = null;
                        ppFunctionProfile = (PPFunctionProfile) list.elementAt(tmpInt2);
                        if ((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0) {
                            clickedOnObject = ppFunctionProfile;
                            popup.show(this, evt.getX(), evt.getY());
                        } else {
                            trial.getColorChooser().toggleHighlightedFunction(ppFunctionProfile.getFunction());
                        }

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

    public Dimension getPreferredSize() {
        return new Dimension(xPanelSize, (yPanelSize + 10));
    }

    //Instance data.
    private int xPanelSize = 800;
    private int yPanelSize = 600;
    private int newXPanelSize = 0;
    private int newYPanelSize = 0;

    //Some drawing details.
    private int startLocation = 0;
    private int maxFontAscent = 0;
    private int maxFontDescent = 0;
    private int spacing = 0;

    private ParaProfTrial trial = null;
    private StatWindow window = null;
    private boolean userEventWindow;
    private Vector list = null;

    private Font monoFont = null;
    private FontMetrics fmMonoFont = null;

    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    private int lastHeaderEndPosition = 0;

}