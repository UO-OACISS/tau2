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
    public StatWindowPanel() {
        try {
            setSize(new java.awt.Dimension(xPanelSize, yPanelSize));

            //Schedule a repaint of this panel.
            this.repaint();
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SWP01");
        }

    }

    public StatWindowPanel(ParaProfTrial inParaProfTrial, int nodeID, int contextID, int threadID,
            StatWindow sWindow, boolean userEventWindow, boolean debug) {

        try {
            setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
            setBackground(Color.white);

            trial = inParaProfTrial;
            this.nodeID = nodeID;
            this.contextID = contextID;
            this.threadID = threadID;
            this.sWindow = sWindow;
            this.userEventWindow = userEventWindow;
            this.debug = debug;

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

                JMenuItem changeColorItem = new JMenuItem("Change Function Color");
                changeColorItem.addActionListener(this);
                popup.add(changeColorItem);
            }

            JMenuItem maskColorItem = new JMenuItem("Reset to Generic Color");
            maskColorItem.addActionListener(this);
            popup.add(maskColorItem);

            this.repaint();
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "SWP02");
        }

    }

    public void paintComponent(Graphics g) {
        try {
            super.paintComponent(g);
            renderIt((Graphics2D) g, 0, false);
        } catch (Exception e) {
            System.out.println(e);
            UtilFncs.systemError(e, null, "SWP03");
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
                System.out.println("StatWindowPanel.renderIt(...)");
                System.out.println("####################################");
            }

            list = sWindow.getData();

            //With group support present, it is possible that the number of
            // mappings in
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
            if (header) {
                //FontRenderContext frc2 = g2D.getFontRenderContext();
                Insets insets = this.getInsets();
                yCoord = yCoord + (spacing);
                String headerString = sWindow.getHeaderString();
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
                tmpString = UserEventProfile.getUserEventStatStringHeading();
                
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

            if (instruction == 0)
                startLocation = yCoord;

            //Set up some panel dimensions.
            newYPanelSize = yCoord + ((list.size() + 1) * spacing);

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
                    viewRect = sWindow.getViewRect();
                    yBeg = (int) viewRect.getY();
                    yEnd = (int) (yBeg + viewRect.getHeight());
                    /*
                     * System.out.println("Viewing Rectangle: xBeg,xEnd:
                     * "+viewRect.getX()+","+((viewRect.getX())+(viewRect.getWidth()))+ "
                     * yBeg,yEnd:
                     * "+viewRect.getY()+","+((viewRect.getY())+(viewRect.getHeight())));
                     */
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

                if (instruction == 0)
                    yCoord = yCoord + (startElement * spacing);
            } else if (instruction == 2 || instruction == 3) {
                startElement = 0;
                endElement = ((list.size()) - 1);
            }

            for (int i = startElement; i <= endElement; i++) {
                if (userEventWindow) {
                    ppUserEventProfile = (PPUserEventProfile) list.elementAt(i);
                    tmpString = ppUserEventProfile.getUserEventStatString(ParaProf.defaultNumberPrecision);
                } else {
                    ppFunctionProfile = (PPFunctionProfile) list.elementAt(i);
                    tmpString = ppFunctionProfile.getStatString(sWindow.units());
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
                        (new TextLayout(ppUserEventProfile.getUserEventName(), monoFont, frc)).draw(
                                g2D, namePosition, yCoord);
                    } else {
                        g2D.setColor(ppFunctionProfile.getColor());
                        (new TextLayout(ppFunctionProfile.getFunctionName(), monoFont, frc)).draw(
                                g2D, namePosition, yCoord);
                    }

                } else if (!userEventWindow
                        && (ppFunctionProfile.isGroupMember(trial.getColorChooser().getHighlightedGroup()))) {
                    g2D.setColor(trial.getColorChooser().getGroupHighlightColor());
                    (new TextLayout(tmpString, monoFont, frc)).draw(g2D, 20, yCoord);
                    //g2D.drawString(tmpString, 20, yCoord);
                    g2D.setColor(ppFunctionProfile.getColor());
                    (new TextLayout(ppFunctionProfile.getFunctionName(), monoFont, frc)).draw(
                            g2D, namePosition, yCoord);
                } else {
                    (new TextLayout(tmpString, monoFont, frc)).draw(g2D, 20, yCoord);
                    //g2D.drawString(tmpString, 20, yCoord);
                    if (userEventWindow) {
                        g2D.setColor(ppUserEventProfile.getColor());
                        (new TextLayout(ppUserEventProfile.getUserEventName(), monoFont, frc)).draw(
                                g2D, namePosition, yCoord);
                    } else {
                        g2D.setColor(ppFunctionProfile.getColor());
                        (new TextLayout(ppFunctionProfile.getFunctionName(), monoFont, frc)).draw(
                                g2D, namePosition, yCoord);
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
        } catch (Exception e) {
            UtilFncs.systemError(e, null, "TSWP03");
        }
    }    //####################################
    //Interface code.
    //####################################

    //######
    //ActionListener.
    //######
    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            PPFunctionProfile ppFunctionProfile = null;
            PPUserEventProfile ppUserEventProfile = null;

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Show Function Details")) {

                    if (clickedOnObject instanceof PPFunctionProfile) {
                        ppFunctionProfile = (PPFunctionProfile) clickedOnObject;
                        //Bring up an expanded data window for this function and highlight it
                        trial.getColorChooser().setHighlightedFunction(
                                ppFunctionProfile.getFunction());
                        FunctionDataWindow tmpRef = new FunctionDataWindow(trial,
                                ppFunctionProfile.getFunction(),
                                trial.getStaticMainWindow().getDataSorter(), this.debug());
                        trial.getSystemEvents().addObserver(tmpRef);
                        tmpRef.show();
                    }
                }
                if (arg.equals("Show User Event Details")) {

                    if (clickedOnObject instanceof PPUserEventProfile) {
                        ppUserEventProfile = (PPUserEventProfile) clickedOnObject;
                        //Bring up an expanded data window for this user event and highlight it
                        trial.getColorChooser().setHighlightedUserEvent(
                                ppUserEventProfile.getUserEvent());
                        UserEventWindow tmpRef = new UserEventWindow(trial,
                                ppUserEventProfile.getUserEvent(),
                                trial.getStaticMainWindow().getDataSorter(), this.debug());
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
                } else if (arg.equals("Change Userevent Color")) {

                    //TODO: I don't think this ever happens, does it?
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
            UtilFncs.systemError(e, null, "TSWP04");
        }
    }

    //######
    //End - ActionListener
    //######

    //######
    //MouseListener.
    //######
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
                            trial.getColorChooser().toggleHighlightedUserEvent(ppUserEventProfile.getUserEvent());
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
            UtilFncs.systemError(e, null, "TSWP05");
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
    //End - MouseListener.
    //######

    //######
    //ParaProfImageInterface
    //######
    public Dimension getImageSize(boolean fullScreen, boolean header) {
        Dimension d = null;
        if (fullScreen)
            d = this.getSize();
        else
            d = sWindow.getSize();
        d.setSize(d.getWidth(), d.getHeight() + lastHeaderEndPosition);
        return d;
    }

    //######
    //End - ParaProfImageInterface
    //######

    //####################################
    //End - Interface code.
    //####################################

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
    private int nodeID = -1;
    private int contextID = -1;
    private int threadID = -1;
    private StatWindow sWindow = null;
    private boolean userEventWindow;
    private Vector list = null;

    private Font monoFont = null;
    private FontMetrics fmMonoFont = null;

    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    private int lastHeaderEndPosition = 0;

    private boolean debug = false; //Off by default.private boolean debug =
    // false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################
}