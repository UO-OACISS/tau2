package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.print.*;
import java.awt.geom.*;
import javax.swing.*;
import edu.uoregon.tau.dms.dss.*;

/**
 * LedgerWindowPanel This object represents the ledger window panel.
 * 
 * <P>
 * CVS $Id: LedgerWindowPanel.java,v 1.5 2005/01/06 22:49:43 amorris Exp $
 * </P>
 * 
 * @author Robert Bell, Alan Morris
 * @version $Revision: 1.5 $
 * @see LedgerDataElement
 * @see LedgerWindow
 */
public class LedgerWindowPanel extends JPanel implements ActionListener, MouseListener, Printable,
        ParaProfImageInterface {


    public void setupMenus() {
        JMenuItem jMenuItem = null;
        switch (windowType) {
        case FUNCTION_LEDGER:
            jMenuItem = new JMenuItem("Show Function Details");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Change Function Color");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Reset to Generic Color");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            break;
        case GROUP_LEDGER:

            jMenuItem = new JMenuItem("Change Group Color");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Reset to Generic Color");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Show This Group Only");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Show All Groups Except This One");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Show All Groups");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            break;
        case USEREVENT_LEDGER:
            jMenuItem = new JMenuItem("Show User Event Details");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Change User Event Color");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Reset to Generic Color");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            break;
        }

    }

    public LedgerWindowPanel(ParaProfTrial trial, LedgerWindow window, int windowType) {

        setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
        setBackground(Color.white);

        this.trial = trial;
        this.window = window;
        this.windowType = windowType;

        //Add this object as a mouse listener.
        addMouseListener(this);

        setupMenus();

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

    public void renderIt(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        list = window.getData();

        int xCoord = 0;
        int yCoord = 0;
        int barXCoord = 0;
        int tmpXWidthCalc = 0;

        //To make sure the bar details are set, this
        //method must be called.
        trial.getPreferences().setBarDetails(g2D);

        //Now safe to grab spacing and bar heights.
        barSpacing = trial.getPreferences().getBarSpacing();
        barHeight = trial.getPreferences().getBarHeight();

        //Obtain the font and its metrics.
        Font font = new Font(trial.getPreferences().getParaProfFont(), trial.getPreferences().getFontStyle(),
                barHeight);
        g2D.setFont(font);
        FontMetrics fmFont = g2D.getFontMetrics(font);

        
        if (!widthSet) {  // only do this once
            for (int i = 0; i < list.size(); i++) {
                LedgerDataElement lde = (LedgerDataElement) list.get(i);
                if (lde.getName() != null) {
                    int tmpWidth = 5 + barHeight + (fmFont.stringWidth(lde.getName()));

                    //Figure out how wide that string was for x coord reasons.
                    if (xPanelSize < tmpWidth) {
                        xPanelSize = (tmpWidth + 11);
                    }
                }
            }
            widthSet = true;
        }

        if (resizePanel(fmFont, barXCoord) && toScreen) {
            this.revalidate();
            return;
        }

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
                yCoord = yCoord + ((startElement - 1) * barSpacing);
        } else {
            startElement = 0;
            endElement = ((list.size()) - 1);
        }

        xCoord = 5;
        yCoord = yCoord + (barSpacing);

        for (int i = startElement; i <= endElement; i++) {
            LedgerDataElement lde = (LedgerDataElement) list.get(i);

            if (lde.getName() != null) {

                //For consistency in drawing, the y coord is updated at the
                // beginning of the loop.
                yCoord = yCoord + (barSpacing);

                //First draw the color box.
                g2D.setColor(lde.getColor());
                g2D.fillRect(xCoord, (yCoord - barHeight), barHeight, barHeight);

                if (lde.isHighlighted(trial.getColorChooser())) {
                    g2D.setColor(lde.getHighlightColor(trial.getColorChooser()));
                    g2D.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
                    g2D.drawRect(xCoord + 1, (yCoord - barHeight) + 1, barHeight - 2, barHeight - 2);
                } else {
                    g2D.setColor(Color.black);
                    g2D.drawRect(xCoord, (yCoord - barHeight), barHeight, barHeight);
                }

                //Update the xCoord to draw the name.
                xCoord = xCoord + (barHeight + 10);
                //Reset the drawing color to the text color ... in this
                // case, black.
                g2D.setColor(Color.black);

                //Draw the name.
                String s = lde.getName();

                g2D.drawString(s, xCoord, yCoord);

                //Figure out how wide that string was for x coord
                // reasons.
                int tmpWidth = 5 + barHeight + (fmFont.stringWidth(s));

                //Figure out how wide that string was for x coord reasons.
                if (tmpXWidthCalc < tmpWidth) {
                    tmpXWidthCalc = (tmpWidth + 11);
                }

                // only set the boundaries (for clicking) if we are drawing to the screen
                if (toScreen)
                    lde.setDrawCoords(0, tmpWidth, (yCoord - barHeight), yCoord);

                //Reset the xCoord.
                xCoord = xCoord - (barHeight + 10);

            }
        }

        //            //Resize the panel if needed.
        //            if (((yCoord >= yPanelSize) || (tmpXWidthCalc >= xPanelSize)) && instruction == 0) {
        //                yPanelSize = yCoord + 1;
        //                xPanelSize = tmpXWidthCalc + 1;
        //
        //                revalidate();
        //            }
    }

    //This method sets both xPanelSize and yPanelSize.
    private boolean resizePanel(FontMetrics fmFont, int barXCoord) {
        boolean resized = false;
        int newYPanelSize = ((window.getData().size())) * barSpacing;

        if ((newYPanelSize != yPanelSize)) {
            yPanelSize = newYPanelSize;
            this.setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
            resized = false;
        }
        return resized;
    }

    public void actionPerformed(ActionEvent evt) {

        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();

                if (clickedOnObject instanceof LedgerDataElement) {
                    LedgerDataElement lde = (LedgerDataElement) clickedOnObject;

                    if (arg.equals("Show Function Details")) {
                        // Highlight the function and bring up the Function Data
                        // Window
                        trial.getColorChooser().setHighlightedFunction(lde.getFunction());
                        FunctionDataWindow tmpRef = new FunctionDataWindow(trial, lde.getFunction(),
                                trial.getStaticMainWindow().getDataSorter());
                        trial.getSystemEvents().addObserver(tmpRef);
                        tmpRef.show();

                    } else if (arg.equals("Show User Event Details")) {
                        // Highlight the user event and bring up the User Event
                        // Window
                        trial.getColorChooser().setHighlightedUserEvent(lde.getUserEvent());
                        UserEventWindow tmpRef = new UserEventWindow(trial, lde.getUserEvent(),
                                trial.getStaticMainWindow().getDataSorter());
                        trial.getSystemEvents().addObserver(tmpRef);
                        tmpRef.show();
                    } else if ((arg.equals("Change Function Color")) || (arg.equals("Change User Event Color"))
                            || (arg.equals("Change Group Color"))) {

                        Color color = lde.getColor();
                        // JColorChooser tmpJColorChooser = new JColorChooser();
                        color = JColorChooser.showDialog(this, "Please select a new color", color);
                        if (color != null) {
                            lde.setSpecificColor(color);
                            lde.setColorFlag(true);
                            trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                        }
                    } else if (arg.equals("Reset to Generic Color")) {
                        lde.setColorFlag(false);
                        trial.getSystemEvents().updateRegisteredObjects("colorEvent");
                    } else if (arg.equals("Show This Group Only")) {
                        trial.setSelectedGroup(lde.getGroup());
                        trial.setGroupFilter(1);
                        trial.getSystemEvents().updateRegisteredObjects("dataEvent");
                    } else if (arg.equals("Show All Groups Except This One")) {
                        trial.setSelectedGroup(lde.getGroup());
                        trial.setGroupFilter(2);
                        trial.getSystemEvents().updateRegisteredObjects("dataEvent");
                    } else if (arg.equals("Show All Groups")) {
                        trial.setSelectedGroup(null);
                        trial.setGroupFilter(0);
                        trial.getSystemEvents().updateRegisteredObjects("dataEvent");
                    }
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void mouseClicked(MouseEvent evt) {
        try {
            if (list == null)
                return;

            //Get the location of the mouse.
            int xCoord = evt.getX();
            int yCoord = evt.getY();

            //Get the number of times clicked.
            int clickCount = evt.getClickCount();

            for (Enumeration e1 = list.elements(); e1.hasMoreElements();) {
                LedgerDataElement lde = (LedgerDataElement) e1.nextElement();

                if (yCoord <= (lde.getYEnd())) {
                    if ((yCoord >= (lde.getYBeg())) && (xCoord >= (lde.getXBeg()))
                            && (xCoord <= (lde.getXEnd()))) {
                        if ((evt.getModifiers() & InputEvent.BUTTON1_MASK) == 0) {
                            // not left click (middle and right)
                            clickedOnObject = lde;
                            popup.show(this, evt.getX(), evt.getY());
                            return;
                        } else { // left click
                            if (windowType == USEREVENT_LEDGER) {
                                trial.getColorChooser().toggleHighlightedUserEvent(lde.getUserEvent());
                            } else if (windowType == GROUP_LEDGER) {
                                trial.getColorChooser().toggleHighlightedGroup(lde.getGroup());
                            } else {
                                trial.getColorChooser().toggleHighlightedFunction(lde.getFunction());
                            }
                        }
                        //Nothing more to do ... return.
                        return;
                    } else {
                        /*
                         * If we get here, it means that we are outside the draw
                         * area. We are either to the left or right of the draw
                         * area, or just above it. It is better to return here
                         * as we do not want the system to cycle through the
                         * rest of the objects, which would be pointless as we
                         * know that it will not be one of the others.
                         * Significantly improves performance.
                         */
                        return;
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

    public Dimension getImageSize(boolean fullScreen, boolean prependHeader) {
        if (fullScreen)
            return this.getPreferredSize();
        else
            return window.getSize();
    }


    public Dimension getPreferredSize() {
        return new Dimension((xPanelSize + 10), (yPanelSize + 10));
    }

    //Instance data.
    private int xPanelSize = 300;
    private int yPanelSize = 400;

    private int barHeight = -1;
    private int barSpacing = -1;

    private ParaProfTrial trial = null;
    private LedgerWindow window = null;

    

    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    Vector list = null;

    private boolean widthSet = false;


    
    public static final int FUNCTION_LEDGER = 0;
    public static final int GROUP_LEDGER = 1;
    public static final int USEREVENT_LEDGER = 2;
    private int windowType = -1;


}