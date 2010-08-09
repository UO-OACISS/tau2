package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.util.Enumeration;
import java.util.Vector;

import javax.swing.JColorChooser;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JSeparator;

import edu.uoregon.tau.common.ImageExport;

/**
 * LedgerWindowPanel This object represents the ledger window panel.
 * 
 * <P>
 * CVS $Id: LedgerWindowPanel.java,v 1.10 2007/01/31 19:38:08 amorris Exp $
 * </P>
 * 
 * @author Robert Bell, Alan Morris
 * @version $Revision: 1.10 $
 * @see LedgerDataElement
 * @see LedgerWindow
 */
public class LedgerWindowPanel extends JPanel implements ActionListener, MouseListener, Printable, ImageExport {

    /**
	 * 
	 */
	private static final long serialVersionUID = -3708637652568786073L;
	private int xPanelSize = 300;
    private int yPanelSize = 400;

    private int barHeight = -1;

    private int rowHeight = -1;

    private ParaProfTrial ppTrial = null;
    private LedgerWindow window = null;

    private JPopupMenu popup = new JPopupMenu();
    private Object clickedOnObject = null;

    private Vector<LedgerDataElement> list = new Vector<LedgerDataElement>();

    private boolean widthSet = false;
    private int windowType = -1;

    public LedgerWindowPanel(ParaProfTrial ppTrial, LedgerWindow window, int windowType) {

        setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
        setBackground(Color.white);

        this.ppTrial = ppTrial;
        this.window = window;
        this.windowType = windowType;

        //Add this object as a mouse listener.
        addMouseListener(this);

        setupMenus();

        //Schedule a repaint of this panel.
        this.repaint();

    }

    public void setupMenus() {
        JMenuItem jMenuItem = null;

        if (windowType == LedgerWindow.GROUP_LEGEND) {

            //            jMenuItem = new JMenuItem("Change Group Color");
            //            jMenuItem.addActionListener(this);
            //            popup.add(jMenuItem);
            //
            //            jMenuItem = new JMenuItem("Reset to Generic Color");
            //            jMenuItem.addActionListener(this);
            //            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Hide This Group");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Show This Group");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            popup.add(new JSeparator());

            jMenuItem = new JMenuItem("Show This Group Only");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Show All Groups Except This One");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

            jMenuItem = new JMenuItem("Show All Groups");
            jMenuItem.addActionListener(this);
            popup.add(jMenuItem);

        }

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
            ParaProfUtils.handleException(e);
            return NO_SUCH_PAGE;
        }
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        list = window.getData();

        int xCoord = 0;
        int yCoord = 0;
        int barXCoord = 0;
        int tmpXWidthCalc = 0;

        //Now safe to grab spacing and bar heights.
        barHeight = ppTrial.getPreferencesWindow().getFontSize();

        //Obtain the font and its metrics.
        Font font = ParaProf.preferencesWindow.getFont();
        g2D.setFont(font);

        FontMetrics fmFont = g2D.getFontMetrics(font);

        rowHeight = fmFont.getHeight();

        if (!widthSet) { // only do this once
            for (int i = 0; i < list.size(); i++) {
                LedgerDataElement lde = list.get(i);
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

        //      determine which elements to draw (clipping)
        int[] clips = ParaProfUtils.computeClipping(g2D.getClipBounds(), window.getViewRect(), toScreen, fullWindow, list.size(),
                rowHeight, yCoord);
        int startElement = clips[0];
        int endElement = clips[1];
        yCoord = clips[2];

        xCoord = 5;

        for (int i = startElement; i <= endElement; i++) {
            LedgerDataElement lde = list.get(i);

            if (lde.getName() != null) {

                //For consistency in drawing, the y coord is updated at the
                // beginning of the loop.
                yCoord = yCoord + rowHeight;

                //First draw the color box.
                g2D.setColor(lde.getColor());
                g2D.fillRect(xCoord, (yCoord - barHeight), barHeight, barHeight);

                if (lde.isHighlighted(ppTrial)) {
                    g2D.setColor(lde.getHighlightColor(ppTrial.getColorChooser()));
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

                if (lde.isShown(ppTrial)) {
                    g2D.setColor(Color.black);
                } else {
                    g2D.setColor(new Color(150, 150, 150));
                }

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
                    lde.setDrawCoords(0, xCoord + tmpWidth, (yCoord - barHeight), yCoord);

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
        int newYPanelSize = ((window.getData().size())) * rowHeight;

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

                    if (arg.equals("Change Group Color")) {

                        Color color = lde.getColor();
                        color = JColorChooser.showDialog(this, "Please select a new color", color);
                        if (color != null) {
                            lde.setSpecificColor(color);
                            lde.setColorFlag(true);
                            ppTrial.updateRegisteredObjects("colorEvent");
                        }
                    } else if (arg.equals("Reset to Generic Color")) {
                        lde.setColorFlag(false);
                        ppTrial.updateRegisteredObjects("colorEvent");
                    } else if (arg.equals("Show This Function")) {
                        ppTrial.showFunction(lde.getFunction());
                    } else if (arg.equals("Hide This Function")) {
                        ppTrial.hideFunction(lde.getFunction());
                    } else if (arg.equals("Show This Group")) {
                        ppTrial.showGroup(lde.getGroup());
                    } else if (arg.equals("Hide This Group")) {
                        ppTrial.hideGroup(lde.getGroup());
                    } else if (arg.equals("Show This Group Only")) {
                        ppTrial.showGroupOnly(lde.getGroup());
                    } else if (arg.equals("Show All Groups Except This One")) {
                        ppTrial.showAllExcept(lde.getGroup());
                    } else if (arg.equals("Show All Groups")) {
                        ppTrial.showAllExcept(null);
                        ppTrial.updateRegisteredObjects("dataEvent");
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
            //int clickCount = evt.getClickCount();

            for (Enumeration<LedgerDataElement> e1 = list.elements(); e1.hasMoreElements();) {
                LedgerDataElement lde = e1.nextElement();

                if (yCoord <= (lde.getYEnd())) {
                    if ((yCoord >= (lde.getYBeg())) && (xCoord >= (lde.getXBeg())) && (xCoord <= (lde.getXEnd()))) {
                        if (ParaProfUtils.rightClick(evt)) {
                            // not left click (middle and right)
                            clickedOnObject = lde;

                            if (windowType == LedgerWindow.FUNCTION_LEGEND || windowType == LedgerWindow.PHASE_LEGEND) {
                                JPopupMenu popup = ParaProfUtils.createFunctionClickPopUp(ppTrial, lde.getFunction(),
                                        ppTrial.getDataSource().getTotalData(), this);
                                popup.add(new JSeparator());

                                JMenuItem jMenuItem = new JMenuItem("Hide This Function");
                                jMenuItem.addActionListener(this);
                                popup.add(jMenuItem);

                                jMenuItem = new JMenuItem("Show This Function");
                                jMenuItem.addActionListener(this);
                                popup.add(jMenuItem);
                                popup.show(this, evt.getX(), evt.getY());
                            } else if (windowType == LedgerWindow.USEREVENT_LEGEND) {
                                ParaProfUtils.handleUserEventClick(ppTrial, lde.getUserEvent(), this, evt);
                            } else {
                                popup.show(this, evt.getX(), evt.getY());
                            }
                            return;
                        } else { // left click
                            if (windowType == LedgerWindow.USEREVENT_LEGEND) {
                                ppTrial.toggleHighlightedUserEvent(lde.getUserEvent());
                            } else if (windowType == LedgerWindow.GROUP_LEGEND) {
                                ppTrial.toggleHighlightedGroup(lde.getGroup());
                            } else {
                                ppTrial.toggleHighlightedFunction(lde.getFunction());
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

    public void help(boolean display) {
        // TODO Auto-generated method stub

    }

    public Rectangle getViewRect() {
        // TODO Auto-generated method stub
        return null;
    }

}