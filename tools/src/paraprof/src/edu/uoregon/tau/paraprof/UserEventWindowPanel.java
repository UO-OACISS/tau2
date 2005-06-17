/*
 * 
 * UserEventWindowPanel.java
 * 
 * Title: ParaProf Author: Robert Bell Description:
 * 
 * Things to do:
 *  
 */

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.awt.font.*;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.text.AttributedCharacterIterator;
import java.text.AttributedString;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import javax.swing.*;

import edu.uoregon.tau.dms.dss.UserEvent;
import edu.uoregon.tau.dms.dss.UtilFncs;
import edu.uoregon.tau.paraprof.interfaces.ImageExport;

public class UserEventWindowPanel extends JPanel implements ActionListener, MouseListener, Printable,
        ImageExport {

    private String counterName = null;
    private UserEvent userEvent = null;
    private int barHeight = -1;
    private int barSpacing = -1;
    private int baseBarLength = 250;
    private int barLength = 0;
    private int textOffset = 60;
    private int maxXLength = 0;
    private boolean groupMember = false;
    private ParaProfTrial ppTrial = null;
    private UserEventWindow window = null;
    private List list = new ArrayList();
    private int xPanelSize = 0;
    private int yPanelSize = 0;
    private JPopupMenu popup = new JPopupMenu();
    private int lastHeaderEndPosition = 0;

    public UserEventWindowPanel(ParaProfTrial ppTrial, UserEvent userEvent, UserEventWindow uEWindow) {
        this.ppTrial = ppTrial;
        this.window = uEWindow;
        this.userEvent = userEvent;
        barLength = baseBarLength;

        //Want the background to be white.
        setBackground(Color.white);

        //Add this object as a mouse listener.
        addMouseListener(this);

        //Add items to the popu menu.
        JMenuItem jMenuItem = new JMenuItem("Change User Event Color");
        jMenuItem.addActionListener(this);
        popup.add(jMenuItem);

        jMenuItem = new JMenuItem("Reset to Generic Color");
        jMenuItem.addActionListener(this);
        popup.add(jMenuItem);
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

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {

        list = window.getData();

        //Some declarations.
        double value = 0.0;
        double maxValue = 0.0;
        int stringWidth = 0;
        int yCoord = 0;
        int barXCoord = barLength + textOffset;
        PPUserEventProfile ppUserEventProfile = null;

        //To make sure the bar details are set, this
        //method must be called.
        ppTrial.getPreferencesWindow().setBarDetails(g2D);

        //Now safe to grab spacing and bar heights.
        barSpacing = ppTrial.getPreferencesWindow().getBarSpacing();
        barHeight = ppTrial.getPreferencesWindow().getBarHeight();

        //Obtain the font and its metrics.
        Font font = new Font(ppTrial.getPreferencesWindow().getParaProfFont(),
                ppTrial.getPreferencesWindow().getFontStyle(), barHeight);
        g2D.setFont(font);
        FontMetrics fmFont = g2D.getFontMetrics(font);

        maxValue = window.getValueType().getMaxValue(userEvent);

        stringWidth = fmFont.stringWidth(UtilFncs.getOutputString(0, maxValue, ParaProf.defaultNumberPrecision)); //No units required in
        // this window. Thus pass
        // in 0 for type.
        barXCoord = barXCoord + stringWidth;

        //At this point we can determine the size this panel will
        //require. If we need to resize, don't do any more drawing,
        //just call revalidate.
        if (resizePanel(fmFont, barXCoord) && toScreen) {
            this.revalidate();
            return;
        }

        //determine which elements to draw (clipping)
        int[] clips = ParaProfUtils.computeClipping(g2D.getClipBounds(), window.getViewRect(), toScreen, fullWindow,
                list.size(), barSpacing, yCoord);
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

        for (int i = startElement; i <= endElement; i++) {
            ppUserEventProfile = (PPUserEventProfile) list.get(i);

            value = window.getValueType().getValue(ppUserEventProfile.getUserEventProfile());

            //For consistancy in drawing, the y coord is updated at the
            // beginning of the loop.
            yCoord = yCoord + (barSpacing);
            drawBar(g2D, fmFont, value, maxValue, "n,c,t " + (ppUserEventProfile.getNodeID()) + ","
                    + (ppUserEventProfile.getContextID()) + "," + (ppUserEventProfile.getThreadID()), barXCoord,
                    yCoord, barHeight, groupMember);
        }

    }

    private void drawBar(Graphics2D g2D, FontMetrics fmFont, double value, double maxValue, String text, int barXCoord,
            int yCoord, int barHeight, boolean groupMember) {

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
            g2D.setColor(userEvent.getColor());
            g2D.fillRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 1, barHeight - 1);

            if (userEvent == (ppTrial.getHighlightedUserEvent())) {
                g2D.setColor(ppTrial.getColorChooser().getUserEventHighlightColor());
                g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
                g2D.drawRect(barXCoord - xLength + 1, (yCoord - barHeight) + 1, xLength - 2, barHeight - 2);
            } else {
                g2D.setColor(Color.black);
                g2D.drawRect(barXCoord - xLength, (yCoord - barHeight), xLength, barHeight);
            }
        } else {
            if (userEvent == (ppTrial.getHighlightedUserEvent()))
                g2D.setColor(ppTrial.getColorChooser().getUserEventHighlightColor());
            else {
                g2D.setColor(userEvent.getColor());
            }
            g2D.fillRect((barXCoord - xLength), (yCoord - barHeight), xLength, barHeight);
        }

        //Draw the value next to the bar.
        g2D.setColor(Color.black);
        s = UtilFncs.getOutputString(0, value, ParaProf.defaultNumberPrecision); //Set
        // the unit value (first arg) to 0).
        //This will ensure that UtilFncs.getOutputString
        //Does the right thing. This is of course because
        //we do not have units in this display.
        stringWidth = fmFont.stringWidth(s);
        //Now draw the percent value to the left of the bar.
        stringStart = barXCoord - xLength - stringWidth - 5;
        g2D.drawString(s, stringStart, yCoord);
        g2D.drawString(text, (barXCoord + 5), yCoord);
    }

    public void actionPerformed(ActionEvent evt) {
        try {
            Object EventSrc = evt.getSource();

            if (EventSrc instanceof JMenuItem) {
                String arg = evt.getActionCommand();
                if (arg.equals("Change User Event Color")) {
                    Color tmpCol = userEvent.getColor();

                    //		    JColorChooser tmpJColorChooser = new JColorChooser();
                    tmpCol = JColorChooser.showDialog(this, "Please select a new color", tmpCol);
                    if (tmpCol != null) {
                        userEvent.setSpecificColor(tmpCol);
                        userEvent.setColorFlag(true);
                        ppTrial.getSystemEvents().updateRegisteredObjects("colorEvent");
                    }
                } else if (arg.equals("Reset to Generic Color")) {
                    userEvent.setColorFlag(false);
                    ppTrial.getSystemEvents().updateRegisteredObjects("colorEvent");
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    //Ok, now the mouse listeners for this panel.
    public void mouseClicked(MouseEvent evt) {
        try {
            //For the moment, I am just showing the popup menu anywhere.
            //For a future release, there will be more here.
            if (ParaProfUtils.rightClick(evt)) {
                popup.show(this, evt.getX(), evt.getY());
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

    //This method sets both xPanelSize and yPanelSize.
    private boolean resizePanel(FontMetrics fmFont, int barXCoord) {
        boolean resized = false;
        int newYPanelSize = ((window.getData().size()) + 2) * barSpacing + 10;
        int[] nct = ppTrial.getMaxNCTNumbers();
        String nctString = "n,c,t " + nct[0] + "," + nct[1] + "," + nct[2];
        ;
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

    public void setBarLength(int barLength) {
        this.barLength = Math.max(1, barLength);
        this.repaint();
    }

}