/*
 * 
 * StatWindowPanel.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Insets;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.font.FontRenderContext;
import java.awt.font.LineBreakMeasurer;
import java.awt.font.TextAttribute;
import java.awt.font.TextLayout;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.text.AttributedCharacterIterator;
import java.text.AttributedString;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import javax.swing.JPanel;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.UserEvent;
import edu.uoregon.tau.perfdmf.UtilFncs;

public class StatWindowPanel extends JPanel implements MouseListener, Printable, ImageExport {

    /**
	 * 
	 */
	private static final long serialVersionUID = -525538914597587767L;
	//Instance data.
    private int xPanelSize = 800;
    private int yPanelSize = 600;
    //private int newXPanelSize = 0;
    private int newYPanelSize = 0;

    //Some drawing details.
    private int startLocation = 0;
    private int maxFontAscent = 0;
    private int maxFontDescent = 0;
    private int spacing = 0;

    private ParaProfTrial ppTrial = null;
    private StatWindow window = null;
    private boolean userEventWindow;
    private List<Comparable> list = new ArrayList<Comparable>();

    private Font monoFont = null;
    private FontMetrics fmMonoFont = null;

    //private Object clickedOnObject = null;

    private int lastHeaderEndPosition = 0;

    private int maxLinePixelWidth = 0;
    private Searcher searcher;

    private int charWidth = 0;
    private int xOffset = 20;

    public StatWindowPanel(ParaProfTrial pptrial, StatWindow window, boolean userEventWindow) {

        setSize(new java.awt.Dimension(xPanelSize, yPanelSize));
        setBackground(Color.white);

        setAutoscrolls(true);
        searcher = new Searcher(this, window);
        addMouseListener(searcher);
        addMouseMotionListener(searcher);

        ppTrial = pptrial;
        this.window = window;
        this.userEventWindow = userEventWindow;

        //Add this object as a mouse listener.
        addMouseListener(this);


        this.repaint();

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

    public static String getUserEventStatStringHeading() {

        int w = 18;
        return UtilFncs.pad("Total", w) + UtilFncs.pad("NumSamples", w) + UtilFncs.pad("Max", w) + UtilFncs.pad("Min", w)
                + UtilFncs.pad("Mean", w) + UtilFncs.pad("Std. Dev", w);

    }

    public void setSearchLines(String headerString, String dashString) {

        if (searcher.getSearchLines() == null && list != null) {
            List<String> searchLines = new ArrayList<String>();
            searchLines.add(dashString);
            searchLines.add(headerString);
            searchLines.add(dashString);

            for (int i = 0; i < list.size(); i++) {
                String statString;
                String nameString;

                if (userEventWindow) {
                    nameString = ((PPUserEventProfile) list.get(i)).getUserEventName();
                    statString = ((PPUserEventProfile) list.get(i)).getUserEventStatString(ParaProf.defaultNumberPrecision);
                    statString = statString + nameString;
                } else {
                    nameString = ((PPFunctionProfile) list.get(i)).getDisplayName();
                    if (window.getPhase() != null) {
                        nameString = UtilFncs.getRightSide(nameString);
                    }
                    statString = ((PPFunctionProfile) list.get(i)).getStatString(window.units());
                    statString = statString + "   " + nameString;
                }

                maxLinePixelWidth = Math.max(maxLinePixelWidth, charWidth * statString.length() + xOffset);

                searchLines.add(statString);
            }

            searcher.setSearchLines(searchLines);
        }

    }

    private void setStatStringColor(Graphics2D g2D, PPUserEventProfile ppUserEventProfile,
            PPFunctionProfile ppFunctionProfile) {

        int highLightID = -1;
        if (userEventWindow) {
            UserEvent userEvent = ppTrial.getHighlightedUserEvent();
            if (userEvent != null)
                highLightID = userEvent.getID();
        } else {
            Function function = ppTrial.getHighlightedFunction();
            if (function != null)
                highLightID = function.getID();
        }

        if ((userEventWindow && ppUserEventProfile.getUserEvent().getID() == highLightID)
                || (!userEventWindow && ppFunctionProfile.getFunction().getID() == highLightID)) {
            g2D.setColor(ppTrial.getColorChooser().getHighlightColor());
        } else if (!userEventWindow && (ppFunctionProfile.isGroupMember(ppTrial.getHighlightedGroup()))) {
            g2D.setColor(ppTrial.getColorChooser().getGroupHighlightColor());
        } else {
            g2D.setColor(Color.black);
        }
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {

        list = window.getData();

        //With group support present, it is possible that the number of functions in
        //our data list is zero. If so, just return.
        if ((list.size()) == 0)
            return;

        int yCoord = 0;

        //In this window, a Monospaced font has to be used. This will probably
        // not be the same font as the rest of ParaProf. As a result, some extra work will
        // have to be done to calculate spacing.
        int fontSize = ppTrial.getPreferencesWindow().getFontSize();
        monoFont = new Font("Monospaced", ppTrial.getPreferencesWindow().getFontStyle(), fontSize);
        fmMonoFont = g2D.getFontMetrics(monoFont);
        maxFontAscent = fmMonoFont.getMaxAscent();
        maxFontDescent = fmMonoFont.getMaxDescent();
        spacing = fmMonoFont.getHeight();
        g2D.setFont(monoFont);
        FontRenderContext frc = g2D.getFontRenderContext();

        if (spacing <= (maxFontAscent + maxFontDescent)) {
            spacing = spacing + 1;
        }

        searcher.setLineHeight(spacing);

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
                //float y = insets.right;
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

        String headerString;
        if (userEventWindow) {
            headerString = StatWindowPanel.getUserEventStatStringHeading();

        } else {
            if (ppTrial.isTimeMetric())
                headerString = PPFunctionProfile.getStatStringHeading("Time");
            else
                headerString = PPFunctionProfile.getStatStringHeading("Counts");
        }

        //Calculate the name position.
        int namePosition = fmMonoFont.stringWidth(headerString) + xOffset;

        //Now append "name" to the end of the string.
        headerString = headerString + "Name";

        String dashString = "";
        for (int i = 0; i < headerString.length(); i++) {
            dashString = dashString + "-";
        }

        charWidth = fmMonoFont.stringWidth("A");

        setSearchLines(headerString, dashString);

        searcher.setG2d(g2D);
        searcher.setXOffset(xOffset);

        g2D.setColor(Color.black);

        //Draw the first dashed string.
        yCoord = yCoord + spacing;
        searcher.drawHighlights(g2D, xOffset, yCoord, 0);
        g2D.setColor(Color.black);
        g2D.drawString(dashString, xOffset, yCoord);
        yCoord = yCoord + spacing;

        //Draw the heading.
        searcher.drawHighlights(g2D, xOffset, yCoord, 1);
        g2D.setColor(Color.black);
        g2D.drawString(headerString, xOffset, yCoord);
        yCoord = yCoord + spacing;

        //Draw the second dashed string.
        searcher.drawHighlights(g2D, xOffset, yCoord, 2);
        g2D.setColor(Color.black);
        g2D.drawString(dashString, xOffset, yCoord);

        if (toScreen)
            startLocation = yCoord;

        //Set up some panel dimensions.
        newYPanelSize = yCoord + ((list.size() + 1) * spacing);

        //      determine which elements to draw (clipping)
        int[] clips = ParaProfUtils.computeClipping(g2D.getClipBounds(), window.getViewRect(), toScreen, fullWindow,
                list.size(), spacing, yCoord);
        int startElement = clips[0];
        int endElement = clips[1];
        yCoord = clips[2];

        searcher.setVisibleLines(startElement, endElement);

        for (int i = startElement; i <= endElement; i++) {
            String statString;
            PPFunctionProfile ppFunctionProfile = null;
            PPUserEventProfile ppUserEventProfile = null;

            if (userEventWindow) {
                ppUserEventProfile = (PPUserEventProfile) list.get(i);
                statString = ppUserEventProfile.getUserEventStatString(ParaProf.defaultNumberPrecision);
            } else {
                ppFunctionProfile = (PPFunctionProfile) list.get(i);
                statString = ppFunctionProfile.getStatString(window.units());
            }

            yCoord = yCoord + spacing;

            String nameString;
            Color nameColor;
            if (userEventWindow) {
                nameColor = ppUserEventProfile.getColor();
                nameString = ppUserEventProfile.getUserEventName();
            } else {
                nameColor = ppFunctionProfile.getColor();
                nameString = ppFunctionProfile.getDisplayName();
                if (window.getPhase() != null) {
                    nameString = UtilFncs.getRightSide(nameString);
                }
            }
            
            
            searcher.drawHighlights(g2D, xOffset, yCoord, i + 3);

            setStatStringColor(g2D, ppUserEventProfile, ppFunctionProfile);
            (new TextLayout(statString, monoFont, frc)).draw(g2D, xOffset, yCoord);

            g2D.setColor(nameColor);
            (new TextLayout(nameString, monoFont, frc)).draw(g2D, namePosition, yCoord);
        }
        //Resize the panel if needed.
        if ((newYPanelSize != yPanelSize) || (maxLinePixelWidth >= xPanelSize)) {
            yPanelSize = newYPanelSize + 1;
            xPanelSize = maxLinePixelWidth + 5;
            revalidate();
        }
    }

  

    public void mouseClicked(MouseEvent evt) {
        try {
            int yCoord = evt.getY();

            int tmpInt1 = yCoord - startLocation;
            int tmpInt2 = tmpInt1 / spacing;
            int tmpInt3 = (tmpInt2 + 1) * spacing;
            int tmpInt4 = tmpInt3 - maxFontAscent;

            if ((tmpInt1 >= tmpInt4) && (tmpInt1 <= tmpInt3)) {
                if (tmpInt2 < (list.size())) {
                    if (userEventWindow) {
                        PPUserEventProfile ppUserEventProfile = null;

                        ppUserEventProfile = (PPUserEventProfile) list.get(tmpInt2);
                        if (ParaProfUtils.rightClick(evt)) {
                            ParaProfUtils.handleUserEventClick(ppTrial, ppUserEventProfile.getUserEvent(), this, evt);
                        } else {
                            ppTrial.toggleHighlightedUserEvent(ppUserEventProfile.getUserEvent());
                        }

                    } else {
                        PPFunctionProfile ppFunctionProfile = null;
                        ppFunctionProfile = (PPFunctionProfile) list.get(tmpInt2);
                        if (ParaProfUtils.rightClick(evt)) {
                            
                            (ParaProfUtils.createFunctionClickPopUp(ppTrial, ppFunctionProfile.getFunction(), ppFunctionProfile.getThread(), this)).show(this, evt.getX(),
                                    evt.getY());

                        } else {
                            ppTrial.toggleHighlightedFunction(ppFunctionProfile.getFunction());
                        }

                    }
                }
            }
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    public void resetStringSize() {
        maxLinePixelWidth = 0;
        searcher.setSearchLines(null);
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

    public Searcher getSearcher() {
        return searcher;
    }

}