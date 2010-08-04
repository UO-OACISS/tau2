package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.ClipboardOwner;
import java.awt.datatransfer.Transferable;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.font.TextHitInfo;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JComponent;

import edu.uoregon.tau.paraprof.interfaces.ScrollBarController;
import edu.uoregon.tau.paraprof.interfaces.Searchable;

/**
 * Searches text for ParaProf windows
 *    
 * TODO : ...
 *
 * <P>CVS $Id: Searcher.java,v 1.6 2007/01/04 01:55:32 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.6 $
 */
public class Searcher implements Searchable, MouseListener, MouseMotionListener, ClipboardOwner {

    private List<String> searchLines = new ArrayList<String>();
    private String searchString = "";
    private int searchLine;
    private int searchColumn;
    private boolean searchHighlight;
    private boolean searchMatchCase;
    private boolean searchUp;

    private int lineHeight = 1; // initialize to 1 to eliminate the possiblility of divide by zero
    private int maxDescent;

    private JComponent panel;
    private ScrollBarController scrollBarController;

    private int firstVisibleLine;
    private int lastVisibleLine;

    private Graphics2D g2d;
    private int xOffset;
    private int topMargin;

    // for selection
    private String selectionString = "";
    private boolean selectionReversed;
    private int selectionStartLine = -1, selectionEndLine = -1;
    private int selectionStartX, selectionStartY;
    private int selectionEndX, selectionEndY;

    public Searcher(JComponent panel, ScrollBarController scrollBarController) {
        this.panel = panel;
        this.scrollBarController = scrollBarController;
    }

    public void setVisibleLines(int first, int last) {
        this.firstVisibleLine = first;
        this.lastVisibleLine = last;
    }

    public void setSearchHighlight(boolean highlight) {
        searchHighlight = highlight;
        panel.repaint();
    }

    public void setSearchMatchCase(boolean matchCase) {
        searchMatchCase = matchCase;
        panel.repaint();
    }

    public boolean searchNext() {
        searchColumn++;
        setSearchString(searchString);
        return false;
    }

    public boolean searchPrevious() {
        searchUp = true;
        setSearchString(searchString);
        searchUp = false;
        return false;
    }

    public void setSearchLines(List<String> searchLines) {
        this.searchLines = searchLines;
    }

    public List<String> getSearchLines() {
        return searchLines;
    }

    public boolean setSearchString(String searchString) {
        this.searchString = searchString;

        if (searchLines == null)
            return false;

        if (searchString.length() == 0) { // reset
            searchLine = 0;
            searchColumn = 0;
            panel.repaint();
            return true;
        }

        if (!searchMatchCase) { // not matching case, so convert everything to uppercase for comparison
            searchString = searchString.toUpperCase();
        }

        boolean found = false;

        if (searchUp) {

            for (int i = searchLine; i >= 0; i--) {
                String line = searchLines.get(i);

                if (!searchMatchCase) {
                    line = line.toUpperCase();
                }

                if (i != searchLine) {
                    searchColumn = line.length();
                } else {
                    searchColumn--;
                }
                if (line.lastIndexOf(searchString, searchColumn) != -1) {
                    searchLine = i;
                    searchColumn = line.lastIndexOf(searchString, searchColumn);
                    found = true;
                    break;
                }
            }

            if (!found) { // wrap
                for (int i = searchLines.size() - 1; i >= 0; i--) {
                    String line = searchLines.get(i);

                    if (!searchMatchCase) {
                        line = line.toUpperCase();
                    }

                    if (i != searchLine) {
                        searchColumn = line.length();
                    } else {
                        searchColumn--;
                    }
                    if (line.lastIndexOf(searchString, searchColumn) != -1) {
                        searchLine = i;
                        searchColumn = line.lastIndexOf(searchString, searchColumn);
                        found = true;
                        break;
                    }
                }
            }

        } else {
            for (int i = searchLine; i < searchLines.size(); i++) {
                String line = searchLines.get(i);

                if (!searchMatchCase) {
                    line = line.toUpperCase();
                }

                if (line.indexOf(searchString, searchColumn) != -1) {
                    searchLine = i;
                    searchColumn = line.indexOf(searchString, searchColumn);
                    found = true;
                    break;
                }

                searchColumn = 0;

            }

            if (!found) { // wrap
                for (int i = 0; i < searchLines.size(); i++) {
                    String line = searchLines.get(i);

                    if (!searchMatchCase) {
                        line = line.toUpperCase();
                    }

                    if (i == searchLine) {
                        searchColumn++;
                    } else {
                        searchColumn = 0;
                    }

                    if (line.indexOf(searchString, searchColumn) != -1) {
                        searchLine = i;
                        searchColumn = line.indexOf(searchString, searchColumn);
                        found = true;
                        break;
                    }
                }
            }
        }

        if (!found) {
            panel.repaint();
            return false;
        }

        checkSearchStringVisibility();

        panel.repaint();
        return true;
    }

    private void checkSearchStringVisibility() {
        if (g2d == null) {
            return;
        }
        String localSearchString = searchString;

        // now determine if it is visible vertically
        if (searchLine <= firstVisibleLine || searchLine >= (lastVisibleLine - 1)) {
            Dimension dimension = scrollBarController.getThisViewportSize();
            scrollBarController.setVerticalScrollBarPosition((searchLine * lineHeight - ((int) dimension.getHeight() / 2)));
        }

        // now check the horizontal scrollbar

        String line = searchLines.get(searchLine);

        if (!searchMatchCase) {
            localSearchString = searchString.toUpperCase();
            line = line.toUpperCase();
        }

        TextLayout textLayout = new TextLayout(line, g2d.getFont(), g2d.getFontRenderContext());

        Shape base = textLayout.getLogicalHighlightShape(line.indexOf(localSearchString, searchColumn), line.indexOf(
                localSearchString, searchColumn)
                + localSearchString.length());

        AffineTransform at = AffineTransform.getTranslateInstance(xOffset, searchLine * lineHeight);
        Shape highlight = at.createTransformedShape(base);

        // move all the way left first
        scrollBarController.setHorizontalScrollBarPosition(0);

        // now make sure that the rectangle is visible
        panel.scrollRectToVisible(highlight.getBounds());

    }

    public void drawHighlights(Graphics2D g2D, int x, int y, int line) {
        String text = searchLines.get(line);
        String originalText = searchLines.get(line);

        if (text.length() < 1) {
            return;
        }
        TextLayout textLayout = new TextLayout(text, g2D.getFont(), g2D.getFontRenderContext());

        String localSearchString = searchString;

        if (!searchMatchCase) { // switch to uppercase if not doing matching case
            text = text.toUpperCase();
            localSearchString = searchString.toUpperCase();
        }

        // highlight all matches if highlighting is on
        if (searchHighlight && localSearchString.length() > 0) {
            int column = 0;
            while (text.indexOf(localSearchString, column) != -1) {
                column = text.indexOf(localSearchString, column);
                Shape base = textLayout.getLogicalHighlightShape(text.indexOf(localSearchString, column), text.indexOf(
                        localSearchString, column)
                        + localSearchString.length());
                AffineTransform at = AffineTransform.getTranslateInstance(x, y);
                Shape highlight = at.createTransformedShape(base);
                g2D.setPaint(Searchable.highlightColor);
                g2D.fill(highlight);
                column++;
            }

        }

        if (line == searchLine) {
            // the current incremental search line

            if (text.indexOf(localSearchString, searchColumn) != -1 && localSearchString.length() > 0) {
                Shape base = textLayout.getLogicalHighlightShape(text.indexOf(localSearchString, searchColumn), text.indexOf(
                        localSearchString, searchColumn)
                        + localSearchString.length());
                AffineTransform at = AffineTransform.getTranslateInstance(x, y);
                Shape highlight = at.createTransformedShape(base);
                g2D.setPaint(Searchable.searchColor);
                g2D.fill(highlight);
            }
        }

        // now do selection
        if (line >= selectionStartLine || line <= selectionEndLine) {
            // 4 cases

            if (selectionStartLine == selectionEndLine && line == selectionStartLine) {
                // selection only on one line
                int localStart = Math.min(selectionStartX, selectionEndX);
                int localEnd = Math.max(selectionStartX, selectionEndX);

                TextHitInfo startHit = textLayout.hitTestChar(localStart - x, selectionStartY);
                TextHitInfo endHit = textLayout.hitTestChar(localEnd - x, selectionEndY);

                int startChar = 0, endChar = textLayout.getCharacterCount();

                if (startHit != null) {
                    startChar = startHit.getInsertionIndex();
                }

                if (endHit != null) {
                    endChar = endHit.getInsertionIndex();
                }

                Shape base = textLayout.getLogicalHighlightShape(startChar, endChar);
                //Shape base = textLayout.getBlackBoxBounds(startChar, endChar);
                AffineTransform at = AffineTransform.getTranslateInstance(x, y);
                Shape highlight = at.createTransformedShape(base);
                g2D.setPaint(Searchable.selectionColor);

                Rectangle rec = highlight.getBounds();
                rec.grow(0, lineHeight - rec.height);

                g2D.fill(rec);

            } else {

                if (line == selectionStartLine) {

                    int localX = selectionStartX;
                    if (selectionReversed) {
                        localX = selectionEndX;
                    }

                    // the first line of a multi-line selection
                    TextHitInfo hit = textLayout.hitTestChar(localX - x, selectionStartY);
                    if (hit != null) {
                        Shape base = textLayout.getLogicalHighlightShape(hit.getInsertionIndex(), textLayout.getCharacterCount());
                        AffineTransform at = AffineTransform.getTranslateInstance(x, y);
                        Shape highlight = at.createTransformedShape(base);

                        Rectangle rec = highlight.getBounds();
                        rec.grow(0, lineHeight - rec.height);

                        g2D.setPaint(new Color(184, 207, 229));
                        g2D.fill(rec);

                    }

                }

                if (line > selectionStartLine && line < selectionEndLine) {
                    // a middle line of the selection
                    //Shape base = textLayout.getLogicalHighlightShape(0, textLayout.getCharacterCount());
                    Shape base = textLayout.getLogicalHighlightShape(0, textLayout.getCharacterCount());

                    AffineTransform at = AffineTransform.getTranslateInstance(x, y);
                    Shape highlight = at.createTransformedShape(base);

                    Rectangle rec = highlight.getBounds();
                    rec.grow(0, lineHeight - rec.height);

                    g2D.setPaint(new Color(184, 207, 229));
                    g2D.fill(rec);

                }

                if (line == selectionEndLine) {
                    // last line of selection

                    int localX = selectionEndX;
                    if (selectionReversed) {
                        localX = selectionStartX;
                    }

                    TextHitInfo hit = textLayout.hitTestChar(localX - x, selectionStartY);
                    if (hit != null) {
                        Shape base = textLayout.getLogicalHighlightShape(0, hit.getInsertionIndex());
                        AffineTransform at = AffineTransform.getTranslateInstance(x, y);
                        Shape highlight = at.createTransformedShape(base);

                        Rectangle rec = highlight.getBounds();
                        rec.grow(0, lineHeight - rec.height);

                        g2D.setPaint(new Color(184, 207, 229));
                        g2D.fill(rec);

                    }
                }
            }
        }
    }

    public String getSelectionString() {
        return selectionString;
    }

    public void mouseClicked(MouseEvent e) {
        // TODO Auto-generated method stub

    }

    public void mouseEntered(MouseEvent e) {
        // TODO Auto-generated method stub

    }

    public void mouseExited(MouseEvent e) {
        // TODO Auto-generated method stub

    }

    public void mousePressed(MouseEvent e) {
        selectionStartX = e.getX();
        selectionStartY = e.getY() - topMargin;
        selectionStartLine = (e.getY() - topMargin) / lineHeight;
    }

    private void determineSelection() {
        if (g2d == null) {
            return;
        }

        selectionString = "";

        for (int line = selectionStartLine; line <= selectionEndLine; line++) {
            if (line >= searchLines.size()) {
                break;
            }
            if (line < 0 || line >= searchLines.size()) {
                return;
            }

            String text = searchLines.get(line);

            TextLayout textLayout = new TextLayout(text, g2d.getFont(), g2d.getFontRenderContext());

            // 4 cases

            if (selectionStartLine == selectionEndLine && line == selectionStartLine) {
                // selection only on one line
                int localStart = Math.min(selectionStartX, selectionEndX);
                int localEnd = Math.max(selectionStartX, selectionEndX);

                TextHitInfo startHit = textLayout.hitTestChar(localStart - xOffset, selectionStartY);
                TextHitInfo endHit = textLayout.hitTestChar(localEnd - xOffset, selectionEndY);

                int startChar = 0, endChar = textLayout.getCharacterCount();

                if (startHit != null) {
                    startChar = startHit.getInsertionIndex();
                }

                if (endHit != null) {
                    endChar = endHit.getInsertionIndex();
                }

                selectionString = text.substring(startChar, endChar);

            } else {

                if (line == selectionStartLine) {

                    int localX = selectionStartX;
                    if (selectionReversed) {
                        localX = selectionEndX;
                    }

                    // the first line of a multi-line selection
                    TextHitInfo hit = textLayout.hitTestChar(localX - xOffset, selectionStartY);
                    if (hit != null) {
                        selectionString = selectionString + text.substring(hit.getInsertionIndex()) + "\n";
                    }

                }

                if (line > selectionStartLine && line < selectionEndLine) {
                    // a middle line of the selection
                    selectionString = selectionString + text + "\n";
                }

                if (line == selectionEndLine) {
                    // last line of selection

                    int localX = selectionEndX;
                    if (selectionReversed) {
                        localX = selectionStartX;
                    }

                    TextHitInfo hit = textLayout.hitTestChar(localX - xOffset, selectionStartY);
                    if (hit != null) {
                        selectionString = selectionString + text.substring(0, hit.getInsertionIndex());
                    }
                }
            }
        }

    }

    public void mouseReleased(MouseEvent e) {
        selectionEndX = e.getX();
        selectionEndY = e.getY() - topMargin;

        if (selectionStartY < selectionEndY) {
            selectionStartLine = selectionStartY / lineHeight;
            selectionEndLine = selectionEndY / lineHeight;
        } else {
            selectionEndLine = selectionStartY / lineHeight;
            selectionStartLine = selectionEndY / lineHeight;
        }

        determineSelection();

        // set the clipboard
        JVMDependent.setClipboardContents(selectionString, this);
        panel.repaint();

    }

    public void mouseDragged(MouseEvent e) {
        selectionEndX = e.getX();
        selectionEndY = (e.getY() - topMargin) - maxDescent;

        if (selectionStartY < selectionEndY) {
            selectionStartLine = selectionStartY / lineHeight;
            selectionEndLine = selectionEndY / lineHeight;
            selectionReversed = false;
        } else {
            selectionEndLine = selectionStartY / lineHeight;
            selectionStartLine = selectionEndY / lineHeight;
            selectionReversed = true;
        }

        // for auto-scrolling
        Rectangle r = new Rectangle(e.getX(), e.getY(), 1, 1);
        panel.scrollRectToVisible(r);
        panel.repaint();
    }

    public void mouseMoved(MouseEvent e) {
        // TODO Auto-generated method stub

    }

    public void lostOwnership(Clipboard clipboard, Transferable contents) {
        // reset the selection
        selectionStartY = -1;
        selectionEndY = -1;
        selectionStartLine = -1;
        selectionEndLine = -1;
        panel.repaint();
    }

    public void setXOffset(int offset) {
        this.xOffset = offset;
    }

    public void setG2d(Graphics2D g2d) {
        this.g2d = g2d;
    }

    public void setLineHeight(int lineHeight) {
        // disallow less than one to eliminate divide by zero errors
        this.lineHeight = Math.max(1, lineHeight);
    }

    public void setMaxDescent(int maxDescent) {
        this.maxDescent = maxDescent;
    }

    public void setTopMargin(int topMargin) {
        this.topMargin = topMargin;
    }

}
