package edu.uoregon.tau.paraprof.barchart;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.util.ArrayList;

import javax.swing.JPanel;

import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.paraprof.Searcher;
import edu.uoregon.tau.paraprof.interfaces.ImageExport;

/**
 * Component for drawing all of the barcharts of ParaProf.
 * Clients should probably use BarChartPanel instead of BarChart
 * directly.
 * 
 * <P>CVS $Id: BarChart.java,v 1.1 2005/09/26 21:12:12 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 * @see BarChartPanel
 */
public class BarChart extends JPanel implements MouseListener, BarChartModelListener {

    private BarChartModel model;

    // internal variables for lazy evaluation
    private int maxRowLabelStringWidth;
    private boolean maxRowLabelStringWidthSet;
    private int maxValueLabelStringWidth;
    private boolean maxValueLabelStringWidthSet;
    private boolean preferredSizeSet;
    private boolean dataProcessed;

    // maximum values
    double maxRowValues[];
    double maxSubValues[];
    double maxRowSum;
    double maxOverallValue;
    double rowSums[];

    private FontMetrics fontMetrics;

    // row labels on the left or right
    private boolean leftJustified;

    private int barLength = 400;
    private int leftMargin = 8;
    private int rightMargin = 5;
    private int horizSpacing = 10;
    private int barVerticalSpacing = 4;
    private int barHeight;

    private int topMargin = 0;

    private int rowStart;

    // list of row label draw objects (only the ones on the screen)
    private ArrayList rowLabelDrawObjects = new ArrayList();
    // list of value label draw objects (only the ones on the screen)
    private ArrayList valueDrawObjects = new ArrayList();

    private BarChartPanel panel;
    private Searcher searcher;

    // Multi-graph only stuff

    // horizontal spacing between bars in multi-graph mode
    private int barHorizSpacing = 5;

    // threshold before not drawing a bar (add it to "other" instead)
    private int threshold = 2;

    // stacked or not
    private boolean stacked = true;

    // normalized or not
    private boolean normalized = true;

    private boolean singleLine = true;

    private int fontHeight;

    public BarChart(BarChartModel model, BarChartPanel panel) {
        this.model = model;
        this.panel = panel;
        model.addBarChartModelListener(this);

        setBackground(Color.white);
        addMouseListener(this);

        setDoubleBuffered(true);
        setOpaque(true);

        searcher = new Searcher(this, panel);
        searcher.setTopMargin(topMargin);
        addMouseListener(searcher);
        addMouseMotionListener(searcher);
        setAutoscrolls(true);

        barChartChanged();

        this.setToolTipText("...");

    }

    public boolean getLeftJustified() {
        return leftJustified;
    }

    public void setLeftJustified(boolean leftJustified) {
        this.leftJustified = leftJustified;
    }

    public String getToolTipText(MouseEvent e) {
        int x = e.getX();
        int y = e.getY();

        // search the row labels
        int size = rowLabelDrawObjects.size();
        for (int i = 0; i < size; i++) {
            DrawObject drawObject = (DrawObject) rowLabelDrawObjects.get(i);
            if (x >= drawObject.getXBeg() && x <= drawObject.getXEnd() && y >= drawObject.getYBeg() && y <= drawObject.getYEnd()) {
                return model.getRowLabelToolTipText(i + rowStart);
            }
        }

        // search the values
        size = valueDrawObjects.size();
        for (int i = 0; i < size; i++) {
            ArrayList subList = (ArrayList) valueDrawObjects.get(i);

            int size2 = subList.size();
            for (int j = 0; j < size2; j++) {
                DrawObject drawObject = (DrawObject) subList.get(j);
                if (drawObject != null) {
                    if (x >= drawObject.getXBeg() && x <= drawObject.getXEnd() && y >= drawObject.getYBeg()
                            && y <= drawObject.getYEnd()) {
                        if (j == size2 - 1) { // "other"
                            return model.getOtherToolTopText(i + rowStart);
                        } else {
                            return model.getValueToolTipText(i + rowStart, j);
                        }
                    }
                }

            }
        }
        return null;
    }

    protected void paintComponent(Graphics g) {
        try {
            Rectangle rect = g.getClipBounds();
            //setBackground(Color.white);
            //g.clearRect(rect.x, rect.y, rect.width, rect.height);
            //super.paintComponent(g);

            // Java 1.3 seems to need this rather than clearRect
            g.setColor(Color.white);
            g.fillRect(rect.x, rect.y, rect.width, rect.height);

            export((Graphics2D) g, true, false);
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
            //            window.closeThisWindow();
        }
    }

    public void mouseClicked(MouseEvent e) {
        int x = e.getX();
        int y = e.getY();

        // search the row labels
        int size = rowLabelDrawObjects.size();
        for (int i = 0; i < size; i++) {
            DrawObject drawObject = (DrawObject) rowLabelDrawObjects.get(i);
            if (x >= drawObject.getXBeg() && x <= drawObject.getXEnd() && y >= drawObject.getYBeg() && y <= drawObject.getYEnd()) {
                model.fireRowLabelClick(i + rowStart, e, this);
                return;
            }
        }

        // search the values
        size = valueDrawObjects.size();
        for (int i = 0; i < size; i++) {
            ArrayList subList = (ArrayList) valueDrawObjects.get(i);

            int size2 = subList.size();
            for (int j = 0; j < size2; j++) {
                DrawObject drawObject = (DrawObject) subList.get(j);
                if (drawObject != null) {
                    if (x >= drawObject.getXBeg() && x <= drawObject.getXEnd() && y >= drawObject.getYBeg()
                            && y <= drawObject.getYEnd()) {
                        if (j == size2 - 1) { // "other"
                            // we don't support clicking on "other" yet
                        } else {
                            model.fireValueClick(i + rowStart, j, e, this);
                        }
                    }
                }

            }
        }

    }

    public void mouseEntered(MouseEvent e) {
    }

    public void mouseExited(MouseEvent e) {
    }

    public void mousePressed(MouseEvent e) {
    }

    public void mouseReleased(MouseEvent e) {
    }

    private int getMaxRowLabelStringWidth() {
        if (!maxRowLabelStringWidthSet) {
            maxRowLabelStringWidth = 0;
            for (int i = 0; i < model.getNumRows(); i++) {
                String rowLabel = model.getRowLabel(i);
                maxRowLabelStringWidth = Math.max(maxRowLabelStringWidth, fontMetrics.stringWidth(rowLabel));
            }
            maxRowLabelStringWidthSet = true;
        }

        return maxRowLabelStringWidth;
    }

    private int getMaxValueLabelStringWidth() {
        if (!maxValueLabelStringWidthSet) {
            maxValueLabelStringWidth = 0;
            for (int i = 0; i < model.getNumRows(); i++) {
                String valueLabel = model.getValueLabel(i, 0);
                maxValueLabelStringWidth = Math.max(maxValueLabelStringWidth, fontMetrics.stringWidth(valueLabel));
            }
            maxValueLabelStringWidthSet = true;
        }

        return maxValueLabelStringWidth;
    }

    private void checkPreferredSize() {

        if (preferredSizeSet) {
            return;
        }
        int maxWidth, maxHeight;

        maxHeight = model.getNumRows() * fontHeight + topMargin + fontMetrics.getMaxDescent() + fontMetrics.getLeading();

        if (model.getSubSize() == 1) {
            maxWidth = barLength + getMaxRowLabelStringWidth() + getMaxValueLabelStringWidth() + leftMargin + (2 * horizSpacing)
                    + rightMargin;
        } else {
            if (singleLine) {
                if (stacked) {
                    maxWidth = leftMargin + getMaxRowLabelStringWidth() + horizSpacing + barLength + rightMargin;
                } else {
                    maxWidth = leftMargin + getMaxRowLabelStringWidth() + horizSpacing + rightMargin;
                    for (int i = 0; i < model.getSubSize(); i++) {
                        int subIndexMaxWidth = (int) (maxSubValues[i] / maxRowSum * barLength);
                        //if (subIndexMaxWidth >= threshold) {
                        maxWidth += (maxSubValues[i] / maxRowSum * barLength) + barHorizSpacing;
                        //}
                    }
                }
            } else {

                int rowHeight = (fontHeight * model.getSubSize()) + 10;

                maxWidth = barLength + getMaxRowLabelStringWidth() + getMaxValueLabelStringWidth() + leftMargin
                        + (2 * horizSpacing) + rightMargin;
                maxHeight = model.getNumRows() * rowHeight;
            }
        }
        super.setSize(new Dimension(maxWidth, maxHeight));
        super.setPreferredSize(new Dimension(maxWidth, maxHeight));
        preferredSizeSet = true;
        this.invalidate();
    }

    private void processData() {
        if (dataProcessed) {
            return;
        }

        dataProcessed = true;
        maxRowValues = new double[model.getNumRows()];
        maxSubValues = new double[model.getSubSize()];
        rowSums = new double[model.getNumRows()];
        maxRowSum = 0;
        maxOverallValue = 0;

        for (int row = 0; row < model.getNumRows(); row++) {
            double rowSum = 0;
            for (int i = 0; i < model.getSubSize(); i++) {
                double value = Math.max(0, model.getValue(row, i));
                maxRowValues[row] = Math.max(maxRowValues[row], value);
                maxSubValues[i] = Math.max(maxSubValues[i], value);
                rowSum += value;
                rowSums[row] += value;
                maxOverallValue = Math.max(maxOverallValue, value);
            }
            maxRowSum = Math.max(maxRowSum, rowSum);
        }
    }

    // returns a "lighter" color
    private Color lighter(Color c) {
        int r = c.getRed();
        int g = c.getGreen();
        int b = c.getBlue();

        int max = Math.max(r, g);
        max = Math.max(max, b);
        max = Math.max(max, 255);

        r = r + (int) ((max - r) / 2.36);
        g = g + (int) ((max - g) / 2.36);
        b = b + (int) ((max - b) / 2.36);
        return new Color(r, g, b);
    }

    private void drawBar(Graphics2D g2D, int x, int y, int length, int height, Color color, Color highlight) {

//         if (length > 5000) {
//             System.out.println("length = " + length);
//         }

        // special is whether or not we do the new style bars with the highlight along the top
        boolean special = true;
        //special = tr;

        if (special && height > 4) {
            g2D.setColor(color);

            g2D.fillRect(x, y, length, height-1);

            g2D.setColor(lighter(color));

            int innerHeight = height / 4;
            g2D.fillRect(x, y + (innerHeight / 2) + 1, length, innerHeight);

            int innerHeight2 = innerHeight / 3;
            g2D.setColor(lighter(lighter(color)));
            g2D.fillRect(x, y + (innerHeight / 2) + 1 + innerHeight2, length, innerHeight2);

            g2D.setColor(Color.black);
            if (highlight != null) {
                g2D.setColor(highlight);
                g2D.drawRect(x + 1, y + 1, length - 2, height - 3);
            }
            g2D.drawRect(x, y, length, height - 1);

        } else {
            height = Math.max(height, 1);
            g2D.setColor(color);
            g2D.fillRect(x, y, length, height);

            if (height > 3) {
                g2D.setColor(Color.black);
                if (highlight != null) {
                    g2D.setColor(highlight);
                    g2D.drawRect(x + 1, y + 1, length - 2, height - 2);
                }
                g2D.drawRect(x, y, length, height);
            }
        }
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow) {

        rowLabelDrawObjects.clear();
        valueDrawObjects.clear();

        Font font = ParaProf.preferencesWindow.getFont();
        g2D.setFont(font);
        fontMetrics = g2D.getFontMetrics(font);

        barVerticalSpacing = 0;

        fontHeight = fontMetrics.getHeight();
        int maxDescent = fontMetrics.getMaxDescent();
        int maxAscent = fontMetrics.getMaxAscent();
        int leading = fontMetrics.getLeading();
        barHeight = maxDescent + maxAscent + leading - 2;

        //        System.out.println("\n");
        //        System.out.println("getHeight = " + fontMetrics.getHeight());
        //        System.out.println("getAscent = " + fontMetrics.getAscent());
        //        System.out.println("getDescent = " + fontMetrics.getDescent());
        //        System.out.println("getMaxAscent = " + fontMetrics.getMaxAscent());
        //        System.out.println("getMaxDescent = " + fontMetrics.getMaxDescent());
        //        System.out.println("getLeading = " + fontMetrics.getLeading());

        processData();
        int fulcrum;

        if (leftJustified) {
            fulcrum = leftMargin + getMaxRowLabelStringWidth();
        } else {
            fulcrum = leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLength + horizSpacing;
        }

        //        System.out.println("leftMargin=" + leftMargin);
        //        System.out.println("getMaxValueLabelStringWidth() = " + getMaxValueLabelStringWidth());
        //        System.out.println("barlength = " + barLength);
        //        System.out.println("fulcrum = " + fulcrum);

        // we want the bar to be in the middle of the text
        int barOffset = (int) (maxAscent - (((float) maxDescent + maxAscent + leading) - barHeight) / 2);

        checkPreferredSize();

        int rowHeight = fontHeight;

        //System.out.println("rowHeight = " + rowHeight);

        int y = rowHeight + topMargin;

        if (singleLine == false) {
            rowHeight = (rowHeight * model.getSubSize()) + 10;
        }

        searcher.setLineHeight(rowHeight);
        searcher.setMaxDescent(fontMetrics.getMaxDescent());

        // this could be made faster, but the DrawObjects thing would have to change
        // the problem is that if I just redraw the clipRect, then there are objects on the screen
        // that weren't in the last draw, so we would have to keep track of them some other way
        int[] clips = ParaProfUtils.computeClipping(panel.getViewport().getViewRect(), panel.getViewport().getViewRect(), true,
                fullWindow, model.getNumRows(), rowHeight, y);
        rowStart = clips[0];
        int rowEnd = clips[1];
        y = clips[2];

        double maxValue = maxRowSum;

        searcher.setVisibleLines(rowStart, rowEnd);
        searcher.setG2d(g2D);
        searcher.setXOffset(fulcrum);

        for (int row = rowStart; row <= rowEnd; row++) {
            String rowLabel = model.getRowLabel(row);
            int rowLabelStringWidth = fontMetrics.stringWidth(rowLabel);

            ArrayList subDrawObjects = new ArrayList();
            valueDrawObjects.add(subDrawObjects);
            if (model.getSubSize() == 1) { // single graph style

                String valueLabel = model.getValueLabel(row, 0);

                double value = model.getValue(row, 0);

                double ratio = (value / maxValue);
                int length = (int) (ratio * barLength);

                int valueLabelStringWidth = fontMetrics.stringWidth(valueLabel);

                int rowLabelPosition;
                int valueLabelPosition;
                int barStartX;
                int barStartY;

                barStartY = y - barOffset;
                if (leftJustified) {
                    barStartX = fulcrum + horizSpacing;

                    rowLabelPosition = fulcrum - rowLabelStringWidth;
                    valueLabelPosition = fulcrum + length + (2 * horizSpacing);
                } else {
                    barStartX = fulcrum - length - horizSpacing;
                    rowLabelPosition = fulcrum;
                    valueLabelPosition = fulcrum - length - valueLabelStringWidth - (2 * horizSpacing);
                }

                drawBar(g2D, barStartX, y - barOffset, length, barHeight, model.getValueColor(row, 0),
                        model.getValueHighlightColor(row, 0));
                subDrawObjects.add(new DrawObject(barStartX, y - barOffset, barStartX + length, y - barOffset + barHeight));

                searcher.drawHighlights(g2D, rowLabelPosition, y, row);

                g2D.setColor(Color.black);
                g2D.drawString(rowLabel, rowLabelPosition, y);
                rowLabelDrawObjects.add(new DrawObject(rowLabelPosition, y - fontHeight, rowLabelPosition + rowLabelStringWidth,
                        y));

                g2D.drawString(valueLabel, valueLabelPosition, y);

                y = y + rowHeight;
                subDrawObjects.add(null); // "other"

            } else { // multi-graph

                if (singleLine) {
                    int barStartX;
                    int barStartY;
                    int rowLabelPosition;
                    barStartY = y - barOffset;

                    if (normalized) {
                        maxValue = rowSums[row];
                    } else {
                        maxValue = maxRowSum;
                    }

                    if (leftJustified) {
                        barStartX = fulcrum + horizSpacing;
                        rowLabelPosition = fulcrum - rowLabelStringWidth;
                    } else {
                        barStartX = 0;
                        rowLabelPosition = 0;
                    }

                    searcher.drawHighlights(g2D, rowLabelPosition, y, row);
                    g2D.setColor(Color.black);
                    g2D.drawString(rowLabel, rowLabelPosition, y);
                    rowLabelDrawObjects.add(new DrawObject(rowLabelPosition, y - fontHeight, rowLabelPosition
                            + rowLabelStringWidth, y));

                    double otherValue = 0;

                    if (maxValue > 0) {
                        for (int i = 0; i < model.getSubSize(); i++) {
                            g2D.setColor(model.getValueColor(row, i));
                            double value = model.getValue(row, i);
                            Color color = model.getValueColor(row, i);

                            double ratio = (value / maxValue);
                            int length = (int) (ratio * barLength);

                            if (length < threshold && stacked) {
                                otherValue += value;
                                subDrawObjects.add(null);
                            } else {

                                int subIndexMaxWidth = (int) (maxSubValues[i] / maxValue * barLength);

                                if (subIndexMaxWidth < threshold) {
                                    // this column will be skipped by all rows since no one's has at least 3 pixels
                                    subDrawObjects.add(null);
                                    otherValue += value;

                                } else {
                                    if (value < 0) { // negative means no value
                                        subDrawObjects.add(null);

                                    } else if (length < threshold) {
                                        subDrawObjects.add(null);
                                        otherValue += value;
                                    } else {
                                        drawBar(g2D, barStartX, y - barOffset, length, barHeight, color,
                                                model.getValueHighlightColor(row, i));

                                        subDrawObjects.add(new DrawObject(barStartX, y - barOffset, barStartX + length, y
                                                - barOffset + barHeight));
                                    }

                                    if (!stacked) {
                                        barStartX += (maxSubValues[i] / maxValue * barLength) + barHorizSpacing;
                                    } else {
                                        barStartX += length;
                                    }

                                }
                            }
                        }

                        int otherLength;
                        if (normalized) {
                            otherLength = barLength + fulcrum + horizSpacing - barStartX;

                        } else {
                            // draw "other" (should make this optional)
                            double ratio = (otherValue / maxValue);
                            otherLength = (int) (ratio * barLength);
                        }

                        drawBar(g2D, barStartX, y - barOffset, otherLength, barHeight, Color.black, null);
                        subDrawObjects.add(new DrawObject(barStartX, y - barOffset, barStartX + otherLength, y - barOffset
                                + barHeight));
                    }
                    y = y + rowHeight;
                } else {
                    int barStartX;
                    int rowLabelPosition;

                    maxValue = maxOverallValue;

                    leftJustified = false;
                    if (leftJustified) {
                        rowLabelPosition = fulcrum - rowLabelStringWidth;
                    } else {
                        rowLabelPosition = fulcrum;
                    }

                    int rowLabelPositionY = y + rowHeight / 2 - maxAscent;
                    searcher.drawHighlights(g2D, rowLabelPosition, rowLabelPositionY, row);
                    g2D.setColor(Color.black);
                    g2D.drawString(rowLabel, rowLabelPosition, rowLabelPositionY);
                    rowLabelDrawObjects.add(new DrawObject(rowLabelPosition, rowLabelPositionY - fontHeight, rowLabelPosition
                            + rowLabelStringWidth, rowLabelPositionY));

                    int suby = y;
                    for (int i = 0; i < model.getSubSize(); i++) {
                        double value = model.getValue(row, i);
                        if (value < 0) {
                            suby += fontHeight;
                            continue;
                        }
                        String valueLabel = model.getValueLabel(row, i);
                        int valueLabelStringWidth = fontMetrics.stringWidth(valueLabel);

                        double ratio = (value / maxValue);
                        int length = (int) (ratio * barLength);
                        int valueLabelPosition;

                        if (leftJustified) {
                            barStartX = fulcrum + horizSpacing;
                            valueLabelPosition = fulcrum + length + (2 * horizSpacing);

                        } else {
                            barStartX = fulcrum - length - horizSpacing;
                            valueLabelPosition = fulcrum - length - valueLabelStringWidth - (2 * horizSpacing);

                        }
                        drawBar(g2D, barStartX, suby - barOffset, length, barHeight, model.getValueColor(row, i),
                                model.getValueHighlightColor(row, 0));

                        g2D.drawString(valueLabel, valueLabelPosition, suby);

                        suby += fontHeight;
                    }

                    y = y + rowHeight;

                }
            }

        }

    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        // TODO Auto-generated method stub
        return null;
    }

    public void barChartChanged() {
        preferredSizeSet = false;
        maxValueLabelStringWidthSet = false;
        maxRowLabelStringWidthSet = false;
        dataProcessed = false;
        setSearchLines();
        this.repaint();
    }

    private void setSearchLines() {
        ArrayList searchLines = new ArrayList();
        for (int i = 0; i < model.getNumRows(); i++) {
            searchLines.add(model.getRowLabel(i));
        }
        searcher.setSearchLines(searchLines);
    }

    public int getBarLength() {
        return barLength;
    }

    public void setBarLength(int barLength) {
        this.barLength = barLength;
        this.preferredSizeSet = false;
    }

    public Searcher getSearcher() {
        return searcher;
    }

    public boolean getNormalized() {
        return normalized;
    }

    public void setNormalized(boolean normalized) {
        this.normalized = normalized;
        this.preferredSizeSet = false;
    }

    public boolean getStacked() {
        return stacked;
    }

    public void setStacked(boolean stacked) {
        this.stacked = stacked;
        this.preferredSizeSet = false;
    }

    public boolean getSingleLine() {
        return singleLine;
    }

    public void setSingleLine(boolean singleLine) {
        this.singleLine = singleLine;
    }

}
