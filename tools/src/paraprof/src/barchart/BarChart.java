package edu.uoregon.tau.paraprof.barchart;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.ArrayList;

import javax.swing.JPanel;

import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.paraprof.Searcher;

/**
 * Component for drawing all of the barcharts of ParaProf.
 * Clients should probably use BarChartPanel instead of BarChart
 * directly.
 * 
 * <P>CVS $Id: BarChart.java,v 1.11 2009/11/05 09:43:32 khuck Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.11 $
 * @see BarChartPanel
 */
public class BarChart extends JPanel implements MouseListener, MouseMotionListener, BarChartModelListener {

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
    double maxSubSum;

    private FontMetrics fontMetrics;

    // row labels on the left or right
    private boolean leftJustified;

    private int barLengthMultiple = 0;
    private int barLength = 0;
    private int leftMargin = 8;
    private int rightMargin = 12;
    private int horizSpacing = 10;
    private int additionalVerticalSpacing = 0;
    private int barHeight;

    private int topMargin = 0;

    // the range of rows currently shown (the ones not clipped)
    private int rowStart;
    private int rowEnd;

    // list of row label draw objects (only the ones on the screen)
    private ArrayList rowLabelDrawObjects = new ArrayList();
    // list of value label draw objects (only the ones on the screen)
    private ArrayList valueDrawObjects = new ArrayList();

    private BarChartPanel panel;
    private Searcher searcher;

    // Multi-graph only stuff

    // horizontal spacing between bars in multi-graph mode, when unstacked
    private int barHorizSpacing = 5;

    // threshold before not drawing a bar (add it to "other" instead)
    private int threshold = 2;

    // stacked or not
    private boolean stacked = true;

    // normalized or not
    private boolean normalized = true;

    // single line means that the bars go along one line
    // the comparison window is the only one not using singleLine
    private boolean singleLine = true;

    private int fontHeight;

    private int mouseCol = -1;

    private int rowHeight;

    //    private boolean autoResize = false;
    private boolean autoResize = true;
    private int autoWidth;
    private static final int minBarSize = 250;

    // mouseover highlighting
    private static final boolean mouseOverHighlighting = false;

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

        if (mouseOverHighlighting) {
            addMouseMotionListener(this);
        }
    }

    public void sizeChanged() {
        // when the size changes, we recompute the bar length
        if (!autoResize) {
            return;
        }

        if (fontMetrics == null) {
            return;
        }
        if (leftJustified) {
            int fulcrum = leftMargin + getMaxRowLabelStringWidth();
            setBarLength(getParent().getWidth() - (fulcrum + horizSpacing + rightMargin));
        } else {
            barLengthMultiple = minBarSize;
            int maxWidth = leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLengthMultiple + horizSpacing
                    + getMaxRowLabelStringWidth() + rightMargin;

            if (maxWidth < getParent().getWidth()) {
                barLengthMultiple = Math.max(barLengthMultiple, getParent().getWidth() - (maxWidth - barLengthMultiple));
            }
            setBarLength(barLengthMultiple);

            maxWidth = leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLengthMultiple + horizSpacing
                    + getMaxRowLabelStringWidth() + rightMargin;

            checkPreferredSize();
            this.revalidate();
        }
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
        if (fontMetrics == null) {
            return 0;
        }
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
                for (int j = 0; j < model.getSubSize(); j++) {
                    String valueLabel = model.getValueLabel(i, j);
                    maxValueLabelStringWidth = Math.max(maxValueLabelStringWidth, fontMetrics.stringWidth(valueLabel));
                }
            }
            maxValueLabelStringWidthSet = true;
        }

        return maxValueLabelStringWidth;
    }

    private void checkPreferredSize() {
        if (preferredSizeSet || fontMetrics == null) {
            return;
        }
        preferredSizeSet = true;
        int maxWidth, maxHeight;

        maxHeight = model.getNumRows() * fontHeight + topMargin + fontMetrics.getMaxDescent() + fontMetrics.getLeading();

        if (!singleLine) {
            int rowHeight = (fontHeight * model.getSubSize()) + 10;
            maxHeight = model.getNumRows() * rowHeight;
        }

        if (autoResize) {

            if (!leftJustified) {
                maxWidth = leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLengthMultiple + horizSpacing
                        + getMaxRowLabelStringWidth() + rightMargin;

                if (barLengthMultiple > minBarSize) {
                    maxWidth = 0;
                }
            } else {
                maxWidth = 0;
            }
            setPreferredSize(new Dimension(maxWidth, maxHeight + 5));
            // if I don't call setSize in addition to setPreferredSize, then when it initially draws, there are no scrollbars for some reason
            setSize(new Dimension(maxWidth, maxHeight + 5));
            invalidate();
            return;
        }

        // autoResize is forced on for now, below this is never executed

        if (model.getSubSize() == 1) {
            maxWidth = leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLengthMultiple + horizSpacing
                    + getMaxRowLabelStringWidth() + rightMargin;
        } else {
            if (singleLine) {
                if (stacked) {
                    maxWidth = leftMargin + getMaxRowLabelStringWidth() + horizSpacing + barLengthMultiple + rightMargin;
                } else {
                    maxWidth = leftMargin + getMaxRowLabelStringWidth() + horizSpacing + rightMargin;
                    double otherValues = 0;
                    for (int i = 0; i < model.getSubSize(); i++) {
                        int subIndexMaxWidth = (int) (maxSubValues[i] / maxRowSum * barLengthMultiple);
                        if (subIndexMaxWidth >= threshold) {
                            System.out.println("goo");
                            //maxWidth += (maxSubValues[i] / maxRowSum * barLengthMultiple) + barHorizSpacing;
                            maxWidth += (maxSubValues[i] / maxRowSum * barLengthMultiple);
                        }
                    }
                }
            } else {
                // multi-line (i.e. such as comparison windows)

                int rowHeight = (fontHeight * model.getSubSize()) + 10;

                maxWidth = barLengthMultiple + getMaxRowLabelStringWidth() + getMaxValueLabelStringWidth() + leftMargin
                        + (2 * horizSpacing) + rightMargin;
                maxHeight = model.getNumRows() * rowHeight;
            }
        }

        super.setPreferredSize(new Dimension(maxWidth, maxHeight + 5));
        super.setSize(new Dimension(maxWidth, maxHeight + 5));
        this.invalidate();
    }

    private void processData() {
        if (dataProcessed) {
            return;
        }
        //long time = System.currentTimeMillis();

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

        // maxSubSum is the sum of each of the maxSubValues
        // each maxSubValue is the maximum value for a column (timer)
        // this is only useful for the unstacked mode
        maxSubSum = 0;
        for (int i = 0; i < model.getSubSize(); i++) {
            double value = maxSubValues[i];
            maxSubSum += value;
        }

        //time = (System.currentTimeMillis()) - time;
        //System.out.println("Time for processData : " + time);
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

    private void drawBar(Graphics2D g2D, int x, int y, int length, int height, Color color, Color borderColor) {

        // special is whether or not we do the new style bars with the highlight along the top
        boolean special = true;

        if (special && height > 4) {
            g2D.setColor(color);

            g2D.fillRect(x, y, length, height - 1);

            
            g2D.setColor(lighter(color));

            int innerHeight = height / 4;
            g2D.fillRect(x, y + (innerHeight / 2) + 1, length, innerHeight);

            int innerHeight2 = innerHeight / 3;
            g2D.setColor(lighter(lighter(color)));
            g2D.fillRect(x, y + (innerHeight / 2) + 1 + innerHeight2, length, innerHeight2);

            g2D.setColor(Color.black);
            if (borderColor != null) {
                g2D.setColor(borderColor);
                g2D.drawRect(x + 1, y + 1, length - 2, height - 3);
            }
            g2D.drawRect(x, y, length, height - 1);

        } else {
            height = Math.max(height, 1);
            g2D.setColor(color);
            g2D.fillRect(x, y, length, height);

            if (height > 3) {
                g2D.setColor(Color.black);
                if (borderColor != null) {
                    g2D.setColor(borderColor);
                    g2D.drawRect(x + 1, y + 1, length - 2, height - 2);
                }
                g2D.drawRect(x, y, length, height);
            }
        }
    }

    private void drawSingleGraph(Graphics2D g2D, int startY, int fulcrum, int barOffset) {
        // "single" graph, like function BarChart, or thread BarChart

        int y = startY;
        for (int row = rowStart; row <= rowEnd; row++) {
            String rowLabel = model.getRowLabel(row);
            int rowLabelStringWidth = fontMetrics.stringWidth(rowLabel);

            ArrayList subDrawObjects = new ArrayList();
            valueDrawObjects.add(subDrawObjects);

            String valueLabel = model.getValueLabel(row, 0);

            double value = model.getValue(row, 0);
            double ratio = (value / maxRowSum);
            int length = (int) (ratio * barLengthMultiple);

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

            drawBar(g2D, barStartX, barStartY, length, barHeight, model.getValueColor(row, 0), model.getValueHighlightColor(row,
                    0));
            subDrawObjects.add(new DrawObject(barStartX, barStartY, barStartX + length, y - barOffset + barHeight));

            searcher.drawHighlights(g2D, rowLabelPosition, y, row);

            g2D.setColor(Color.black);
            g2D.drawString(rowLabel, rowLabelPosition, y);
            rowLabelDrawObjects.add(new DrawObject(rowLabelPosition, y - fontHeight, rowLabelPosition + rowLabelStringWidth, y));

            g2D.drawString(valueLabel, valueLabelPosition, y);

            y = y + rowHeight;
            subDrawObjects.add(null); // "other"
        }

    }

    private void drawMultiGraphHorizontal(Graphics2D g2D, int startY, int fulcrum, int barOffset) {
        int y = startY;
        for (int row = rowStart; row <= rowEnd; row++) {
            String rowLabel = model.getRowLabel(row);
            int rowLabelStringWidth = fontMetrics.stringWidth(rowLabel);

            ArrayList subDrawObjects = new ArrayList();
            valueDrawObjects.add(subDrawObjects);

            int barStartX;
            int barStartY;
            int rowLabelPosition;
            barStartY = y - barOffset;

            double maxValue;
            if (!stacked) {
                maxValue = maxSubSum;
            } else if (normalized) {
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
            rowLabelDrawObjects.add(new DrawObject(rowLabelPosition, y - fontHeight, rowLabelPosition + rowLabelStringWidth, y));

            if (maxValue > 0) {

                // the "otherValue" is the sum of values that are less than the threshold (e.g. 2 pixels)
                // we put them together in their own box at the end
                double otherValue = 0;

                // the "bonus" is the remainder of the conversion of the ratio to integer pixels
                // we add it to the next bar so that 1.1 + 1.9 is the same length as 3.0
                // it makes the bars more representative
                double bonus = 0;

                for (int i = 0; i < model.getSubSize(); i++) {
                    g2D.setColor(model.getValueColor(row, i));
                    double value = model.getValue(row, i);
                    Color color = model.getValueColor(row, i);

                    double ratio = (value / maxValue);
                    int length = (int) (ratio * barLengthMultiple + bonus);
                    if (stacked) {
                        bonus = (ratio * barLengthMultiple + bonus) - length;
                    }

                    if (length < threshold && stacked) {
                        otherValue += value;
                        subDrawObjects.add(null);
                    } else {

                        int subIndexMaxWidth = (int) ((maxSubValues[i] / maxValue * barLengthMultiple));

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

                                if (i == mouseCol) {
                                    color = lighter(color);
                                }

                                drawBar(g2D, barStartX, barStartY, length, barHeight, color, model.getValueHighlightColor(row, i));

                                subDrawObjects.add(new DrawObject(barStartX, barStartY, barStartX + length, barStartY + barHeight));
                            }

                            if (!stacked) { // when unstacked, we shift by the amount given by the max value for this subIndex (e.g. column)
                                barStartX += (maxSubValues[i] / maxValue * barLengthMultiple) + barHorizSpacing;
                            } else {
                                barStartX += length;
                            }

                        }
                    }
                }

                int otherLength;
                if (normalized) { // implies stacked as well
                    otherLength = barLengthMultiple + fulcrum + horizSpacing - barStartX;
                } else { // unstacked will always take this path
                    double ratio = (otherValue / maxValue);
                    if (stacked) { // only add the bonus length when stacked
                        otherLength = (int) (ratio * barLengthMultiple + bonus);
                    } else {
                        otherLength = (int) (ratio * barLengthMultiple);
                    }
                }

                // draw "other"
                drawBar(g2D, barStartX, barStartY, otherLength, barHeight, Color.black, null);
                subDrawObjects.add(new DrawObject(barStartX, barStartY, barStartX + otherLength, barStartY + barHeight));
            }
            y = y + rowHeight;
        }

    }

    private void drawMultiGraphVertical(Graphics2D g2D, int startY, int fulcrum, int barOffset, int maxAscent) {
        int y = startY;
        for (int row = rowStart; row <= rowEnd; row++) {
            String rowLabel = model.getRowLabel(row);
            int rowLabelStringWidth = fontMetrics.stringWidth(rowLabel);

            ArrayList subDrawObjects = new ArrayList();
            valueDrawObjects.add(subDrawObjects);
            int barStartX;
            int rowLabelPosition;

            double maxValue = maxOverallValue;

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

            int suby = y; // y value within the "row"
            for (int i = 0; i < model.getSubSize(); i++) {
                double value = model.getValue(row, i);
                if (value < 0) {
                    suby += fontHeight;
                    continue;
                }
                String valueLabel = model.getValueLabel(row, i);
                int valueLabelStringWidth = fontMetrics.stringWidth(valueLabel);

                double ratio = (value / maxValue);
                int length = (int) (ratio * barLengthMultiple);
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

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow) {

        Font font = ParaProf.preferencesWindow.getFont();
        g2D.setFont(font);
        fontMetrics = g2D.getFontMetrics(font);

        rowLabelDrawObjects.clear();
        valueDrawObjects.clear();

        fontHeight = fontMetrics.getHeight();
        int maxDescent = fontMetrics.getMaxDescent();
        int maxAscent = fontMetrics.getMaxAscent();
        int leading = fontMetrics.getLeading();
        barHeight = maxDescent + maxAscent + leading - 2;

        if (autoResize) {
            if (autoWidth != getParent().getWidth()) {
                sizeChanged();
            }
            autoWidth = getParent().getWidth();
        }

        

        //        System.out.println("\n");
        //        System.out.println("getHeight = " + fontMetrics.getHeight());
        //        System.out.println("getAscent = " + fontMetrics.getAscent());
        //        System.out.println("getDescent = " + fontMetrics.getDescent());
        //        System.out.println("getMaxAscent = " + fontMetrics.getMaxAscent());
        //        System.out.println("getMaxDescent = " + fontMetrics.getMaxDescent());
        //        System.out.println("getLeading = " + fontMetrics.getLeading());

        processData();
        
        if (barLengthMultiple == 0) {
            sizeChanged();
        }
        int fulcrum;

        if (leftJustified) {
            fulcrum = leftMargin + getMaxRowLabelStringWidth();
        } else {
            fulcrum = leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLengthMultiple + horizSpacing;
        }

        //        System.out.println("leftMargin=" + leftMargin);
        //        System.out.println("getMaxValueLabelStringWidth() = " + getMaxValueLabelStringWidth());
        //        System.out.println("barlength = " + barLength);
        //        System.out.println("fulcrum = " + fulcrum);

        // we want the bar to be in the middle of the text
        int barOffset = (int) (maxAscent - (((float) maxDescent + maxAscent + leading) - barHeight) / 2);

        checkPreferredSize();

        rowHeight = fontHeight + additionalVerticalSpacing;
        //System.out.println("rowHeight = " + rowHeight);

        // uncomment this to see each segment of the bar graph shown with small lines at the top
//                        if (leftJustified) {
//            g2D.setColor(Color.BLUE);
//            g2D.drawLine(leftMargin, 2, fulcrum, 2);
//            g2D.setColor(Color.RED);
//            g2D.drawLine(fulcrum + horizSpacing, 2, fulcrum + barLength + horizSpacing, 2);
//            g2D.setColor(Color.DARK_GRAY);
//            g2D.drawLine(0, 0, leftMargin, 0);
//            g2D.drawLine(fulcrum, 0, fulcrum + horizSpacing, 0);
//            g2D.drawLine(fulcrum + barLength + horizSpacing, 0, fulcrum + barLength + horizSpacing + rightMargin, 0);
//
//        } else {
//            g2D.setColor(Color.DARK_GRAY);
//            g2D.drawLine(0, 0, leftMargin, 0);
//            g2D.drawLine(leftMargin, 2, leftMargin + getMaxValueLabelStringWidth(), 2);
//            g2D.drawLine(leftMargin + getMaxValueLabelStringWidth(), 0,
//                    leftMargin + getMaxValueLabelStringWidth() + horizSpacing, 0);
//            g2D.drawLine(leftMargin + getMaxValueLabelStringWidth() + horizSpacing, 2, leftMargin + getMaxValueLabelStringWidth()
//                    + horizSpacing + barLengthMultiple, 2);
//            g2D.drawLine(leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLengthMultiple, 0, leftMargin
//                    + getMaxValueLabelStringWidth() + horizSpacing + barLengthMultiple + horizSpacing, 0);
//            g2D.drawLine(leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLengthMultiple + horizSpacing, 2,
//                    leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLengthMultiple + horizSpacing
//                            + getMaxRowLabelStringWidth(), 2);
//            g2D.drawLine(leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLengthMultiple + horizSpacing
//                    + getMaxRowLabelStringWidth(), 0, leftMargin + getMaxValueLabelStringWidth() + horizSpacing
//                    + barLengthMultiple + horizSpacing + getMaxRowLabelStringWidth() + rightMargin, 0);
//        }
        int startY = rowHeight + topMargin;

        if (singleLine == false) {
            rowHeight = (rowHeight * model.getSubSize()) + 10;
        }

        searcher.setLineHeight(rowHeight);
        searcher.setMaxDescent(fontMetrics.getMaxDescent());

        // This could be made faster, but the DrawObjects thing would have to change.
        // The problem is that if I just redraw the clipRect, then there are objects on the screen
        // that weren't in the last draw, so we would have to keep track of them some other way
        int[] clips = ParaProfUtils.computeClipping(panel.getViewport().getViewRect(), panel.getViewport().getViewRect(), true,
                fullWindow, model.getNumRows(), rowHeight, startY);
        rowStart = clips[0];
        rowEnd = clips[1];
        startY = clips[2];

        searcher.setVisibleLines(rowStart, rowEnd);
        searcher.setG2d(g2D);
        searcher.setXOffset(fulcrum);

        if (model.getSubSize() == 1) { // single graph style
            drawSingleGraph(g2D, startY, fulcrum, barOffset);
        } else {
            if (singleLine) {
                drawMultiGraphHorizontal(g2D, startY, fulcrum, barOffset);
            } else {
                drawMultiGraphVertical(g2D, startY, fulcrum, barOffset, maxAscent);
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
        // hacky way to trigger recomputation in sizeChanged(), can't do right it now
        barLengthMultiple = 0;
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
        return barLengthMultiple;
    }

    public void setBarLength(int barLength) {
        this.barLength = barLength;
        this.barLengthMultiple = barLength;
        this.preferredSizeSet = false;

        if (!stacked) {
            // When un-stacked, we face the difficult problem of determining which routines to show and 
            // we must deal with the "other" bar, which is of an unknown size
            // so we find the upper-bound on it by looking at the largest value of all the non-shown timers

            // this is an iterative process because we don't know the length of the "other" bar until we've
            // determined which routines to show, but we must adjust which routines are shown based on the 
            // length of the "other" bar
            int tmpBarLength = barLength;

            // force at least one execution of the loop
            int width = barLength + 1;
            while (width > barLength) {
                width = 0;
                double otherSum = 0;

                // sum up the widths of the visible timers and the sum of the "other" values
                for (int i = 0; i < model.getSubSize(); i++) {
                    int subIndexWidth = (int) ((maxSubValues[i] / maxSubSum * tmpBarLength));
                    if (subIndexWidth < threshold) {
                        otherSum += maxSubValues[i];
                    } else {
                        width += subIndexWidth + barHorizSpacing;
                    }
                }

                // calculate the length of the "other" bar
                int otherBarWidth = (int) ((otherSum / maxSubSum * tmpBarLength));
                width += otherBarWidth;
                width += horizSpacing + horizSpacing;

                // reduce the bar length and try again
                if (width > barLength) {
                    tmpBarLength--;
                }
            }
            this.barLengthMultiple = tmpBarLength;
        }
        checkPreferredSize();

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
        setBarLength(barLength);
        this.preferredSizeSet = false;
        checkPreferredSize();
    }

    public boolean getSingleLine() {
        return singleLine;
    }

    public void setSingleLine(boolean singleLine) {
        this.singleLine = singleLine;
    }

    public void mouseDragged(MouseEvent e) {
        // TODO Auto-generated method stub

    }

    public void mouseMoved(MouseEvent e) {
        mouseCol = -1;
        int x = e.getX();
        int y = e.getY();
        int size;

        // search the row labels
        //        int size = rowLabelDrawObjects.size();
        //        for (int i = 0; i < size; i++) {
        //            DrawObject drawObject = (DrawObject) rowLabelDrawObjects.get(i);
        //            if (x >= drawObject.getXBeg() && x <= drawObject.getXEnd() && y >= drawObject.getYBeg() && y <= drawObject.getYEnd()) {
        //                return model.getRowLabelToolTipText(i + rowStart);
        //            }
        //        }

        // search the values
        size = valueDrawObjects.size();
        for (int i = 0; i < size; i++) {
            ArrayList subList = (ArrayList) valueDrawObjects.get(i);

            int size2 = subList.size();
            for (int j = 0; j < size2; j++) {
                DrawObject drawObject = (DrawObject) subList.get(j);
                if (drawObject != null) {
                    int minx = drawObject.getXBeg();
                    int maxx = drawObject.getXEnd();
                    int miny = drawObject.getYBeg() - ((rowHeight - drawObject.getHeight()) / 2);
                    int maxy = drawObject.getYEnd() + (int) ((rowHeight - drawObject.getHeight() + 0.5) / 2);

                    if (x >= minx && x <= maxx && y >= miny && y <= maxy) {
                        mouseCol = j;
                    }
                }
            }
        }
        this.repaint();

    }

    public int getAdditionalVerticalSpacing() {
        return additionalVerticalSpacing;
    }

    public void setAdditionalVerticalSpacing(int additionalVerticalSpacing) {
        this.additionalVerticalSpacing = additionalVerticalSpacing;
    }

    public boolean getAutoResize() {
        return autoResize;
    }

    public void setAutoResize(boolean autoResize) {
        this.autoResize = autoResize;
    }
    
    public BarChartModel getBarChartModel() {
    	return this.model;
    }

}
