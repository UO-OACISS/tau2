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

public class BarChart extends JPanel implements MouseListener, Printable, ImageExport, BarChartModelListener {

    private BarChartModel model;

    private int maxRowLabelStringWidth;
    private boolean maxRowLabelStringWidthSet;

    private int maxValueLabelStringWidth;
    private boolean maxValueLabelStringWidthSet;

    private FontMetrics fontMetrics;
    private boolean leftJustified = false;

    private boolean preferredSizeSet = false;

    private int barLength = 400;
    private int leftMargin = 5;
    private int rightMargin = 5;
    private int horizSpacing = 10;
    private int barVerticalSpacing = 4;
    private int barHeight;

    private int barHorizSpacing = 5;

    private boolean dataProcessed = false;
    private boolean stacked = true;
    private boolean normalized = true;

    private int threshold = 2;

    private int rowStart;
    private ArrayList rowLabelDrawObjects = new ArrayList();
    private ArrayList valueDrawObjects = new ArrayList();

    private BarChartPanel panel;
    private Searcher searcher;

    public BarChart(BarChartModel model, BarChartPanel panel) {
        this.model = model;
        this.panel = panel;
        model.addBarChartModelListener(this);

        setBackground(Color.white);
        addMouseListener(this);

        setDoubleBuffered(true);
        setOpaque(true);

        searcher = new Searcher(this, panel);
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

        int size = rowLabelDrawObjects.size();
        for (int i = 0; i < size; i++) {
            DrawObject drawObject = (DrawObject) rowLabelDrawObjects.get(i);
            if (x >= drawObject.getXBeg() && x <= drawObject.getXEnd() && y >= drawObject.getYBeg() && y <= drawObject.getYEnd()) {
                return model.getRowLabelToolTipText(i + rowStart);
            }
        }

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
            //System.out.println("rect = " + rect);
            //g.setColor(Color.white);
            //g.clearRect(rect.x, rect.y, rect.width, rect.height);
            //            super.paintComponent(g);

            g.setColor(Color.white);
            g.fillRect(rect.x, rect.y, rect.width, rect.height);
            
            //Rectangle viewRect = panel.getViewport().getViewRect();
            //g.translate(0,viewRect.y);
            export((Graphics2D) g, true, false, false);
        } catch (Exception e) {
            ParaProfUtils.handleException(e);
            //            window.closeThisWindow();
        }
    }

    public void mouseClicked(MouseEvent e) {
        int x = e.getX();
        int y = e.getY();

        int size = rowLabelDrawObjects.size();
        for (int i = 0; i < size; i++) {
            DrawObject drawObject = (DrawObject) rowLabelDrawObjects.get(i);
            if (x >= drawObject.getXBeg() && x <= drawObject.getXEnd() && y >= drawObject.getYBeg() && y <= drawObject.getYEnd()) {
                model.reportRowLabelClick(i + rowStart, e, this);
                return;
            }
        }

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
                            model.reportValueClick(i + rowStart, j, e, this);
                        }
                    }
                }

            }
        }

    }

    public void mouseEntered(MouseEvent e) {
        // TODO Auto-generated method stub

    }

    public void mouseExited(MouseEvent e) {
        // TODO Auto-generated method stub

    }

    public void mousePressed(MouseEvent e) {
        // TODO Auto-generated method stub

    }

    public void mouseReleased(MouseEvent e) {
        // TODO Auto-generated method stub

    }

    public int print(Graphics graphics, PageFormat pageFormat, int pageIndex) throws PrinterException {
        // TODO Auto-generated method stub
        return 0;
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

        maxHeight = model.getNumRows() * (barHeight + barVerticalSpacing) + (barVerticalSpacing * 2);

        if (model.getSubSize() == 1) {
            maxWidth = barLength + getMaxRowLabelStringWidth() + getMaxValueLabelStringWidth() + leftMargin + (2 * horizSpacing)
                    + rightMargin;
        } else {
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
        }
        super.setSize(new Dimension(maxWidth, maxHeight));
        super.setPreferredSize(new Dimension(maxWidth, maxHeight));
        preferredSizeSet = true;
        this.invalidate();
    }

    double maxRowValues[];
    double maxSubValues[];
    double maxRowSum;
    double rowSums[];

    private void processData() {
        if (dataProcessed) {
            return;
        }

        dataProcessed = true;
        maxRowValues = new double[model.getNumRows()];
        maxSubValues = new double[model.getSubSize()];
        rowSums = new double[model.getNumRows()];
        maxRowSum = 0;

        for (int row = 0; row < model.getNumRows(); row++) {
            double rowSum = 0;
            for (int i = 0; i < model.getSubSize(); i++) {
                double value = model.getValue(row, i);
                maxRowValues[row] = Math.max(maxRowValues[row], value);
                maxSubValues[i] = Math.max(maxSubValues[i], value);
                rowSum += value;
                rowSums[row] += value;
            }
            maxRowSum = Math.max(maxRowSum, rowSum);
        }
    }


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
    private Color darker(Color c) {
        int r = c.getRed();
        int g = c.getGreen();
        int b = c.getBlue();

        int max = Math.max(r, g);
        max = Math.max(max, b);
        max = Math.max(max, 255);

        r = r - (int) ((max - r) / 2.36);
        g = g - (int) ((max - g) / 2.36);
        b = b - (int) ((max - b) / 2.36);
        r = Math.max(r,0);
        g = Math.max(g,0);
        b = Math.max(b,0);
        return new Color(r, g, b);
    }

    private void drawBar(Graphics2D g2D, int x, int y, int length, int height, Color color, Color highlight) {
        boolean special = true;

        if (special && height > 4) {
            g2D.setColor(color);

            g2D.fillRect(x, y, length, height);

            g2D.setColor(lighter(color));

            int innerHeight = height / 4;
            g2D.fillRect(x, y + (innerHeight / 2) + 1, length, innerHeight);

            int innerHeight2 = innerHeight / 3;
            g2D.setColor(lighter(lighter(color)));
            g2D.fillRect(x, y + (innerHeight / 2) + 1 + innerHeight2, length, innerHeight2);

            
            g2D.setColor(Color.black);
            if (highlight != null) {
                g2D.setColor(highlight);
                g2D.drawRect(x + 1, y + 1, length - 2, height - 2);
            }
            g2D.drawRect(x, y, length, barHeight);

        } else {
            g2D.setColor(color);
            g2D.fillRect(x, y, length, height);

            if (height > 3) {
                g2D.setColor(Color.black);
                if (highlight != null) {
                    g2D.setColor(highlight);
                    g2D.drawRect(x + 1, y + 1, length - 2, height - 2);
                }
                g2D.drawRect(x, y, length, barHeight);
            }
        }
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {

        rowLabelDrawObjects.clear();
        valueDrawObjects.clear();

        //To make sure the bar details are set, this method must be called.
        ParaProf.preferencesWindow.setBarDetails(g2D);

        //Now safe to grab spacing and bar heights.
        barHeight = ParaProf.preferencesWindow.getBarHeight();
        barVerticalSpacing = ParaProf.preferencesWindow.getBarSpacing() - barHeight;

        Font font = new Font(ParaProf.preferencesWindow.getParaProfFont(), ParaProf.preferencesWindow.getFontStyle(), barHeight);
        g2D.setFont(font);
        fontMetrics = g2D.getFontMetrics(font);

        //Obtain the font and its metrics.
        //Font font = new Font(ParaProf.getPreferencesWindow().getParaProfFont(), ppTrial.getPreferencesWindow().getFontStyle(),
        //        barHeight);
        //g2D.setFont(font);

        //g2D.setFont(font);
        //fontMetrics = g2D.getFontMetrics(g2D.getFont());
        //barHeight = fontMetrics.getMaxAscent() + fontMetrics.getMaxDescent();
        int fontHeight = fontMetrics.getMaxAscent();

        processData();
        int fulcrum;

        if (leftJustified) {
            fulcrum = leftMargin + getMaxRowLabelStringWidth();
        } else {
            fulcrum = leftMargin + getMaxValueLabelStringWidth() + horizSpacing + barLength + horizSpacing;
        }

        //System.out.println("fulcrum = " + fulcrum);
        int barOffset = fontMetrics.getMaxAscent();

        checkPreferredSize();

        int rowHeight = barHeight + barVerticalSpacing;

        searcher.setLineHeight(rowHeight);
        searcher.setMaxDescent(fontMetrics.getMaxDescent());

        int y = rowHeight;

        // this could be made faster, but the DrawObjects thing would have to change

        // determine which elements to draw (clipping)
        //int[] clips = ParaProfUtils.computeClipping(g2D.getClipBounds(), g2D.getClipBounds(), toScreen, fullWindow, model.getNumRows(),
        //      rowHeight, y);
        int[] clips = ParaProfUtils.computeClipping(panel.getViewport().getViewRect(), panel.getViewport().getViewRect(), true,
                fullWindow, model.getNumRows(), rowHeight, y);
        rowStart = clips[0];
        int rowEnd = clips[1];
        y = clips[2];

        //double maxValue = model.getMaxValue();
        double maxValue = maxRowSum;

        //        System.err.println("\nrowStart = " + rowStart);
        //        System.err.println("rowEnd = " + rowEnd);

        searcher.setVisibleLines(rowStart, rowEnd);
        searcher.setG2d(g2D);
        searcher.setXOffset(fulcrum);

        for (int row = rowStart; row <= rowEnd; row++) {
            String rowLabel = model.getRowLabel(row);
            int rowLabelStringWidth = fontMetrics.stringWidth(rowLabel);

            ArrayList subDrawObjects = new ArrayList();
            valueDrawObjects.add(subDrawObjects);
            if (model.getSubSize() == 1) {

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

                y = y + barHeight + barVerticalSpacing;
                subDrawObjects.add(null); // "other"

            } else {
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
                rowLabelDrawObjects.add(new DrawObject(rowLabelPosition, y - fontHeight, rowLabelPosition + rowLabelStringWidth,
                        y));

                double otherValue = 0;

                for (int i = 0; i < model.getSubSize(); i++) {
                    g2D.setColor(model.getValueColor(row, i));
                    double value = model.getValue(row, i);
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
                                drawBar(g2D, barStartX, y - barOffset, length, barHeight, model.getValueColor(row, i),
                                        model.getValueHighlightColor(row, i));

                                subDrawObjects.add(new DrawObject(barStartX, y - barOffset, barStartX + length, y - barOffset
                                        + barHeight));
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
                g2D.setColor(Color.black);
                g2D.fillRect(barStartX, y - barOffset, otherLength, barHeight);
                g2D.drawRect(barStartX, barStartY, otherLength, barHeight);
                subDrawObjects.add(new DrawObject(barStartX, y - barOffset, barStartX + otherLength, y - barOffset + barHeight));

                y = y + rowHeight;
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

}
