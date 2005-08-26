package edu.uoregon.tau.paraprof.barchart;

import java.awt.*;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;

import javax.swing.JComponent;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;

import edu.uoregon.tau.paraprof.ParaProfErrorDialog;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.paraprof.interfaces.ImageExport;
import edu.uoregon.tau.paraprof.interfaces.ScrollBarController;

/**
 * Adds scroll ability, and handles image export/printing with header support.
 * 
 * <P>CVS $Id: BarChartPanel.java,v 1.2 2005/08/26 01:49:03 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class BarChartPanel extends JScrollPane implements Printable, ImageExport, ScrollBarController {

    BarChart barChart;

    //BarChartHeader barChartHeader;

    public BarChartPanel(BarChartModel barChartModel, JComponent header) {
        barChart = new BarChart(barChartModel, this);
        this.setViewportView(barChart);
        this.setColumnHeaderView(header);

    }

   

    public int print(Graphics graphics, PageFormat pageFormat, int pageIndex) throws PrinterException {
        try {
            if (pageIndex >= 1) {
                return NO_SUCH_PAGE;
            }
            Dimension size = this.getImageSize(true, true);
            ParaProfUtils.scaleForPrint(graphics, pageFormat, (int)size.getWidth(), (int)size.getHeight());

            this.getColumnHeader().paintAll(graphics);
            graphics.translate(0, this.getColumnHeader().getHeight());
            export((Graphics2D) graphics, false, true, false);
            return Printable.PAGE_EXISTS;

        } catch (Exception e) {
            new ParaProfErrorDialog(e);
            return NO_SUCH_PAGE;
        }

    }

    public BarChart getBarChart() {
        return barChart;
    }

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader) {
        if (drawHeader) {
            this.getColumnHeader().paintAll(g2D);
            g2D.translate(0, this.getColumnHeader().getHeight());
        }

        Rectangle rect = this.getViewport().getViewRect();

        g2D.translate(0, -rect.getMinY());
        barChart.export(g2D, false, fullWindow, false);

    }

    public Dimension getImageSize(boolean fullScreen, boolean header) {
        if (header) {
            Dimension d = this.getColumnHeader().getSize();

            Dimension chart;
            if (fullScreen) {
                chart = barChart.getSize();
            } else {
                chart = this.getViewport().getExtentSize();
            }
            return new Dimension((int) Math.max(d.getWidth(), chart.getWidth()), (int) (d.getHeight() + chart.getHeight()));
        } else {
            if (fullScreen) {
                return barChart.getSize();
            } else {
                return this.getViewport().getExtentSize();
            }
        }

    }

    public void setVerticalScrollBarPosition(int position) {
        JScrollBar scrollBar = this.getVerticalScrollBar();
        scrollBar.setValue(position);
    }

    public void setHorizontalScrollBarPosition(int position) {
        JScrollBar scrollBar = this.getHorizontalScrollBar();
        scrollBar.setValue(position);
    }

    public Dimension getThisViewportSize() {
        return this.getViewport().getExtentSize();
    }

}
