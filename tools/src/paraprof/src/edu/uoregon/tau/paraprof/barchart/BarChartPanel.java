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

public class BarChartPanel extends JScrollPane implements Printable, ImageExport, ScrollBarController {

    BarChart barChart;

    //BarChartHeader barChartHeader;

    public BarChartPanel(BarChartModel barChartModel, JComponent header) {
        barChart = new BarChart(barChartModel, this);
        this.setViewportView(barChart);
        this.setColumnHeaderView(header);

        //JButton button = new JButton("asdf");
        //button.setMinimumSize(new Dimension(1730,400));
        //button.setPreferredSize(new Dimension(1730,400));
        //button.setMaximumSize(new Dimension(1730,400));

        //this.setViewportView(button);
    }

    //    public static void main(String[] args) {
    //        // TODO Auto-generated method stub
    //
    //        AbstractBarChartModel model = new AbstractBarChartModel() {
    //
    //            int numRows = 25;
    //
    //            public int getSubSize() {
    //                return 9;
    //            };
    //
    //            public int getNumRows() {
    //                return numRows;
    //            }
    //
    //            public String getRowLabel(int row) {
    //                return "n,c,t " + row + "," + row + "," + row;
    //            }
    //
    //            public double getValue(int row, int subIndex) {
    //                return subIndex + row * subIndex * subIndex * 200 + (((row * 32) % 115 + (subIndex + 50) * 25) % 253);
    //            }
    //
    //            public String getValueLabel(int row, int subIndex) {
    //                if (subIndex == 0) {
    //                    return Double.toString((numRows - row) * 5) + "0000";
    //                } else {
    //                    return Double.toString(row * 5) + "0000";
    //                }
    //            }
    //
    //            public double getMaxValue() {
    //                return numRows * 5;
    //            }
    //
    //            public Color getValueColor(int row, int subIndex) {
    //                return new Color((row * 30 + 36 + subIndex * 30) % 250, (row * 40 + 64) % 250, (row * 70 + 35) % 250);
    //            }
    //
    //            public Color getValueHighlightColor(int row, int subIndex) {
    //                if (row % 3 == 0) {
    //                    return Color.red;
    //                }
    //                return null;
    //            }
    //
    //            public String getRowValueLabel(int row) {
    //                return "none";
    //            }
    //
    //            public void revalidate() {
    //                // TODO Auto-generated method stub
    //                
    //            }
    //
    //            public boolean hasChanged() {
    //                // TODO Auto-generated method stub
    //                return false;
    //            }
    //
    //            public void reportValueClick(int row, int subIndex) {
    //                // TODO Auto-generated method stub
    //                
    //            }
    //
    //            public void reportRowLabelClick(int row) {
    //                // TODO Auto-generated method stub
    //                
    //            }
    //        };
    //
    //        JFrame frame = new JFrame();
    //        frame.setTitle("Testing Bar Chart");
    //        frame.setSize(600, 400);
    //        frame.setLocation(500, 300);
    //
    //        frame.getContentPane().add(new BarChartPanel(model, new JTextArea("Hi!")));
    //        frame.show();
    //        frame.addWindowListener(new WindowListener() {
    //
    //            public void windowActivated(WindowEvent e) {
    //                // TODO Auto-generated method stub
    //
    //            }
    //
    //            public void windowClosed(WindowEvent e) {
    //                System.exit(0);
    //                // TODO Auto-generated method stub
    //
    //            }
    //
    //            public void windowClosing(WindowEvent e) {
    //                System.exit(0);
    //                // TODO Auto-generated method stub
    //
    //            }
    //
    //            public void windowDeactivated(WindowEvent e) {
    //                // TODO Auto-generated method stub
    //
    //            }
    //
    //            public void windowDeiconified(WindowEvent e) {
    //                // TODO Auto-generated method stub
    //
    //            }
    //
    //            public void windowIconified(WindowEvent e) {
    //                // TODO Auto-generated method stub
    //
    //            }
    //
    //            public void windowOpened(WindowEvent e) {
    //                // TODO Auto-generated method stub
    //
    //            }
    //        });
    //    }

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
