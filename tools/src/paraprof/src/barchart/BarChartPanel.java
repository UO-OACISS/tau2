package edu.uoregon.tau.paraprof.barchart;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.MouseEvent;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JScrollBar;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import edu.uoregon.tau.common.ImageExport;
import edu.uoregon.tau.paraprof.DataSorter;
import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.paraprof.interfaces.ScrollBarController;

/**
 * Adds scroll ability, and handles image export/printing with header support.
 * 
 * <P>CVS $Id: BarChartPanel.java,v 1.9 2009/11/05 09:43:32 khuck Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.9 $
 */
public class BarChartPanel extends JScrollPane implements Printable, ImageExport, ScrollBarController {

    /**
	 * 
	 */
	private static final long serialVersionUID = 4417813804029452411L;
	BarChart barChart;

    //BarChartHeader barChartHeader;

    public BarChartPanel(BarChartModel barChartModel) {
        this(barChartModel, null);
    }
    
    public BarChartPanel(BarChartModel barChartModel, JComponent header) {
        barChart = new BarChart(barChartModel, this);
        this.setViewportView(barChart);
        this.setColumnHeaderView(header);
        this.getVerticalScrollBar().setUnitIncrement(35);
            
        addComponentListener(new ComponentListener() {

            public void componentHidden(ComponentEvent e) {
            }

            public void componentMoved(ComponentEvent e) {
            }

            public void componentResized(ComponentEvent e) {
                barChart.sizeChanged();
            }

            public void componentShown(ComponentEvent e) {
            }
        });
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
            ParaProfUtils.handleException(e);
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

        // translate to the clipped area
        Rectangle rect = this.getViewport().getViewRect();
        g2D.translate(0, -rect.getMinY());
        barChart.export(g2D, false, fullWindow);
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

    
    public static void main(String[] args) {
        //final ParaProf paraProf = 
    	new ParaProf();
        ParaProf.initialize();

        
        JFrame frame = new JFrame("Bar Chart Test");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        
        BarChartModel model = new AbstractBarChartModel() {

            public int getNumRows() {
                // TODO Auto-generated method stub
                return 5;
            }

            public int getSubSize() {
                return 3;
            };
            
            public String getRowLabel(int row) {
                // TODO Auto-generated method stub
                return "row " + row;
            }

            public String getValueLabel(int row, int subIndex) {
                // TODO Auto-generated method stub
                return "xygYpAcol " + subIndex;
            }

            public double getValue(int row, int subIndex) {
                // TODO Auto-generated method stub
                return (row+5) * (subIndex+15);
            }

            public Color getValueColor(int row, int subIndex) {
                // TODO Auto-generated method stub
                return Color.orange;
            }

            public Color getValueHighlightColor(int row, int subIndex) {
                // TODO Auto-generated method stub
                return null;
            }

            public void fireValueClick(int row, int subIndex, MouseEvent e, JComponent owner) {
                // TODO Auto-generated method stub
                
            }

            public void fireRowLabelClick(int row, MouseEvent e, JComponent owner) {
                // TODO Auto-generated method stub
                
            }

            public String getValueToolTipText(int row, int subIndex) {
                // TODO Auto-generated method stub
                return null;
            }

            public String getRowLabelToolTipText(int row) {
                // TODO Auto-generated method stub
                return null;
            }

            public String getOtherToolTopText(int row) {
                // TODO Auto-generated method stub
                return null;
            }

            public void reloadData() {
                // TODO Auto-generated method stub
                
            }
            
            public DataSorter getDataSorter() {
            	return null;
            }
        };
        
        
        BarChartPanel panel = new BarChartPanel(model, new JTextArea("asdf"));

        panel.getBarChart().setLeftJustified(true);
        panel.getBarChart().setSingleLine(false);
        panel.getBarChart().setBarLength(300);
        
        frame.getContentPane().add(panel);
        //Display the window.
        frame.pack();
        frame.setVisible(true);
        frame.setLocation(500,500);
        frame.setSize(640,480);
    }
    
}
