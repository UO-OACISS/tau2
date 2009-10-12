package edu.uoregon.tau.perfexplorer.common;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartTheme;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.StandardChartTheme;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.labels.StandardCategoryToolTipGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.StackedBarRenderer;
import org.jfree.chart.urls.StandardCategoryURLGenerator;
import org.jfree.data.category.CategoryDataset;

public abstract class CustomChartFactory extends ChartFactory{
	
	/** The chart theme. */
    private static ChartTheme currentTheme = new StandardChartTheme("JFree");
	
	/**
     * Creates a stacked bar chart with default settings.  The chart object
     * returned by this method uses a {@link CategoryPlot} instance as the
     * plot, with a {@link CategoryAxis} for the domain axis, a
     * {@link NumberAxis} as the range axis, and a {@link StackedBarRenderer}
     * as the renderer.
     *
     * @param title  the chart title (<code>null</code> permitted).
     * @param domainAxisLabel  the label for the category axis
     *                         (<code>null</code> permitted).
     * @param rangeAxisLabel  the label for the value axis
     *                        (<code>null</code> permitted).
     * @param dataset  the dataset for the chart (<code>null</code> permitted).
     * @param orientation  the orientation of the chart (horizontal or
     *                     vertical) (<code>null</code> not permitted).
     * @param legend  a flag specifying whether or not a legend is required.
     * @param tooltips  configure chart to generate tool tips?
     * @param urls  configure chart to generate URLs?
     *
     * @return A stacked bar chart.
     */
    public static JFreeChart createAlignedStackedBarChart(String title,
                                                   String domainAxisLabel,
                                                   String rangeAxisLabel,
                                                   CategoryDataset dataset,
                                                   PlotOrientation orientation,
                                                   boolean legend,
                                                   boolean tooltips,
                                                   boolean urls) {

        if (orientation == null) {
            throw new IllegalArgumentException("Null 'orientation' argument.");
        }

        CategoryAxis categoryAxis = new CategoryAxis(domainAxisLabel);
        ValueAxis valueAxis = new NumberAxis(rangeAxisLabel);
        valueAxis.setVisible(false);
        AlignedStackedBarRenderer renderer = new AlignedStackedBarRenderer();
        if (tooltips) {
            renderer.setBaseToolTipGenerator(
                    new StandardCategoryToolTipGenerator());
        }
        if (urls) {
            renderer.setBaseItemURLGenerator(
                    new StandardCategoryURLGenerator());
        }

        CategoryPlot plot = new CategoryPlot(dataset, categoryAxis, valueAxis,
                renderer);
        plot.setOrientation(orientation);
        JFreeChart chart = new JFreeChart(title, JFreeChart.DEFAULT_TITLE_FONT,
                plot, legend);
        currentTheme.apply(chart);
        return chart;

    }
}
