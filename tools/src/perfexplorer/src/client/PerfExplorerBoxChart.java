/* ----------------------------
 * BoxAndWhiskerChartDemo1.java
 * ----------------------------
 * (C) Copyright 2005, by Object Refinery Limited.
 *
 */

package edu.uoregon.tau.perfexplorer.client;


import java.awt.Color;
import java.awt.Toolkit;
import java.net.URL;
import java.util.List;

import javax.swing.JFrame;

import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.CategoryToolTipGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.renderer.category.BoxAndWhiskerRenderer;
import org.jfree.chart.renderer.category.CategoryItemRenderer;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.statistics.BoxAndWhiskerCategoryDataset;
import org.jfree.data.statistics.DefaultBoxAndWhiskerCategoryDataset;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.common.ChartDataType;
import edu.uoregon.tau.perfexplorer.common.RMIChartData;

/**
 * A simple demonstration application showing how to create a box-and-whisker
 * chart.
 */
public class PerfExplorerBoxChart extends PerfExplorerChartWindow {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5123404083269413029L;

	public PerfExplorerBoxChart (JFreeChart chart, String name) {
		super (chart, name);
	}

	public static void doIQRBoxChart() {
		// for each event, get the variation across all threads.
		PerfExplorerConnection server = PerfExplorerConnection.getConnection();
		// get the data
		RMIChartData data = server.requestChartData(PerfExplorerModel.getModel(), ChartDataType.IQR_DATA);

		// build the chart
		BoxAndWhiskerCategoryDataset dataset = createDataset(data);
        JFreeChart chart = createChart(dataset);
		JFrame frame = new PerfExplorerBoxChart(chart, "Distributions of Significant Events");
        URL url = Utility.getResource("tau32x32.gif");
		if (url != null) 
			frame.setIconImage(Toolkit.getDefaultToolkit().getImage(url));
	}

    /**
     * Returns a sample dataset.
     *
     * @return The dataset.
     */
    private static BoxAndWhiskerCategoryDataset createDataset(RMIChartData data) {
        int SERIES_COUNT = data.getRows();
        int CATEGORY_COUNT = 1;
        DefaultBoxAndWhiskerCategoryDataset result
            = new DefaultBoxAndWhiskerCategoryDataset();
        List<String> names = data.getRowLabels();
        for (int s = 0; s < SERIES_COUNT; s++) {
            for (int c = 0; c < CATEGORY_COUNT; c++) {
                List<Double> values = createValueList(data.getRowData(s));
                //result.add(values, (String)names.get(s), "Category " + c);
                result.add(values, (String)names.get(s), "");
            }
        }
        return result;
    }
   
    private static List<Double> createValueList(List<double[]> inData) {
    		List<Double> result = new java.util.ArrayList<Double>();
    		double min = ((inData.get(0)))[1];
     		double max = min;  
    		for (int i = 1; i < inData.size(); i++) {
    			if (min > ((inData.get(i)))[1])
    				min = ((inData.get(i)))[1];
    			if (max < ((inData.get(i)))[1])
    				max = ((inData.get(i)))[1];
    		}
    		double range = max - min;
    		//System.out.println("Min: " + min + ", Max: " + max + ", Range: " + range);
    		for (int i = 0; i < inData.size(); i++) {
    			result.add(new Double((((double[])(inData.get(i)))[1]-min)/range));   
    		}
    		return result;
    }

//    private static List<Double> createValueList(double lowerBound, double upperBound,
//                                        int count) {
//        List<Double> result = new java.util.ArrayList<Double>();
//        for (int i = 0; i < count; i++) {
//            double v = lowerBound + (Math.random() * (upperBound - lowerBound));
//            result.add(new Double(v));   
//        }
//        return result;
//    }
   
    /**
     * Creates a sample chart.
     *
     * @param dataset  the dataset.
     *
     * @return The chart.
     */
    private static JFreeChart createChart(BoxAndWhiskerCategoryDataset dataset) {
       
        CategoryAxis domainAxis = new CategoryAxis(null);
        NumberAxis rangeAxis = new NumberAxis("Value");
        CategoryItemRenderer renderer = new BoxAndWhiskerRenderer();
        renderer.setBaseToolTipGenerator(new CategoryToolTipGenerator() {
            public String generateToolTip(CategoryDataset inDataset, int arg1, int arg2) {
				BoxAndWhiskerCategoryDataset dataset = 
					(BoxAndWhiskerCategoryDataset) inDataset;
                return "<html>Min Outlier: " + dataset.getMinOutlier(arg1, arg2) + 
                "<BR>Min Regular Value: " + dataset.getMinRegularValue(arg1, arg2) + 
                "<BR>Q1 Value: " + dataset.getQ1Value(arg1, arg2) + 
                "<BR>Mean Value: " + dataset.getMeanValue(arg1, arg2) + 
                "<BR>Q3 Value: " + dataset.getQ3Value(arg1, arg2) + 
                "<BR>Max Regular Value: " + dataset.getMaxRegularValue(arg1, arg2) + 
                "<BR>Max Outlier: " + dataset.getMaxOutlier(arg1, arg2) + "</html>";
            }
        });

        CategoryPlot plot = new CategoryPlot(
            dataset, domainAxis, rangeAxis, renderer
        );
        JFreeChart chart = new JFreeChart("Significant (>2.0% of runtime) Event IQR Boxplot with Outliers", plot);
       
        chart.setBackgroundPaint(Color.white);

        //plot.setBackgroundPaint(Color.lightGray);
        plot.setBackgroundPaint(Color.white);
        plot.setDomainGridlinePaint(Color.white);
        plot.setDomainGridlinesVisible(true);
        plot.setRangeGridlinePaint(Color.white);

        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());

        // OPTIONAL CUSTOMISATION COMPLETED.
       
        return chart;
       
    }
  
}
