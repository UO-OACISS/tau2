package edu.uoregon.tau.perfexplorer.server;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.List;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.DefaultXYItemRenderer;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.Range;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.function.Function2D;
import org.jfree.data.function.LineFunction2D;
import org.jfree.data.function.PowerFunction2D;
import org.jfree.data.general.DatasetUtilities;
import org.jfree.data.statistics.Regression;
import org.jfree.data.xy.XYDataset;


import edu.uoregon.tau.perfexplorer.clustering.DataNormalizer;
import edu.uoregon.tau.perfexplorer.clustering.RawDataInterface;
import edu.uoregon.tau.perfexplorer.clustering.weka.AnalysisFactory;
import edu.uoregon.tau.perfexplorer.common.ChartType;
import edu.uoregon.tau.perfexplorer.common.RMIPerfExplorerModel;
import edu.uoregon.tau.perfexplorer.constants.Constants;
import edu.uoregon.tau.common.Utility;

public class ImageUtils {
	
	/**
	 * This method is used to generate the thumbnails for the 
	 * stacked bar graphs for showing the breakdown of min, avg, and max behavior of each cluster.
	 * 
	 * @param centroids
	 * @param deviations
	 * @param rowLabels
	 * @return
	 */
	public static File generateBreakdownThumbnail(RMIPerfExplorerModel modelData, RawDataInterface centroids, RawDataInterface deviations, List<String> rowLabels) {
		// create a JFreeChart of this analysis data.  Create a stacked bar chart
		// with standard deviation bars?
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
		for (int x = 0 ; x < centroids.numVectors() ; x++) {
			for (int y = 0 ; y < centroids.numDimensions() ; y++) {
				dataset.addValue(centroids.getValue(x,y), rowLabels.get(y), new String(Integer.toString(x)));
			}
		}
        JFreeChart chart = ChartFactory.createStackedBarChart(
        		null, null, null,     // range axis label
            dataset,                         // data
            PlotOrientation.HORIZONTAL,        // the plot orientation
            false,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		Utility.applyDefaultChartTheme(chart);
        File outfile = new File(Constants.TMPDIR + "thumbnail." + modelData.toShortString() + ".png");
        try {
        		ChartUtilities.saveChartAsPNG(outfile, chart, 100, 100);
        } catch (IOException e) {
            System.err.println(e.getMessage());
			e.printStackTrace();
		}
        return outfile;
	}

 	/**
 	 * This method is used to generate the 
	 * stacked bar graphs for showing the breakdown of min, avg, and max behavior of each cluster.
	 * 
	 * @param centroids
 	 * @param deviations
 	 * @param rowLabels
 	 * @return
 	 */
 	public static File generateBreakdownImage(ChartType chartType, RMIPerfExplorerModel modelData, RawDataInterface centroids, RawDataInterface deviations, List<String> rowLabels) {
 		// create a JFreeChart of this analysis data.  Create a stacked bar chart
 		// with standard deviation bars?
         DefaultCategoryDataset dataset = new DefaultCategoryDataset();
 		for (int x = 0 ; x < centroids.numVectors() ; x++) {
 			for (int y = 0 ; y < centroids.numDimensions() ; y++) {
 				dataset.addValue(centroids.getValue(x,y), rowLabels.get(y), new String(Integer.toString(x)));
 			}
 		}
 		String chartTitle = modelData.toString();
 		if (chartType == ChartType.CLUSTER_AVERAGES) {
             chartTitle = chartTitle + " Average Values";
 		}
 		if (chartType == ChartType.CLUSTER_MAXIMUMS) {
             chartTitle = chartTitle + " Maximum Values";
 		}
 		if (chartType == ChartType.CLUSTER_MINIMUMS) {
             chartTitle = chartTitle + " Minimum Values";
 		}
         JFreeChart chart = ChartFactory.createStackedBarChart(
             chartTitle,  // chart title
             "Cluster Number",          // domain axis label
             "Total Runtime",     // range axis label
             dataset,                         // data
             PlotOrientation.HORIZONTAL,        // the plot orientation
             true,                            // legend
             true,                            // tooltips
             false                            // urls
         );
		Utility.applyDefaultChartTheme(chart);
         File outfile = new File(Constants.TMPDIR + "image." + modelData.toShortString() + ".png");
         try {
         		ChartUtilities.saveChartAsPNG(outfile, chart, 500, 500);
         } catch (IOException e) {
             System.err.println(e.getMessage());
 			e.printStackTrace();
 		}
         return outfile;
 	}

	/**
	 * This method is used to generate the thumbnail of the bargraph histogram showing the cluster sizes.
	 * 
	 * @param clusterSizes
	 * @param rowLabels
	 * @return
	 */
	public static File generateClusterSizeThumbnail(RMIPerfExplorerModel modelData, int[] clusterSizes) {//, List rowLabels
		// create a JFreeChart of this analysis data.  Create a stacked bar chart
		// with standard deviation bars?
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
		for (int x = 0 ; x < clusterSizes.length ; x++) {
			dataset.addValue(clusterSizes[x], "Threads in cluster", new String(Integer.toString(x)));
		}
        JFreeChart chart = ChartFactory.createStackedBarChart(
        		null, null, null,     // range axis label
            dataset,                         // data
            PlotOrientation.HORIZONTAL,        // the plot orientation
            false,                            // legend
            true,                            // tooltips
            false                            // urls
        );
		Utility.applyDefaultChartTheme(chart);
        File outfile = new File(Constants.TMPDIR + "thumbnail." + modelData.toShortString() + ".png");
        try {
        		ChartUtilities.saveChartAsPNG(outfile, chart, 100, 100);
        } catch (IOException e) {
            System.err.println(e.getMessage());
			e.printStackTrace();
		}
        return outfile;
	}

 	/**
 	 * This method is used to generate the bargraph histogram showing the cluster sizes.
	 * 
	 * @param clusterSizes
 	 * @param rowLabels
 	 * @return
 	 */
 	public static File generateClusterSizeImage(RMIPerfExplorerModel modelData, int[] clusterSizes) {//, List rowLabels
 		// create a JFreeChart of this analysis data.  Create a stacked bar chart
 		// with standard deviation bars?
         DefaultCategoryDataset dataset = new DefaultCategoryDataset();
 		for (int x = 0 ; x < clusterSizes.length ; x++) {
 			dataset.addValue(clusterSizes[x], "Threads in cluster", new String(Integer.toString(x)));
 		}
         JFreeChart chart = ChartFactory.createStackedBarChart(
             modelData.toString(),  // chart title
             "Cluster Number",          // domain axis label
             "Threads in cluster",     // range axis label
             dataset,                         // data
             PlotOrientation.HORIZONTAL,        // the plot orientation
             true,                            // legend
             true,                            // tooltips
             false                            // urls
         );
		Utility.applyDefaultChartTheme(chart);
         File outfile = new File(Constants.TMPDIR + "image." + modelData.toShortString() + ".png");
         try {
         		ChartUtilities.saveChartAsPNG(outfile, chart, 500, 500);
         } catch (IOException e) {
             System.err.println(e.getMessage());
 			e.printStackTrace();
 		}
         return outfile;
 	}

	/**
	 * This method is used to generate the thumbnail of the scatterplot of two of the data events in the clustering result.
	 * 
	 * @param pcaData
	 * @param rawData
	 * @param clusterer
	 * @return
	 */
	public static File generateClusterScatterplotThumbnail(ChartType chartType, RMIPerfExplorerModel modelData, RawDataInterface[] clusters) {
		File outfile = null;
		if (chartType == ChartType.PCA_SCATTERPLOT) {
	        XYDataset data = new PCAPlotDataset(clusters, false);
	        JFreeChart chart = ChartFactory.createScatterPlot(
	            null, null, null, data, PlotOrientation.VERTICAL, false, false, false);
	        outfile = new File(Constants.TMPDIR + "thumbnail." + modelData.toShortString() + ".png");
	        try {
        			ChartUtilities.saveChartAsPNG(outfile, chart, 100, 100);
	        } catch (IOException e) {
            	System.err.println(e.getMessage());
				e.printStackTrace();
			}
		}
        return outfile;
	}

	/**
	 * This method is used to generate the scatterplot of two of the data events in the clustering result.

	 * @param pcaData
	 * @param rawData
	 * @param clusterer
	 * @return
	 */
	public static File generateClusterScatterplotImage(ChartType chartType, RMIPerfExplorerModel modelData, RawDataInterface pcaData, RawDataInterface[] clusters, boolean hasNoise) {
		File outfile = null;
		if (chartType == ChartType.PCA_SCATTERPLOT) {
		/*
			int max = pcaData.numDimensions();
			int x = max - 1;
			int y = max - 2;
			if (max < 2) {
				x = 0;
				y = 0;
			}
			*/
	        XYDataset data = new PCAPlotDataset(clusters, hasNoise);
	        JFreeChart chart = ChartFactory.createScatterPlot(
	            "Correlation Results",
	            (String)(Utility.shortenFunctionName(pcaData.getEventNames().get(0))),
	            (String)(Utility.shortenFunctionName(pcaData.getEventNames().get(1))),
	            data,
	            PlotOrientation.VERTICAL,
	            true,
	            false,
	            false
	        );
			Utility.applyDefaultChartTheme(chart);
			if (hasNoise) {
        		XYPlot plot = (XYPlot)chart.getPlot();
				XYItemRenderer renderer = plot.getRenderer();
	    		renderer.setSeriesPaint(clusters.length - 1, Color.black);
			}
	        outfile = new File(Constants.TMPDIR + "image." + modelData.toShortString() + ".png");
	        try {
	        		ChartUtilities.saveChartAsPNG(outfile, chart, 500, 500);
	        } catch (IOException e) {
            	System.err.println(e.getMessage());
				e.printStackTrace();
			}
		}
        return outfile;
	}

    /**
     * This method is used to do the correlation scatterplot thumbnail in the correlation analysis.
     * 
     * @param pcaData
     * @param i
     * @param j
     * @return
     */
    public static File generateCorrelationScatterplotThumbnail(ChartType chartType, RMIPerfExplorerModel modelData, RawDataInterface pcaData, int i, int j, boolean correlateToMain) {
        File outfile = null;
        if (chartType == ChartType.CORRELATION_SCATTERPLOT) {
            DataNormalizer normalizer = AnalysisFactory.createDataNormalizer(pcaData);
            RawDataInterface normalData = normalizer.getNormalizedData();
            XYDataset data = new ScatterPlotDataset(normalData,
            modelData.toString(), i, j, correlateToMain);
            JFreeChart chart = ChartFactory.createScatterPlot(
            null, null, null, data, PlotOrientation.VERTICAL, false, false, false);
            outfile = new File(Constants.TMPDIR + "thumbnail." + modelData.toShortString() + ".png");
            try {
                ChartUtilities.saveChartAsPNG(outfile, chart, 100, 100);
            } catch (IOException e) {
            	System.err.println(e.getMessage());
				e.printStackTrace();
			}
        }
        return outfile;
    }

    /**
     * This method is used to do the correlation scatterplot in the correlation analysis.
     *
     * @param pcaData
     * @param i
     * @param j
     * @return
     */
     public static File generateCorrelationScatterplotImage(ChartType chartType, RMIPerfExplorerModel modelData, 
    		 RawDataInterface pcaData, int i, int j, boolean correlateToMain, double rCorrelation) {
         File outfile = null;
         if (chartType == ChartType.CORRELATION_SCATTERPLOT) {
             DataNormalizer normalizer = AnalysisFactory.createDataNormalizer(pcaData);
             RawDataInterface normalData = normalizer.getNormalizedData();
             XYDataset data = new ScatterPlotDataset(normalData,
             modelData.toString(), i, j, correlateToMain);
             // Create the chart the hard way, to include a linear regression
             NumberAxis xAxis = new NumberAxis(Utility.shortenFunctionName((String)(normalData.getEventNames().get(i))));
             xAxis.setAutoRangeIncludesZero(false);
             NumberAxis yAxis = null;
             if (correlateToMain)
                     yAxis = new NumberAxis(normalData.getMainEventName());
             else
                     yAxis = new NumberAxis(Utility.shortenFunctionName((String)(normalData.getEventNames().get(j))));
             yAxis.setAutoRangeIncludesZero(false);
             StandardXYItemRenderer dotRenderer = new StandardXYItemRenderer(StandardXYItemRenderer.SHAPES);
             dotRenderer.setBaseShapesFilled(true); //TODO: setSeriesShapesFilled? setBaseShapesVisible?
             if (correlateToMain)
                     dotRenderer.setSeriesPaint(0,Color.green);
             XYPlot plot = new XYPlot(data, xAxis, yAxis, dotRenderer);

             // linear regression
             double[] coefficients = Regression.getOLSRegression(data, 0);
             Function2D curve = new LineFunction2D(coefficients[0], coefficients[1]);
             Range range = DatasetUtilities.findDomainBounds(data);
             // if someone does something silly, you may get a null range back.
             // if so, don't try to draw the regression line.
             if (range != null) {
	             XYDataset regressionData = DatasetUtilities.sampleFunction2D(
	                     curve, range.getLowerBound(), range.getUpperBound(), 
	                     100, "Fitted Linear Regression Line");
	             plot.setDataset(1, regressionData);
	             StandardXYItemRenderer lineRenderer = new StandardXYItemRenderer();
	             lineRenderer.setSeriesPaint(0,Color.blue);
	             lineRenderer.setBaseShapesVisible(false);
	             plot.setRenderer(1, lineRenderer);
             }

/* This is never visible, anyway - might as well remove it.
             // power regression
             double[] powerCoefficients = Regression.getPowerRegression(data, 0);
             Function2D powerCurve = new PowerFunction2D(powerCoefficients[0], powerCoefficients[1]);
             XYDataset powerRegressionData = DatasetUtilities.sampleFunction2D(
                 powerCurve, range.getLowerBound(), range.getUpperBound(), 
                 100, "Fitted Power Regression Line");
             plot.setDataset(2, powerRegressionData);
             StandardXYItemRenderer powerLineRenderer = new StandardXYItemRenderer();
             powerLineRenderer.setSeriesPaint(0,Color.black);
             powerLineRenderer.setBaseShapesVisible(true);
             plot.setRenderer(2, powerLineRenderer);
*/
             if (range != null) {
	             plot.getDomainAxis().setRange(range);
	             plot.getRangeAxis().setRange(range);
             }

             JFreeChart chart = new JFreeChart("Correlation Results: r = " + 
                 rCorrelation, JFreeChart.DEFAULT_TITLE_FONT, plot, true);
             
             Utility.applyDefaultChartTheme(chart);
     
             outfile = new File(Constants.TMPDIR + "image." + modelData.toShortString() + ".png");
             try {
                 ChartUtilities.saveChartAsPNG(outfile, chart, 500, 500);
             } catch (IOException e) {
             	System.err.println(e.getMessage());
 				e.printStackTrace();
 			}
         }
     return outfile;
     }

}
