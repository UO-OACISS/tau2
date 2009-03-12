package edu.uoregon.tau.paraprof;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

import javax.swing.JFrame;

import org.jfree.chart.*;
import org.jfree.chart.axis.*;
import org.jfree.chart.labels.StandardXYItemLabelGenerator;
import org.jfree.chart.labels.XYItemLabelGenerator;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.paraprof.enums.ValueType;

public class RegressionGraph {
    public int metric = 0;
    public double unitMultiple = 0.000001;
    public double percent = 0.02;
    public ValueType valueType = ValueType.EXCLUSIVE;

    public String unitsString = "Seconds";
    public String xaxisLabel = "UEDGE Test Cases";
    public String yaxisLabel = "Time (in Seconds)";
    public String title = "Time spent in UEDGE test cases";
    public boolean barChart = true;
    public boolean mainOnly = true;
    public boolean horizontal = false;
    public boolean legend = true;
    public boolean angledXaxis = true;
    public boolean singleTrial = false;
    public boolean useProcCountAsTrialName = false;

    public int stringLimit = 60;
    public boolean logScale = false;

    public boolean scalingChart = false;
    public boolean speedupChart = false;

    public Font titleFont = new Font("SansSerif", Font.BOLD, 16);
    public Font smallFont = new Font("SansSerif", Font.BOLD, 14);
    public Font tickFont = new Font("SansSerif", Font.PLAIN, 12);

    public List trials;
    public List exps;
    public List expnames;

    private RegressionGraph() {};

    public void setTrials(List trials) {
        this.trials = trials;
        this.exps = new ArrayList();
        this.exps.add(trials);
        this.expnames = new ArrayList();
        this.expnames.add("wallclock");
    }

    public static RegressionGraph createChart() {
        RegressionGraph chart = new RegressionGraph();
        return chart;
    }

    public static RegressionGraph createChart(ParaProfTrial trial) {
        List trials = new ArrayList();
        trials.add(trial);
        return createBasicChart(trials);
    }

    public static RegressionGraph createBasicChart(List trials) {
        RegressionGraph chart = new RegressionGraph();
        chart.trials = trials;
        chart.exps = new ArrayList();
        chart.exps.add(trials);
        chart.expnames = new ArrayList();
        chart.expnames.add("wallclock");
        return chart;
    }

    public static RegressionGraph createExperimentChart(List exps, List expnames) {
        RegressionGraph chart = new RegressionGraph();
        chart.exps = exps;
        chart.trials = (List) chart.exps.get(0);
        chart.expnames = expnames;
        return chart;
    }

    private PPFunctionProfile getTopLevelTimer(ParaProfTrial ppTrial) {
        DataSorter dataSorter = new DataSorter(ppTrial);
        dataSorter.setSortType(SortType.VALUE);
        dataSorter.setSortValueType(ValueType.INCLUSIVE);
        dataSorter.setValueType(ValueType.INCLUSIVE);
        List fps = dataSorter.getFunctionProfiles(ppTrial.getMeanThread());
        PPFunctionProfile topfp = (PPFunctionProfile) fps.get(0);
        return topfp;
    }

    private XYDataset getScalingDataSet() {
        XYSeriesCollection dataSet = new XYSeriesCollection();
        XYSeries idealSeries = new XYSeries("ideal");
        dataSet.addSeries(idealSeries);

        double maxProcCount = Double.MIN_VALUE;
        double minProcCount = Double.MAX_VALUE;
        for (int e = 0; e < exps.size(); e++) {
            List trials = (List) exps.get(e);
            XYSeries series = new XYSeries((String) expnames.get(e));

            double rootProcCount = Double.MAX_VALUE;
            double rootValue = Double.MAX_VALUE;

            if (speedupChart) {
                for (int i = 0; i < trials.size(); i++) {
                    ParaProfTrial ppTrial = (ParaProfTrial) trials.get(i);
                    PPFunctionProfile topfp = getTopLevelTimer(ppTrial);
                    int procCount = ppTrial.getMaxNCTNumbers()[0] + 1;
                    maxProcCount = Math.max(maxProcCount, procCount);
                    minProcCount = Math.min(minProcCount, procCount);

                    if (procCount < rootProcCount) {
                        rootProcCount = procCount;
                        rootValue = ValueType.INCLUSIVE.getValue(topfp.getFunctionProfile(), metric) * unitMultiple;
                    }
                }
            }
            for (int i = 0; i < trials.size(); i++) {
                ParaProfTrial ppTrial = (ParaProfTrial) trials.get(i);
                PPFunctionProfile topfp = getTopLevelTimer(ppTrial);
                double value = ValueType.INCLUSIVE.getValue(topfp.getFunctionProfile(), metric) * unitMultiple;
                int procCount = ppTrial.getMaxNCTNumbers()[0] + 1;
                if (speedupChart) {
                    value = rootValue / value;
                }
                series.add(procCount, value);
                //dataSet.addValue(value, (String)expnames.get(e), ppTrial.getName());
            }
            dataSet.addSeries(series);
        }

        if (speedupChart) {
            idealSeries.add(1.0, minProcCount);
            idealSeries.add(maxProcCount, maxProcCount);
        } else {
            dataSet.removeSeries(idealSeries);
        }

        return dataSet;
    }

    private static class NCTComparator implements Comparator {
        public int compare(Object arg0, Object arg1) {
            if (arg0 instanceof ParaProfTrial) {
                ParaProfTrial t1 = (ParaProfTrial) arg0;
                ParaProfTrial t2 = (ParaProfTrial) arg1;
                return t1.getMaxNCTNumbers()[0] - t2.getMaxNCTNumbers()[0];
            } else {
                List e1 = (List) arg0;
                List e2 = (List) arg1;
                ParaProfTrial t1 = (ParaProfTrial) e1.get(0);
                ParaProfTrial t2 = (ParaProfTrial) e2.get(0);
                return t1.getMaxNCTNumbers()[0] - t2.getMaxNCTNumbers()[0];
            }
        }
    }

    private CategoryDataset getMainOnlyDataSet() {

        DefaultCategoryDataset dataSet = new DefaultCategoryDataset();

        TreeMap map = new TreeMap();

        // So that the trials come in order
        for (int e = 0; e < exps.size(); e++) {
            List trials = (List) exps.get(e);
            for (int i = 0; i < trials.size(); i++) {
                ParaProfTrial ppTrial = (ParaProfTrial) trials.get(i);
                map.put(ppTrial.getName(), ppTrial.getName());
            }
        }
        for (Iterator it = map.keySet().iterator(); it.hasNext();) {
            String string = (String) it.next();
            dataSet.addValue(42, "@@@", string);
        }

        for (int e = 0; e < exps.size(); e++) {
            List trials = (List) exps.get(e);
            Collections.sort(trials, new NCTComparator());
            for (int i = 0; i < trials.size(); i++) {
                ParaProfTrial ppTrial = (ParaProfTrial) trials.get(i);
                DataSorter dataSorter = new DataSorter(ppTrial);
                dataSorter.setSortType(SortType.VALUE);
                dataSorter.setSortValueType(ValueType.INCLUSIVE);
                dataSorter.setValueType(ValueType.INCLUSIVE);
                List fps = dataSorter.getFunctionProfiles(ppTrial.getMeanThread());
                PPFunctionProfile topfp = (PPFunctionProfile) fps.get(0);
                double value = ValueType.INCLUSIVE.getValue(topfp.getFunctionProfile(), metric) * unitMultiple;
                dataSet.addValue(value, (String) expnames.get(e), ppTrial.getName());
            }
        }

        dataSet.removeRow("@@@");

        return dataSet;
    }

    private CategoryDataset getDataSet() {

        Map functionMap = new HashMap();

        int funcCount = 0;

        for (Iterator it = trials.iterator(); it.hasNext();) {
            ParaProfTrial ppTrial = (ParaProfTrial) it.next();
            DataSorter dataSorter = new DataSorter(ppTrial);
            dataSorter.setSortType(SortType.VALUE);
            dataSorter.setSortValueType(ValueType.INCLUSIVE);
            dataSorter.setValueType(ValueType.INCLUSIVE);

            List fps = dataSorter.getFunctionProfiles(ppTrial.getMeanThread());

            PPFunctionProfile topfp = getTopLevelTimer(ppTrial);
            double topValue = topfp.getInclusiveValue();
            double threshhold = topValue * percent;
            for (Iterator it2 = fps.iterator(); it2.hasNext();) {
                PPFunctionProfile fp = (PPFunctionProfile) it2.next();
                if (fp.getExclusiveValue() > threshhold) {
                    String displayName = fp.getDisplayName();
                    if (functionMap.get(displayName) == null) { // if not already in
                        functionMap.put(displayName, new Integer(funcCount++));
                    }
                }
            }
        }
        functionMap.put("other", new Integer(funcCount++));
        int otherIndex = ((Integer) functionMap.get("other")).intValue();

        int trialcount = trials.size();

        double[][] data = new double[funcCount][trialcount];

        for (int idx = 0; idx < trials.size(); idx++) {
            ParaProfTrial ppTrial = (ParaProfTrial) trials.get(idx);
            DataSorter dataSorter = new DataSorter(ppTrial);
            List fps = dataSorter.getFunctionProfiles(ppTrial.getMeanThread());
            for (Iterator it2 = fps.iterator(); it2.hasNext();) {
                PPFunctionProfile fp = (PPFunctionProfile) it2.next();
                String displayName = fp.getDisplayName();
                Integer integer = (Integer) functionMap.get(displayName);
                double value = valueType.getValue(fp.getFunctionProfile(), metric) * unitMultiple;
                if (integer != null) {
                    data[integer.intValue()][idx] = value;
                } else {
                    data[otherIndex][idx] += value;
                }
            }
        }

        String trialnames[] = new String[trialcount];
        for (int idx = 0; idx < trials.size(); idx++) {
            ParaProfTrial ppTrial = (ParaProfTrial) trials.get(idx);
            if (useProcCountAsTrialName) {
                trialnames[idx] = Integer.toString(ppTrial.getDataSource().getNumberOfNodes());
            } else {
                trialnames[idx] = ppTrial.getName();
            }
        }

        String funcnames[] = new String[funcCount];
        for (Iterator it = functionMap.keySet().iterator(); it.hasNext();) {
            String string = (String) it.next();
            Integer integer = (Integer) functionMap.get(string);
            if (string.length() > stringLimit) {
                string = string.substring(0, stringLimit) + "...";
            }
            funcnames[integer.intValue()] = string;
        }

        DefaultCategoryDataset dataSet = new DefaultCategoryDataset();
        for (int x = 0; x < funcCount; x++) {
            for (int y = 0; y < trialcount; y++) {
                if (data[x][y] > 0) {
                    dataSet.addValue(data[x][y], funcnames[x], trialnames[y]);
                }
            }
        }

        return dataSet;
    }

    private CategoryDataset getSingleDataSet() {

        ParaProfTrial ppTrial = (ParaProfTrial) trials.get(0);

        PPFunctionProfile topfp = getTopLevelTimer(ppTrial);
        double topValue = topfp.getInclusiveValue();
        double threshhold = topValue * percent;
        DefaultCategoryDataset dataSet = new DefaultCategoryDataset();

        DataSorter dataSorter = new DataSorter(ppTrial);
        dataSorter.setSortType(SortType.VALUE);
        dataSorter.setSortValueType(valueType);
        dataSorter.setValueType(valueType);

        List fps = dataSorter.getFunctionProfiles(ppTrial.getMeanThread());

        for (Iterator it2 = fps.iterator(); it2.hasNext();) {
            PPFunctionProfile fp = (PPFunctionProfile) it2.next();
            if (fp.getExclusiveValue() > threshhold) {
                String displayName = fp.getDisplayName();

                double value = valueType.getValue(fp.getFunctionProfile(), metric) * unitMultiple;

                dataSet.addValue(value, displayName, displayName);
            }
        }

        return dataSet;
    }

    public JFreeChart getChart() {

        ParaProf.preferences.setShowSourceLocation(false);
        //ParaProf.preferences.setShowSourceLocation(true);

        JFreeChart chart;

        CategoryDataset dataSet = null;
        XYDataset xyDataSet = null;
        if (singleTrial) {
            dataSet = getSingleDataSet();
        } else if (scalingChart) {
            xyDataSet = getScalingDataSet();
        } else if (mainOnly) {
            dataSet = getMainOnlyDataSet();
        } else {
            dataSet = getDataSet();
        }

        PlotOrientation orientation;
        if (horizontal) {
            orientation = PlotOrientation.HORIZONTAL;
        } else {
            orientation = PlotOrientation.VERTICAL;
        }

        if (scalingChart) {
            chart = ChartFactory.createXYLineChart(title, xaxisLabel, yaxisLabel, xyDataSet, orientation, legend, true, false);
            XYPlot plot = chart.getXYPlot();

            //XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
            //plot.setRenderer(renderer);
            XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer) plot.getRenderer();
            renderer.setBaseShapesVisible(true);

            renderer.setBaseShapesFilled(true);
            renderer.setBaseStroke(new BasicStroke(2f, BasicStroke.JOIN_ROUND, BasicStroke.JOIN_BEVEL));
            //TODO: This is probably unnecessary
            //StandardLegend legend = (StandardLegend) chart.getLegend();
            //legend.setDisplaySeriesShapes(true);

            if (speedupChart) {
                renderer.setSeriesShapesVisible(0, false);
            }

            XYItemLabelGenerator generator = new StandardXYItemLabelGenerator("{2}", new DecimalFormat("0.00"),
                    new DecimalFormat("0.00"));
            //            renderer.setBaseItemLabelGenerator(generator);
            //            renderer.setBaseItemLabelsVisible(true);

            renderer.setSeriesItemLabelGenerator(xyDataSet.getSeriesCount() - 1, generator);
            renderer.setSeriesItemLabelsVisible(xyDataSet.getSeriesCount() - 1, true);

            ValueAxis xAxis = (ValueAxis) plot.getDomainAxis();
            TickUnitSource units = NumberAxis.createIntegerTickUnits();
            xAxis.setStandardTickUnits(units);

            if (speedupChart) {
                // ideal
                renderer.setSeriesPaint(0, Color.gray);
            } else {
                renderer.setSeriesPaint(0, Color.red);
            }
            renderer.setSeriesPaint(1, Color.blue);
            renderer.setSeriesPaint(2, Color.green);
            renderer.setSeriesPaint(3, Color.magenta);
            renderer.setSeriesPaint(4, Color.cyan);
            renderer.setSeriesPaint(5, Color.pink);
            renderer.setSeriesPaint(6, Color.red);
            renderer.setSeriesPaint(7, Color.orange);
            renderer.setSeriesPaint(8, Color.lightGray);

        } else if (barChart) {
            chart = ChartFactory.createStackedBarChart(title, // title
                    xaxisLabel, // domain axis label
                    yaxisLabel, // range axis label
                    dataSet, // data
                    orientation, // the plot orientation
                    legend, // legend
                    true, // tooltips
                    false // urls
            );
        } else {
            chart = ChartFactory.createLineChart(title, // title
                    xaxisLabel, // domain axis label
                    yaxisLabel, // range axis label
                    dataSet, // data
                    orientation, // the plot orientation
                    legend, // legend
                    true, // tooltips
                    false // urls
            );

            CategoryPlot plot = chart.getCategoryPlot();
            LineAndShapeRenderer renderer = (LineAndShapeRenderer) plot.getRenderer();
            renderer.setBaseShapesVisible(true);
            renderer.setBaseShapesFilled(true);
            renderer.setBaseStroke(new BasicStroke(2f, BasicStroke.JOIN_ROUND, BasicStroke.JOIN_BEVEL));

            renderer.setSeriesPaint(0, Color.red);
            renderer.setSeriesPaint(1, Color.blue);
            renderer.setSeriesPaint(2, Color.green);
            renderer.setSeriesPaint(3, Color.magenta);
            renderer.setSeriesPaint(4, Color.cyan);
            renderer.setSeriesPaint(5, Color.pink);
            renderer.setSeriesPaint(6, Color.gray);
            renderer.setSeriesPaint(7, Color.orange);
            renderer.setSeriesPaint(8, Color.lightGray);

            renderer.setSeriesPaint(dataSet.getRowCount() - 1, Color.black);

        }

        if (angledXaxis && !scalingChart) {
            CategoryPlot plot = chart.getCategoryPlot();
            CategoryAxis xAxis = (CategoryAxis) plot.getDomainAxis();
            xAxis.setCategoryLabelPositions(CategoryLabelPositions.UP_45);
        }

        TextTitle textTitle = new TextTitle();
        textTitle.setText(title);
        textTitle.setFont(titleFont);
        chart.setTitle(textTitle);

        ValueAxis yAxis;
        if (scalingChart) {
            XYPlot plot = chart.getXYPlot();
            ValueAxis xAxis = (ValueAxis) plot.getDomainAxis();
            xAxis.setLabelFont(smallFont);
            xAxis.setTickLabelFont(tickFont);
            yAxis = (ValueAxis) plot.getRangeAxis();

        } else {
            CategoryPlot plot = chart.getCategoryPlot();
            CategoryAxis xAxis = (CategoryAxis) plot.getDomainAxis();
            xAxis.setLabelFont(smallFont);
            xAxis.setTickLabelFont(tickFont);
            yAxis = (ValueAxis) plot.getRangeAxis();

        }

        Utility.applyDefaultChartTheme(chart);

        if (logScale) {
            LogarithmicAxis logAxis = new LogarithmicAxis("");
            logAxis.setAllowNegativesFlag(true);
            logAxis.setLog10TickLabelsFlag(false);
            yAxis = logAxis;
            if (scalingChart) {
                XYPlot plot = chart.getXYPlot();
                plot.setRangeAxis(yAxis);
            } else {
                CategoryPlot plot = chart.getCategoryPlot();
                plot.setRangeAxis(yAxis);
            }
        }

        yAxis.setLabel(yaxisLabel);
        yAxis.setLabelFont(new Font("SansSerif", Font.BOLD, 14));
        yAxis.setTickLabelFont(tickFont);
        return chart;
    }

    public JFrame createFrame() {

        ChartPanel panel;
        panel = new ChartPanel(getChart());
        JFrame frame = new JFrame();
        frame.getContentPane().add(panel);
        frame.setSize(640, 480);
        ParaProfUtils.setFrameIcon(frame);

        return frame;
    }

    public void savePNG(String filename) {
        JFreeChart chart = getChart();
        try {
            ChartUtilities.saveChartAsPNG(new File(filename), chart, 640, 480);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int getMetric() {
        return metric;
    }

    public void setMetric(int metric) {
        this.metric = metric;
    }

    public double getUnitMultiple() {
        return unitMultiple;
    }

    public void setUnitMultiple(double unitMultiple) {
        this.unitMultiple = unitMultiple;
    }

    public double getPercent() {
        return percent;
    }

    public void setPercent(double percent) {
        this.percent = percent;
    }

    public ValueType getValueType() {
        return valueType;
    }

    public void setValueType(ValueType valueType) {
        this.valueType = valueType;
    }

    public String getUnitsString() {
        return unitsString;
    }

    public void setUnitsString(String unitsString) {
        this.unitsString = unitsString;
    }

    public String getXaxisLabel() {
        return xaxisLabel;
    }

    public void setXaxisLabel(String xaxisLabel) {
        this.xaxisLabel = xaxisLabel;
    }

    public String getYaxisLabel() {
        return yaxisLabel;
    }

    public void setYaxisLabel(String yaxisLabel) {
        this.yaxisLabel = yaxisLabel;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public boolean getBarChart() {
        return barChart;
    }

    public void setBarChart(boolean barChart) {
        this.barChart = barChart;
    }

    public boolean getMainOnly() {
        return mainOnly;
    }

    public void setMainOnly(boolean mainOnly) {
        this.mainOnly = mainOnly;
    }

    public boolean getHorizontal() {
        return horizontal;
    }

    public void setHorizontal(boolean horizontal) {
        this.horizontal = horizontal;
    }

    public boolean getLegend() {
        return legend;
    }

    public void setLegend(boolean legend) {
        this.legend = legend;
    }

    public boolean getAngledXaxis() {
        return angledXaxis;
    }

    public void setAngledXaxis(boolean angledXaxis) {
        this.angledXaxis = angledXaxis;
    }

    public int getStringLimit() {
        return stringLimit;
    }

    public void setStringLimit(int stringLimit) {
        this.stringLimit = stringLimit;
    }

    public boolean getLogScale() {
        return logScale;
    }

    public void setLogScale(boolean logScale) {
        this.logScale = logScale;
    }

    public boolean getScalingChart() {
        return scalingChart;
    }

    public void setScalingChart(boolean scalingChart) {
        this.scalingChart = scalingChart;
    }

    public boolean getSpeedupChart() {
        return speedupChart;
    }

    public void setSpeedupChart(boolean speedupChart) {
        this.speedupChart = speedupChart;
    }

    public Font getTitleFont() {
        return titleFont;
    }

    public void setTitleFont(Font titleFont) {
        this.titleFont = titleFont;
    }

    public Font getSmallFont() {
        return smallFont;
    }

    public void setSmallFont(Font smallFont) {
        this.smallFont = smallFont;
    }

    public Font getTickFont() {
        return tickFont;
    }

    public void setTickFont(Font tickFont) {
        this.tickFont = tickFont;
    }

    public void setExps(List exps, List expnames) {
        this.exps = exps;
        this.expnames = expnames;
        this.trials = (List) this.exps.get(0);
    }

    public void setSingleTrial(boolean singleTrial) {
        this.singleTrial = singleTrial;
    }

    public void setUseProcCountAsName(boolean useProcCountAsName) {
        this.useProcCountAsTrialName = useProcCountAsName;
    }

}
