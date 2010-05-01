package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import javax.swing.*;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.xy.XYBarRenderer;
import org.jfree.data.statistics.SimpleHistogramBin;
import org.jfree.data.statistics.SimpleHistogramDataset;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.DataSourceException;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Thread;

public class PreCompHistogramWindow extends JFrame {
    private ParaProfTrial ppTrial;
    private Function function;
    private Thread thread;

    private ChartPanel chartPanel;

    private int numBins;
    private int numHistogramsPerEvent;

    public static class ScrollFlowLayout extends FlowLayout {
        public final static int LEFT_TO_RIGHT = 0;
        public final static int RIGHT_TO_LEFT = 1;
        public final static int TOP_TO_BOTTOM = 2;

        private int orientation = LEFT_TO_RIGHT, rows, cols;
        private ScrollPane sp = null;

        public ScrollFlowLayout() {
            super();
        }

        public ScrollFlowLayout(ScrollPane p) {
            super();
            this.sp = p;
        }

        public ScrollFlowLayout(ScrollPane p, int orientation) {
            super();
            this.sp = p;
            this.orientation = orientation;
        }

        public int getTallestElement(Container c) {
            int ncomponents = c.getComponentCount();
            int h = 0;
            for (int i = 0; i < ncomponents; i++) {
                Component m = c.getComponent(i);
                int x = m.getMinimumSize().height;
                h = (h > x) ? h : x;
            }
            return h;
        }

        public int getWidestElement(Container c) {
            int ncomponents = c.getComponentCount();
            int w = 0;
            for (int i = 0; i < ncomponents; i++) {
                Component m = c.getComponent(i);
                int x = m.getMinimumSize().width;
                w = (w > x) ? w : x;
            }
            return w;
        }

        public Dimension preferredLayoutSize(Container target) {
            int hBounds = target.getSize().height - target.getInsets().bottom - target.getInsets().top;

            int wBounds = target.getSize().width;
            if (sp != null) {
                hBounds = sp.getSize().height - sp.getInsets().bottom - sp.getInsets().top;

                wBounds = sp.getSize().width;
            }
            int ncomponents = target.getComponentCount();
            if (ncomponents == 0)
                return new Dimension(0, 0);
            int widest = getWidestElement(target);
            int tallest = getTallestElement(target);

            int rows = (tallest > hBounds) ? 1 : hBounds / tallest;
            int cols = (int) Math.ceil((double) ncomponents / rows);
            if (orientation == LEFT_TO_RIGHT) {
                cols = (widest > wBounds) ? 1 : wBounds / widest;
                rows = (int) Math.ceil((double) ncomponents / cols);
            }

            Dimension d = new Dimension(cols * widest, rows * tallest);
            return d;
        }

        public void layoutContainer(Container target) {
            synchronized (target.getTreeLock()) {
                int hBounds = target.getSize().height - target.getInsets().bottom - target.getInsets().top;
                int wBounds = target.getSize().width;
                if (sp != null) {
                    hBounds = sp.getSize().height - sp.getInsets().bottom - sp.getInsets().top;

                    wBounds = sp.getSize().width;
                }
                int ncomponents = target.getComponentCount();
                int widest = getWidestElement(target);
                int tallest = getTallestElement(target);
                if (ncomponents == 0) {
                    return;
                }
                target.invalidate();
                if (orientation == LEFT_TO_RIGHT) {
                    int i = 0;
                    int cols = (widest > wBounds) ? 1 : wBounds / widest;
                    int rows = (int) Math.ceil((double) ncomponents / cols);

                    for (int x = 0; x < rows; x++) {
                        for (int y = 0; y < cols; y++) {
                            if (i < ncomponents) {
                                int px = y * widest;
                                int py = x * tallest;
                                target.getComponent(i).setBounds(px, py, widest, tallest);
                            }
                            i++;
                        }
                    }
                } else if (orientation == TOP_TO_BOTTOM) {
                    int rows = (tallest > hBounds) ? 1 : hBounds / tallest;
                    int cols = (int) Math.ceil((double) ncomponents / rows);
                    int i = 0;
                    for (int y = 0; y < cols; y++) {
                        for (int x = 0; x < rows; x++) {
                            if (i < ncomponents) {
                                int px = y * widest;
                                int py = x * tallest;
                                target.getComponent(i).setBounds(px, py, widest, tallest);
                            }
                            i++;
                        }
                    }
                }
            }
        }
    }

    /**
     * A modified version of FlowLayout that allows containers using this
     * Layout to behave in a reasonable manner when placed inside a
     * JScrollPane
       
     * @author Babu Kalakrishnan
     */
    public static class ModifiedFlowLayout extends FlowLayout {
        public ModifiedFlowLayout() {
            super();
        }

        public ModifiedFlowLayout(int align) {
            super(align);
        }

        public ModifiedFlowLayout(int align, int hgap, int vgap) {
            super(align, hgap, vgap);
        }

        public Dimension minimumLayoutSize(Container target) {
            return computeSize(target, false);
        }

        public Dimension preferredLayoutSize(Container target) {
            return computeSize(target, true);
        }

        private Dimension computeSize(Container target, boolean minimum) {
            synchronized (target.getTreeLock()) {
                int hgap = getHgap();
                int vgap = getVgap();
                int w = target.getWidth();

                // Let this behave like a regular FlowLayout (single row)
                // if the container hasn't been assigned any size yet   
                if (w == 0) {
                    w = Integer.MAX_VALUE;
                }

                Insets insets = target.getInsets();
                if (insets == null) {
                    insets = new Insets(0, 0, 0, 0);
                }
                int reqdWidth = 0;

                int maxwidth = w - (insets.left + insets.right + hgap * 2);
                int n = target.getComponentCount();
                int x = 0;
                int y = insets.top;
                int rowHeight = 0;

                for (int i = 0; i < n; i++) {
                    Component c = target.getComponent(i);
                    if (c.isVisible()) {
                        Dimension d = minimum ? c.getMinimumSize() : c.getPreferredSize();
                        if ((x == 0) || ((x + d.width) <= maxwidth)) {
                            if (x > 0) {
                                x += hgap;
                            }
                            x += d.width;
                            rowHeight = Math.max(rowHeight, d.height);
                        } else {
                            x = d.width;
                            y += vgap + rowHeight;
                            rowHeight = d.height;
                        }
                        reqdWidth = Math.max(reqdWidth, x);
                    }
                }
                y += rowHeight;
                return new Dimension(reqdWidth + insets.left + insets.right, y);
            }
        }
    }

    private static class Histogram {
        public String name;
        public int bins[];
        public double minValue, maxValue;
    }

    private List<Histogram> histograms = new ArrayList<Histogram>();

    public static PreCompHistogramWindow createHistogramWindow(ParaProfTrial ppTrial, Function function, Thread thread,
            Component invoker) {
        try {
            PreCompHistogramWindow pchw = new PreCompHistogramWindow(ppTrial, function, thread, invoker);
            return pchw;
        } catch (Exception e) {
            throw new DataSourceException(e);
        }
    }

    private void processData() throws FileNotFoundException, IOException {
        int invocationIndex = thread.getNodeID();
        String histogramFileName = "tau.histograms." + invocationIndex;

        FileInputStream fis = new FileInputStream(new File(histogramFileName));
        InputStreamReader inReader = new InputStreamReader(fis);
        BufferedReader br = new BufferedReader(inReader);

        int numEvents = Integer.parseInt(br.readLine());
        numHistogramsPerEvent = Integer.parseInt(br.readLine());
        numBins = Integer.parseInt(br.readLine());

        for (int i = 0; i < numHistogramsPerEvent; i++) {
            Histogram histogram = new Histogram();
            histogram.name = br.readLine();
            histogram.bins = new int[numBins];
            histograms.add(histogram);
        }

        boolean found = false;
        for (int e = 0; e < numEvents; e++) {
            String eventName = br.readLine().trim();
            if (eventName.equals(function.getName())) {
                found = true;
                for (int i = 0; i < numHistogramsPerEvent; i++) {
                    StringTokenizer tokenizer = new StringTokenizer(br.readLine(), " \t\n\r");
                    histograms.get(i).minValue = Double.parseDouble(tokenizer.nextToken());
                    histograms.get(i).maxValue = Double.parseDouble(tokenizer.nextToken());
                    for (int b = 0; b < numBins; b++) {
                        histograms.get(i).bins[b] = Integer.parseInt(tokenizer.nextToken());
                    }
                }
            } else {
                // skip
                for (int i = 0; i < numHistogramsPerEvent; i++) {
                    br.readLine();
                }
            }
        }
        if (!found) {
            System.err.println("Warning, Function \"" + function.getName() + "\" not found in histogram");
        }

    }

    private void addHistograms() {
        Container contentPane = this.getContentPane();
        //contentPane.setLayout(new FlowLayout());

        JPanel panel = new JPanel();
        panel.setLayout(new ScrollFlowLayout());

        for (Histogram histogram : histograms) {
            // Histogram histogram = histograms.get(0);
            SimpleHistogramDataset dataset = new SimpleHistogramDataset("foo");
            double binWidth = (histogram.maxValue - histogram.minValue) / numBins;

            boolean timeMetric = false;
            double divisor = 1;
            if (histogram.name.contains("TIME")) {
                timeMetric = true;
                histogram.name = histogram.name + " (seconds)";
                divisor = 1e6;
            }
            System.err.println("-------------");
            for (int i = 0; i < numBins; i++) {
                double lowerBound = histogram.minValue + (binWidth * i);
                double upperBound = histogram.minValue + (binWidth * (i + 1));
                lowerBound /= divisor;
                upperBound /= divisor;
                System.out.println("lowerBound = " + lowerBound + ", upperBound = " + upperBound);
                SimpleHistogramBin bin = new SimpleHistogramBin(lowerBound, upperBound, false, false);
                bin.setItemCount(histogram.bins[i]);
                dataset.addBin(bin);
            }

            dataset.setAdjustForBinSize(false);
            String xAxis = histogram.name;

            JFreeChart chart = ChartFactory.createHistogram(function.getName(), xAxis, "Count", dataset,
                    PlotOrientation.VERTICAL, false, // legend
                    true, // tooltips
                    false); // urls

            chart.getXYPlot().getDomainAxis().setUpperBound(histogram.maxValue / divisor);
            chart.getXYPlot().getDomainAxis().setLowerBound(histogram.minValue / divisor);

            //        dataset.addSeries(function.getName() + " : " + histogram.name, values, numBins, minValue, maxValue);

            ((XYBarRenderer) chart.getXYPlot().getRenderer()).setMargin(0.10);
            ((XYBarRenderer) chart.getXYPlot().getRenderer()).setOutlinePaint(Color.black);

            Utility.applyDefaultChartTheme(chart);

            chartPanel = new ChartPanel(chart);
//            chartPanel.setPreferredSize(new java.awt.Dimension(300, 300)); 
            System.out.println("size = " + chartPanel.getPreferredSize());
            chartPanel.setMinimumSize(chartPanel.getPreferredSize());
            
            panel.add(chartPanel);

        }

        ScrollPane scrollPane = new ScrollPane();
        scrollPane.add(panel);
//        JScrollPane scrollPane = new JScrollPane(panel, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
  //              JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);

        panel.setLayout(new ScrollFlowLayout(scrollPane));
        //scrollPane.add(panel);
        setContentPane(scrollPane);
        //
        //        getContentPane().setLayout(new GridBagLayout());
        //        GridBagConstraints gbc = new GridBagConstraints();
        //        gbc.insets = new Insets(5, 5, 5, 5);
        //        gbc.fill = GridBagConstraints.BOTH;
        //        gbc.anchor = GridBagConstraints.CENTER;
        //        gbc.weightx = 0.95;
        //        gbc.weighty = 0.98;
        //        Utility.addCompItem(contentPane, scrollPane, gbc, 0, 0, 1, 1);
        //        Utility.addCompItem(contentPane, new JButton("foobar"), gbc, 1, 0, 1, 1);
        //
        //        this.pack();
    }

    private PreCompHistogramWindow(ParaProfTrial ppTrial, Function function, Thread thread, Component invoker) throws Exception {
        this.ppTrial = ppTrial;
        this.function = function;
        this.thread = thread;

        setTitle("TAU: ParaProf: Histogram: " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));
        ParaProfUtils.setFrameIcon(this);
        setSize(ParaProfUtils.checkSize(new java.awt.Dimension(710, 460)));
        setLocation(WindowPlacer.getNewLocation(this, invoker));

        processData();
        addHistograms();
    }

}
