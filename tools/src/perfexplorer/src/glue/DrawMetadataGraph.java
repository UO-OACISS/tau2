package edu.uoregon.tau.perfexplorer.glue;

import java.awt.BasicStroke;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.StringTokenizer;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.CategoryLabelPositions;
import org.jfree.chart.axis.LogarithmicAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;

import com.sun.tools.javac.code.Attribute.Array;

import edu.uoregon.tau.common.VectorExport;
import edu.uoregon.tau.perfdmf.Trial;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfexplorer.client.MyCategoryAxis;
import edu.uoregon.tau.perfexplorer.client.PerfExplorerChart;
import edu.uoregon.tau.perfexplorer.constants.Constants;
import edu.uoregon.tau.perfexplorer.server.TauNamespaceContext;

/**
 * <p>
 * The DrawGraph class is a PerfExplorer Operation class for drawing a line
 * graph from an analysis script.  The creation of the graph is fairly
 * straightforward.  One or more input {@link PerformanceResult} objects
 * are used as the input for the constructor.  After the constructor,
 * various options for the graph are set.  The fields which are commonly
 * set are the series type ({@link #setSeriesType}), the category type
 * ({@link #setCategoryType}), and the value type ({@link #setValueType}).
 * Once the options are set, the {@link #processData} method is called
 * to generate the graph.
 * </p>
 *
 * <p>
 * This class has undefined behavior when running PerfExplorer without
 * the GUI.
 * </p>
 *
 * <p>
 * Example code from Python script:
 * </p>
 * <pre>
from edu.uoregon.tau.perfexplorer.glue import *
from edu.uoregon.tau.perfdmf import *
from java.util import *
from java.lang import *
from java.util import HashSet
from java.util import ArrayList
from edu.uoregon.tau.perfdmf import Trial
from edu.uoregon.tau.perfdmf import Metric
from edu.uoregon.tau.perfexplorer.glue import PerformanceResult
from edu.uoregon.tau.perfexplorer.glue import PerformanceAnalysisOperation
from edu.uoregon.tau.perfexplorer.glue import ExtractEventOperation
from edu.uoregon.tau.perfexplorer.glue import Utilities
from edu.uoregon.tau.perfexplorer.glue import BasicStatisticsOperation
from edu.uoregon.tau.perfexplorer.glue import DeriveMetricOperation
from edu.uoregon.tau.perfexplorer.glue import ScaleMetricOperation
from edu.uoregon.tau.perfexplorer.glue import DeriveMetricEquation
from edu.uoregon.tau.perfexplorer.glue import DeriveMetricsFileOperation
from edu.uoregon.tau.perfexplorer.glue import MergeTrialsOperation
from edu.uoregon.tau.perfexplorer.glue import TrialResult
from edu.uoregon.tau.perfexplorer.glue import AbstractResult
from edu.uoregon.tau.perfexplorer.glue import DrawGraph
from edu.uoregon.tau.perfexplorer.glue import DrawMetadataGraph
from edu.uoregon.tau.perfexplorer.glue import SaveResultOperation

True = 1
False = 0


def load(inApp, inExp, inTrial):
  trial1 = Utilities.getTrial(inApp, inExp, inTrial)
  result1 = TrialResult(trial1)
  return result1


def main():
        print "--------------- JPython test script start ------------"
        inputs = load("Application","Experiment","Trial")

        grapher = DrawMetadataGraph(inputs)
        grapher.setMetadataField("cluster-membership")
        #grapher.setTitle("My Title")
        #grapher.setXAxisLabel("")
        #grapher.setYAxisLabel("")
        grapher.processData()

        print "---------------- JPython test script end -------------"

if __name__ == "__main__":
    main()





 * </pre>
 *
 * <P>CVS $Id: DrawGraph.java,v 1.14 2009/04/09 00:23:51 khuck Exp $</P>
 * @author khuck
 * @version 0.2
 * @since   0.2
 *
 */
public class DrawMetadataGraph extends AbstractPerformanceOperation {

    private static final long serialVersionUID = -5587605162968129610L;
    protected String title = "My Chart";
    protected String yAxisLabel = "Threads in cluster";
    protected String xAxisLabel = "Cluster Number";
    protected String metadataField = "cluster-membership";
    protected PerfExplorerChart chartWindow = null;
    protected ArrayList<Double> metadatavalues = new ArrayList<Double>();

    /**
     * Creates a graph drawing operator.
     *
     * @param input
     */
    public DrawMetadataGraph(PerformanceResult input) {
	super(input);
	// TODO Auto-generated constructor stub
    }

    /**
     * Creates a graph drawing operator.
     *
     * @param trial
     */
    public DrawMetadataGraph(Trial trial) {
	super(trial);
	// TODO Auto-generated constructor stub
    }

    /**
     * Creates a graph drawing operator.
     *
     * @param inputs
     */
    public DrawMetadataGraph(List<PerformanceResult> inputs) {
	super(inputs);
	// TODO Auto-generated constructor stub
    }

    /* (non-Javadoc)
     * @see glue.PerformanceAnalysisOperation#processData()
     */
    public List<PerformanceResult> processData() {

	DefaultCategoryDataset dataset = new DefaultCategoryDataset();

	for (PerformanceResult input : inputs) {

	  
	    metadatavalues = getclusterData(input.getTrial());
	    for(Double c: metadatavalues){
		dataset.addValue(c, "Threads", metadatavalues.indexOf(c)+"");
	    }


	    JFreeChart chart = ChartFactory.createStackedBarChart(
		    this.title,  // chart title
		    this.xAxisLabel,  // domain Axis label
		    this.yAxisLabel,  // range Axis label
		    dataset,                         // data
		    PlotOrientation.HORIZONTAL,        // the plot orientation
		    true,                            // legend
		    true,                            // tooltips
		    false                            // urls
	    );
	    Utility.applyDefaultChartTheme(chart);
	    chart.removeLegend();
	    this.chartWindow = new PerfExplorerChart(chart, "General Chart");
	}
	return null;

    }
private ArrayList<Double> getclusterData(Trial trial){
    try{
	// build a factory to build the document builder
	DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
	DocumentBuilder builder = factory.newDocumentBuilder();
	
	/// read the XML
	Reader reader = new StringReader(trial.getField(Trial.XML_METADATA));
	InputSource source = new InputSource(reader);
	Document metadata = builder.parse(source);

	/* this is the 1.5 way */
	// build the xpath object to jump around in that document
	javax.xml.xpath.XPath xpath = javax.xml.xpath.XPathFactory.newInstance().newXPath();
	xpath.setNamespaceContext(new TauNamespaceContext());

	// get the profile attributes from the metadata
	NodeList profileAttributes = (NodeList) xpath.evaluate("/metadata/ProfileAttributes", metadata, javax.xml.xpath.XPathConstants.NODESET);

	// iterate through the "uncommon" profile attributes (different for each thread)
	for (int i = 0 ; i < profileAttributes.getLength() ; i++) {
		NodeList children = profileAttributes.item(i).getChildNodes();
		for (int j = 0 ; j < children.getLength(); j++) {
			Node attributeElement = children.item(j);
			Node name = attributeElement.getFirstChild();
			while (name.getFirstChild() == null || name.getFirstChild().getNodeValue() == null) {
				name = name.getNextSibling();
			}
			Node value = name.getNextSibling();
			while (value != null && (value.getFirstChild() == null || value.getFirstChild().getNodeValue() == null)) {
				value = value.getNextSibling();
			}
			if (value == null) { // if there is no value
			} else {
				String tmp = value.getFirstChild().getNodeValue();
				String tmpName = name.getFirstChild().getNodeValue();
				if (tmp != null && tmpName != null && !tmpName.equals("pid") && !tmpName.toLowerCase().contains("time")) {
					try {
						Double tmpDouble = Double.parseDouble(tmp);
						metadatavalues.add(tmpDouble);
					} catch (NumberFormatException e) { 
						//commonAttributes.put(tmpName, tmp);
					}
				}
			}
		}

	}
	return metadatavalues;
	} catch (Exception e) {
		System.err.println(e.getMessage());
		e.printStackTrace();
	}
    return metadatavalues;
	
	
}

    /**
     * Set the title for the graph.
     * @param title The title of the graph
     */
    public void setTitle(String title) {
	this.title = title;
    }

    /**
     * Set the label used for the X Axis.
     * @param xAxisLabel The label used for the X Axis
     */
    public void setXAxisLabel(String xAxisLabel) {
	this.xAxisLabel = xAxisLabel;
    }

    /**
     * Set the label used for the Y Axis.
     * @param yAxisLabel The label used for the Y Axis
     */
    public void setYAxisLabel(String yAxisLabel) {
	this.yAxisLabel = yAxisLabel;
    }



    /**
     * The metadata field to use for either the series name or the category name.
     * This method only has meaning if the series type or category type have been
     * set to {@link #METADATA}.
     * @param metadataField The metadata field to use for either the series name or
     * the category name.
     * @see #setSeriesType
     * @see #setCategoryType
     * @see #METADATA
     */
    public void setMetadataField(String metadataField) {
	this.metadataField = metadataField;
    }


    /**
     * Draws the graph to the file name specified.
     * @param fileName The filename for the graph output.
     */
    public void drawChartToFile(String fileName) {
	try {
	    VectorExport.export(chartWindow, new File(fileName), true, "PerfExplorer", true, true);
	} catch (Exception e) {
	    System.err.println("Could not write graph to file:");
	    System.err.println(e.getMessage());
	    e.printStackTrace();
	}
    }

}
