package edu.uoregon.tau.perfexplorer.common;

import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;

import org.jfree.chart.axis.CategoryAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.entity.EntityCollection;
import org.jfree.chart.labels.CategoryItemLabelGenerator;
import org.jfree.chart.labels.ItemLabelAnchor;
import org.jfree.chart.labels.ItemLabelPosition;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.renderer.category.CategoryItemRendererState;
import org.jfree.chart.renderer.category.StackedBarRenderer;
import org.jfree.data.DataUtilities;
import org.jfree.data.category.CategoryDataset;
import org.jfree.ui.RectangleEdge;
import org.jfree.ui.TextAnchor;

public class AlignedStackedBarRenderer extends StackedBarRenderer{
	
    /** A flag that controls whether the bars display values or percentages. */
    private boolean renderAsPercentages;
    
    /**
     * Creates a new renderer.  By default, the renderer has no tool tip
     * generator and no URL generator.  These defaults have been chosen to
     * minimise the processing required to generate a default chart.  If you
     * require tool tips or URLs, then you can easily add the required
     * generators.
     */
    public AlignedStackedBarRenderer() {
        this(false);
    }

    /**
     * Creates a new renderer.
     *
     * @param renderAsPercentages  a flag that controls whether the data values
     *                             are rendered as percentages.
     */
    public AlignedStackedBarRenderer(boolean renderAsPercentages) {
        super();
        this.renderAsPercentages = false;//renderAsPercentages;

        // set the default item label positions, which will only be used if
        // the user requests visible item labels...
        ItemLabelPosition p = new ItemLabelPosition(ItemLabelAnchor.CENTER,
                TextAnchor.CENTER);
        setBasePositiveItemLabelPosition(p);
        setBaseNegativeItemLabelPosition(p);
        setPositiveItemLabelPositionFallback(null);
        setNegativeItemLabelPositionFallback(null);
    }
	
	/**
     * Tests this renderer for equality with an arbitrary object.
     *
     * @param obj  the object (<code>null</code> permitted).
     *
     * @return A boolean.
     */
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (!(obj instanceof AlignedStackedBarRenderer)) {
            return false;
        }
        AlignedStackedBarRenderer that = (AlignedStackedBarRenderer) obj;
        if (this.renderAsPercentages != that.renderAsPercentages) {
            return false;
        }
        return super.equals(obj);
    }
 
    
    private boolean didAlign = false;
    
    private double[] rowBases;
    
    private void initAlignment(CategoryDataset dataset){
    	
    	int cols=dataset.getColumnCount();
    	int rows=dataset.getRowCount();
    	
    	double[] rowOffsets = new double[rows];
    	double max;
    	Number tmp;
    	double comp;
    	for(int i=0; i<rows;i++){
    		max=0;
    		for(int j=0;j<cols;j++){
    			tmp=dataset.getValue(i, j);
    			if(tmp==null)
    			{
    				comp=0;
    			}
    			else{
    				comp=tmp.doubleValue();
    			}
    			max = Math.max(max, comp);
    		}
    		rowOffsets[i]=max*.7;
    	}
    	
    	rowBases=new double[rows];
    	double prev=0;
    	for(int i=0;i<rows;i++){
    		if(i==0){
    			prev=0;
    		}
    		else{
    			prev=rowBases[i-1];
    		}
    		rowBases[i]=rowOffsets[i]+prev;
    	}
    	
    	didAlign=true;
    }
    
    
    /**
     * Draws a stacked bar for a specific item.
     *
     * @param g2  the graphics device.
     * @param state  the renderer state.
     * @param dataArea  the plot area.
     * @param plot  the plot.
     * @param domainAxis  the domain (category) axis.
     * @param rangeAxis  the range (value) axis.
     * @param dataset  the data.
     * @param row  the row index (zero-based).
     * @param column  the column index (zero-based).
     * @param pass  the pass index.
     */
    public void drawItem(Graphics2D g2,
                         CategoryItemRendererState state,
                         Rectangle2D dataArea,
                         CategoryPlot plot,
                         CategoryAxis domainAxis,
                         ValueAxis rangeAxis,
                         CategoryDataset dataset,
                         int row,
                         int column,
                         int pass) {

    	if(!didAlign){
    		initAlignment(dataset);
    	}
    	
        // nothing is drawn for null values...
        Number dataValue = dataset.getValue(row, column);
        if (dataValue == null) {
            return;
        }

        double value = dataValue.doubleValue()*.7;

        PlotOrientation orientation = plot.getOrientation();
        double barW0 = domainAxis.getCategoryMiddle(column, getColumnCount(),
                dataArea, plot.getDomainAxisEdge())
                - state.getBarWidth() / 2.0;

//        double accPositiveBase = getBase();
//        double accNegativeBase = accPositiveBase;
        double positiveBase=getBase();
        double negativeBase=positiveBase;
//        double d =0;
//        
//        for(int j=0;j<dataset.getColumnCount();j++){
//        	for (int i = 0; i < row; i++) {
//            Number v = dataset.getValue(i, j);
//            if (v != null) {
//                d = v.doubleValue()*.8;
//               
//                if (d > 0) {
//                    accPositiveBase = accPositiveBase + d+j*2;
//                }
//                else {
//                    accNegativeBase = accNegativeBase + d+j*2;
//                }
//            }
//        	}
//        	if(d>0){
//        		positiveBase=Math.max(positiveBase, accPositiveBase);
//        	}
//        	else{
//        		negativeBase=Math.min(negativeBase, accNegativeBase);
//        	}
//        	accPositiveBase=getBase();
//        	accNegativeBase=getBase();
//        }

        double rowBase=0;
        if(row>0)
        {
        	rowBase=rowBases[row-1];
        }
        
        positiveBase=Math.max(getBase(),rowBase);
        negativeBase=Math.min(getBase(),rowBase);
        
        double translatedBase;
        double translatedValue;
        boolean positive = (value > 0.0);
        boolean inverted = rangeAxis.isInverted();
        RectangleEdge barBase;
        if (orientation == PlotOrientation.HORIZONTAL) {
            if (positive && inverted || !positive && !inverted) {
                barBase = RectangleEdge.RIGHT;
            }
            else {
                barBase = RectangleEdge.LEFT;
            }
        }
        else {
            if (positive && !inverted || !positive && inverted) {
                barBase = RectangleEdge.BOTTOM;
            }
            else {
                barBase = RectangleEdge.TOP;
            }
        }

        RectangleEdge location = plot.getRangeAxisEdge();
        if (positive) {
            translatedBase = rangeAxis.valueToJava2D(positiveBase, dataArea,
                    location);
            translatedValue = rangeAxis.valueToJava2D(positiveBase + value,
                    dataArea, location);
        }
        else {
            translatedBase = rangeAxis.valueToJava2D(negativeBase, dataArea,
                    location);
            translatedValue = rangeAxis.valueToJava2D(negativeBase + value,
                    dataArea, location);
        }
        double barL0 = Math.min(translatedBase, translatedValue);
        double barLength = Math.max(Math.abs(translatedValue - translatedBase),
                getMinimumBarLength());
        
//        if(column==dataset.getColumnCount()-1&&row==dataset.getRowCount()-1&&barLength+barL0>dataArea.getWidth()){
//        	dataArea.setRect(dataArea.getX(), dataArea.getY(), barLength+barL0+20, dataArea.getHeight());
//        }
        
        Rectangle2D bar = null;
        
        if (orientation == PlotOrientation.HORIZONTAL) {
            bar = new Rectangle2D.Double(barL0, barW0, barLength,
                    state.getBarWidth());
        }
        else {
            bar = new Rectangle2D.Double(barW0, barL0, state.getBarWidth(),
                    barLength);
        }
        if (pass == 0) {
            if (getShadowsVisible()) {
                boolean pegToBase = (positive && (positiveBase == getBase()))
                        || (!positive && (negativeBase == getBase()));
                getBarPainter().paintBarShadow(g2, this, row, column, bar,
                        barBase, pegToBase);
            }
        }
        else if (pass == 1) {
            getBarPainter().paintBar(g2, this, row, column, bar, barBase);

            // add an item entity, if this information is being collected
            EntityCollection entities = state.getEntityCollection();
            if (entities != null) {
                addItemEntity(entities, dataset, row, column, bar);
            }
        }
        else if (pass == 2) {
            CategoryItemLabelGenerator generator = getItemLabelGenerator(row,
                    column);
            if (generator != null && isItemLabelVisible(row, column)) {
                drawItemLabel(g2, dataset, row, column, plot, generator, bar,
                        (value < 0.0));
            }
        }
    }
}
