/* ===========================================================
 * JFreeChart : a free chart library for the Java(tm) platform
 * ===========================================================
 *
 * (C) Copyright 2000-2004, by Object Refinery Limited and Contributors.
 *
 * Project Info:  http://www.jfree.org/jfreechart/index.html
 *
 * This library is free software; you can redistribute it and/or modify it under the terms
 * of the GNU Lesser General Public License as published by the Free Software Foundation;
 * either version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with this
 * library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * [Java is a trademark or registered trademark of Sun Microsystems, Inc. 
 * in the United States and other countries.]
 *
 * -----------------
 * MyCategoryAxis.java
 * -----------------
 * (C) Copyright 2000-2004, by Object Refinery Limited.
 *
 * Original Author:  David Gilbert;
 * Contributor(s):   -;
 *
 * $Id: MyCategoryAxis.java,v 1.2 2008/03/05 00:25:52 khuck Exp $
 *
 * Changes (from 21-Aug-2001)
 * --------------------------
 * 21-Aug-2001 : Added standard header. Fixed DOS encoding problem (DG);
 * 18-Sep-2001 : Updated header (DG);
 * 04-Dec-2001 : Changed constructors to protected, and tidied up default values (DG);
 * 19-Apr-2002 : Updated import statements (DG);
 * 05-Sep-2002 : Updated constructor for changes in Axis class (DG);
 * 06-Nov-2002 : Moved margins from the CategoryPlot class (DG);
 * 08-Nov-2002 : Moved to new package com.jrefinery.chart.axis (DG);
 * 22-Jan-2002 : Removed monolithic constructor (DG);
 * 26-Mar-2003 : Implemented Serializable (DG);
 * 09-May-2003 : Merged HorizontalMyCategoryAxis and VerticalMyCategoryAxis into this class (DG);
 * 13-Aug-2003 : Implemented Cloneable (DG);
 * 29-Oct-2003 : Added workaround for font alignment in PDF output (DG);
 * 05-Nov-2003 : Fixed serialization bug (DG);
 * 26-Nov-2003 : Added category label offset (DG);
 * 06-Jan-2004 : Moved axis line attributes to Axis class, rationalised category label 
 *               position attributes (DG);
 * 07-Jan-2004 : Added new implementation for linewrapping of category labels (DG);
 * 17-Feb-2004 : Moved deprecated code to bottom of source file (DG);
 * 10-Mar-2004 : Changed Dimension --> Dimension2D in text classes (DG);
 * 16-Mar-2004 : Added support for tooltips on category labels (DG);
 * 01-Apr-2004 : Changed java.awt.geom.Dimension2D to org.jfree.ui.Size2D because of 
 *               JDK bug 4976448 which persists on JDK 1.3.1 (DG);
 * 03-Sep-2004 : Added 'maxCategoryLabelLines' attribute (DG);
 *
 */

package client;

import java.awt.Graphics2D;
import java.awt.Insets;
import java.awt.Shape;
import java.awt.geom.Rectangle2D;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.jfree.chart.axis.*;
import org.jfree.chart.entity.EntityCollection;
import org.jfree.chart.entity.TickLabelEntity;
import org.jfree.chart.event.AxisChangeEvent;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.Plot;
import org.jfree.chart.plot.PlotRenderingInfo;
import org.jfree.text.G2TextMeasurer;
import org.jfree.text.TextBlock;
import org.jfree.text.TextUtilities;
import org.jfree.ui.RectangleAnchor;
import org.jfree.ui.RectangleEdge;
import org.jfree.ui.Size2D;
import org.jfree.util.Log;
import org.jfree.util.LogContext;
import org.jfree.util.ObjectUtils;
import org.jfree.util.ShapeUtils;

/**
 * An axis that displays categories.
 */
public class MyCategoryAxis extends CategoryAxis {

    /** The default margin for the axis (used for both lower and upper margins). */
    public static final double DEFAULT_AXIS_MARGIN = 0.05;

    /** The default margin between categories (a percentage of the overall axis length). */
    public static final double DEFAULT_CATEGORY_MARGIN = 0.20;

    /** The amount of space reserved at the start of the axis. */
    private double lowerMargin;

    /** The amount of space reserved at the end of the axis. */
    private double upperMargin;

    /** The amount of space reserved between categories. */
    private double categoryMargin;
    
    /** The maximum number of lines for category labels. */
    private int maxCategoryLabelLines;

    /** 
     * A ratio that is multiplied by the width of one category to determine the 
     * maximum label width. 
     */
    private float maxCategoryLabelWidthRatio;
    
    /** The category label offset. */
    private int categoryLabelPositionOffset; 
    
    /** A structure defining the category label positions for each axis location. */
    private CategoryLabelPositions categoryLabelPositions;
    
    /** Storage for the category label tooltips (if any). */
    private Map categoryLabelToolTips;
    
    /** Access to logging facilities. */
    protected static final LogContext logger = Log.createContext(MyCategoryAxis.class);

    /** How many tick labels to show; 1 is every, 2 is every other,
     *  3 is every third, etc.
     *
     *  Added by Isaac (14-June-2007)
     */
    private int tickLabelsSkip = 1; 
    
    /**
     * Allows tick labels to be skipped when rendering the chart.
     * Added by Isaac (14-June-2007).
     *
     * @param skip   the frequency of visible labels:
     *             1 = every label showing, 2 = every other,
     *             3 = every third, etc.
     */
     public void setTickLabelSkip ( int skip ) {
       if ( skip < 1 ) skip = 1;
       tickLabelsSkip = skip;
            notifyListeners(new AxisChangeEvent(this));
    } 
     
    /**
     * Creates a new category axis with no label.
     */
    public MyCategoryAxis() {
        this(null);    
    }
    
    /**
     * Constructs a category axis, using default values where necessary.
     *
     * @param label  the axis label (<code>null</code> permitted).
     */
    public MyCategoryAxis(String label) {

        super(label);

        this.lowerMargin = DEFAULT_AXIS_MARGIN;
        this.upperMargin = DEFAULT_AXIS_MARGIN;
        this.categoryMargin = DEFAULT_CATEGORY_MARGIN;
        this.maxCategoryLabelLines = 1;
        this.maxCategoryLabelWidthRatio = 0.0f;
        
        setTickMarksVisible(false);  // not supported by this axis type yet
        
        this.categoryLabelPositionOffset = 4;
        this.categoryLabelPositions = CategoryLabelPositions.STANDARD;
        this.categoryLabelToolTips = new HashMap();
        
    }

    /**
     * Returns the lower margin for the axis.
     *
     * @return the margin.
     */
    public double getLowerMargin() {
        return this.lowerMargin;
    }

    /**
     * Sets the lower margin for the axis and sends an {@link AxisChangeEvent} to all registered
     * listeners.
     *
     * @param margin  the margin as a percentage of the axis length (for example, 0.05 is five 
     *                percent).
     */
    public void setLowerMargin(double margin) {
        this.lowerMargin = margin;
        notifyListeners(new AxisChangeEvent(this));
    }

    /**
     * Returns the upper margin for the axis.
     *
     * @return the margin.
     */
    public double getUpperMargin() {
        return this.upperMargin;
    }

    /**
     * Sets the upper margin for the axis and sends an {@link AxisChangeEvent} to all registered
     * listeners.
     *
     * @param margin  the margin as a percentage of the axis length (for example, 0.05 is five 
     *                percent).
     */
    public void setUpperMargin(double margin) {
        this.upperMargin = margin;
        notifyListeners(new AxisChangeEvent(this));
    }

    /**
     * Returns the category margin.
     *
     * @return the margin.
     */
    public double getCategoryMargin() {
        return this.categoryMargin;
    }

    /**
     * Sets the category margin and sends an {@link AxisChangeEvent} to all registered
     * listeners.
     * <P>
     * The overall category margin is distributed over N-1 gaps, where N is the number of 
     * categories on the axis.
     *
     * @param margin  the margin as a percentage of the axis length (for example, 0.05 is five 
     *                percent).
     */
    public void setCategoryMargin(double margin) {
        this.categoryMargin = margin;
        notifyListeners(new AxisChangeEvent(this));
    }

    /**
     * Returns the maximum number of lines to use for each category label.
     * 
     * @return The maximum number of lines.
     */
    public int getMaxCategoryLabelLines() {
        return this.maxCategoryLabelLines;
    }
    
    /**
     * Sets the maximum number of lines to use for each category label.
     * 
     * @param lines  the maximum number of lines.
     */
    public void setMaxCategoryLabelLines(int lines) {
        this.maxCategoryLabelLines = lines;
        notifyListeners(new AxisChangeEvent(this));
    }
    
    /**
     * Returns the category label width ratio.
     * 
     * @return the ratio.
     */
    public float getMaxCategoryLabelWidthRatio() {
        return this.maxCategoryLabelWidthRatio;
    }
    
    /**
     * Sets the maximum category label width ratio and sends an {@link AxisChangeEvent} to
     * all registered listeners.
     * 
     * @param ratio  the ratio.
     */
    public void setMaxCategoryLabelWidthRatio(float ratio) {
        this.maxCategoryLabelWidthRatio = ratio;
        notifyListeners(new AxisChangeEvent(this));
    }
    
    /**
     * Returns the offset between the axis and the category labels (before label positioning 
     * is taken into account).
     * 
     * @return the offset (in Java2D units).
     */
    public int getCategoryLabelPositionOffset() {
        return this.categoryLabelPositionOffset;
    }
    
    /**
     * Sets the offset between the axis and the category labels (before label positioning 
     * is taken into account).
     * 
     * @param offset  the offset (in Java2D units).
     */
    public void setCategoryLabelPositionOffset(int offset) {
        this.categoryLabelPositionOffset = offset;
        notifyListeners(new AxisChangeEvent(this));
    }
    
    /**
     * Returns the category label position specification (this contains label positioning info 
     * for all four possible axis locations).
     * 
     * @return the positions (never <code>null</code>).
     */
    public CategoryLabelPositions getCategoryLabelPositions() {
        return this.categoryLabelPositions;
    }
    
    /**
     * Sets the category label position specification for the axis and sends an 
     * {@link AxisChangeEvent} to all registered listeners.
     * 
     * @param positions  the positions (<code>null</code> not permitted).
     */
    public void setCategoryLabelPositions(CategoryLabelPositions positions) {
        if (positions == null) {
            throw new IllegalArgumentException("Null 'positions' argument.");   
        }
        this.categoryLabelPositions = positions;
        notifyListeners(new AxisChangeEvent(this));
    }
    
    /**
     * Adds a tooltip to the specified category and sends an {@link AxisChangeEvent} to 
     * all registered listeners.
     * 
     * @param category  the category (<code>null<code> not permitted).
     * @param tooltip  the tooltip text (<code>null</code> permitted).
     */
    public void addCategoryLabelToolTip(Comparable category, String tooltip) {
        if (category == null) {
            throw new IllegalArgumentException("Null 'category' argument.");   
        }
        this.categoryLabelToolTips.put(category, tooltip);
        notifyListeners(new AxisChangeEvent(this));
    }
    
    /**
     * Removes the tooltip for the specified category and sends an {@link AxisChangeEvent} to 
     * all registered listeners.
     * 
     * @param category  the category (<code>null<code> not permitted).
     */
    public void removeCategoryLabelToolTip(Comparable category) {
        if (category == null) {
            throw new IllegalArgumentException("Null 'category' argument.");   
        }
        this.categoryLabelToolTips.remove(category);   
        notifyListeners(new AxisChangeEvent(this));
    }
    
    /**
     * Clears the category label tooltips and sends an {@link AxisChangeEvent} to all registered
     * listeners.
     */
    public void clearCategoryLabelToolTips() {
        this.categoryLabelToolTips.clear();
        notifyListeners(new AxisChangeEvent(this));
    }
    
    /**
     * Returns the Java 2D coordinate for a category.
     * 
     * @param anchor  the anchor point.
     * @param category  the category index.
     * @param categoryCount  the category count.
     * @param area  the data area.
     * @param edge  the location of the axis.
     * 
     * @return the coordinate.
     */
    public double getCategoryJava2DCoordinate(CategoryAnchor anchor, 
                                              int category, 
                                              int categoryCount, 
                                              Rectangle2D area,
                                              RectangleEdge edge) {
    
        double result = 0.0;
        if (anchor == CategoryAnchor.START) {
            result = getCategoryStart(category, categoryCount, area, edge);
        }
        else if (anchor == CategoryAnchor.MIDDLE) {
            result = getCategoryMiddle(category, categoryCount, area, edge);
        }
        else if (anchor == CategoryAnchor.END) {
            result = getCategoryEnd(category, categoryCount, area, edge);
        }
        return result;
                                                      
    }
                                              
    /**
     * Returns the starting coordinate for the specified category.
     *
     * @param category  the category.
     * @param categoryCount  the number of categories.
     * @param area  the data area.
     * @param edge  the axis location.
     *
     * @return the coordinate.
     */
    public double getCategoryStart(int category, int categoryCount, Rectangle2D area,
                                   RectangleEdge edge) {

        double result = 0.0;
        if ((edge == RectangleEdge.TOP) || (edge == RectangleEdge.BOTTOM)) {
            result = area.getX() + area.getWidth() * getLowerMargin();
        }
        else if ((edge == RectangleEdge.LEFT) || (edge == RectangleEdge.RIGHT)) {
            result = area.getMinY() + area.getHeight() * getLowerMargin();
        }

        double categorySize = calculateCategorySize(categoryCount, area, edge);
        double categoryGapWidth = calculateCategoryGapSize(categoryCount, area, edge);

        result = result + category * (categorySize + categoryGapWidth);

        return result;
    }

    /**
     * Returns the middle coordinate for the specified category.
     *
     * @param category  the category.
     * @param categoryCount  the number of categories.
     * @param area  the data area.
     * @param edge  the axis location.
     *
     * @return the coordinate.
     */
    public double getCategoryMiddle(int category, int categoryCount, Rectangle2D area,
                                    RectangleEdge edge) {

        return getCategoryStart(category, categoryCount, area, edge)
               + calculateCategorySize(categoryCount, area, edge) / 2;

    }

    /**
     * Returns the end coordinate for the specified category.
     *
     * @param category  the category.
     * @param categoryCount  the number of categories.
     * @param area  the data area.
     * @param edge  the axis location.
     *
     * @return the coordinate.
     */
    public double getCategoryEnd(int category, int categoryCount, Rectangle2D area,
                                 RectangleEdge edge) {

        return getCategoryStart(category, categoryCount, area, edge)
               + calculateCategorySize(categoryCount, area, edge);

    }

    /**
     * Calculates the size (width or height, depending on the location of the axis) of a category.
     *
     * @param categoryCount  the number of categories.
     * @param area  the area within which the categories will be drawn.
     * @param edge  the axis location.
     *
     * @return the category size.
     */
    protected double calculateCategorySize(int categoryCount, Rectangle2D area,
                                           RectangleEdge edge) {

        double result = 0.0;
        double available = 0.0;

        if ((edge == RectangleEdge.TOP) || (edge == RectangleEdge.BOTTOM)) {
            available = area.getWidth();
        }
        else if ((edge == RectangleEdge.LEFT) || (edge == RectangleEdge.RIGHT)) {
            available = area.getHeight();
        }
        if (categoryCount > 1) {
            result = available * (1 - getLowerMargin() - getUpperMargin() - getCategoryMargin());
            result = result / categoryCount;
        }
        else {
            result = available * (1 - getLowerMargin() - getUpperMargin());
        }
        return result;

    }

    /**
     * Calculates the size (width or height, depending on the location of the axis) of a category
     * gap.
     *
     * @param categoryCount  the number of categories.
     * @param area  the area within which the categories will be drawn.
     * @param edge  the axis location.
     *
     * @return the category gap width.
     */
    protected double calculateCategoryGapSize(int categoryCount, Rectangle2D area,
                                              RectangleEdge edge) {

        double result = 0.0;
        double available = 0.0;

        if ((edge == RectangleEdge.TOP) || (edge == RectangleEdge.BOTTOM)) {
            available = area.getWidth();
        }
        else if ((edge == RectangleEdge.LEFT) || (edge == RectangleEdge.RIGHT)) {
            available = area.getHeight();
        }

        if (categoryCount > 1) {
            result = available * getCategoryMargin() / (categoryCount - 1);
        }

        return result;

    }

    /**
     * Estimates the space required for the axis, given a specific drawing area.
     *
     * @param g2  the graphics device (used to obtain font information).
     * @param plot  the plot that the axis belongs to.
     * @param plotArea  the area within which the axis should be drawn.
     * @param edge  the axis location (top or bottom).
     * @param space  the space already reserved.
     *
     * @return The space required to draw the axis.
     */
    public AxisSpace reserveSpace(Graphics2D g2, Plot plot, Rectangle2D plotArea, 
                                  RectangleEdge edge, AxisSpace space) {

        // create a new space object if one wasn't supplied...
        if (space == null) {
            space = new AxisSpace();
        }
        
        // if the axis is not visible, no additional space is required...
        if (!isVisible()) {
            return space;
        }

        // calculate the max size of the tick labels (if visible)...
        double tickLabelHeight = 0.0;
        double tickLabelWidth = 0.0;
        if (isTickLabelsVisible()) {
            g2.setFont(getTickLabelFont());
            AxisState state = new AxisState();
            refreshTicks(g2, state, plotArea, plotArea, edge);
            if (edge == RectangleEdge.TOP) {
                tickLabelHeight = state.getMax();
            }
            else if (edge == RectangleEdge.BOTTOM) {
                tickLabelHeight = state.getMax();
            }
            else if (edge == RectangleEdge.LEFT) {
                tickLabelWidth = state.getMax(); 
            }
            else if (edge == RectangleEdge.RIGHT) {
                tickLabelWidth = state.getMax(); 
            }
        }
        
        // get the axis label size and update the space object...
        Rectangle2D labelEnclosure = getLabelEnclosure(g2, edge);
        double labelHeight = 0.0;
        double labelWidth = 0.0;
        if (RectangleEdge.isTopOrBottom(edge)) {
            labelHeight = labelEnclosure.getHeight();
            space.add(labelHeight + tickLabelHeight + this.categoryLabelPositionOffset, edge);
        }
        else if (RectangleEdge.isLeftOrRight(edge)) {
            labelWidth = labelEnclosure.getWidth();
            space.add(labelWidth + tickLabelWidth + this.categoryLabelPositionOffset, edge);
        }

        return space;

    }

    /**
     * Configures the axis against the current plot.
     */
    public void configure() {
        // nothing required
    }

    /**
     * Draws the axis on a Java 2D graphics device (such as the screen or a printer).
     *
     * @param g2  the graphics device (<code>null</code> not permitted).
     * @param cursor  the cursor location.
     * @param plotArea  the area within which the axis should be drawn (<code>null</code> not 
     *                  permitted).
     * @param dataArea  the area within which the plot is being drawn (<code>null</code> not 
     *                  permitted).
     * @param edge  the location of the axis (<code>null</code> not permitted).
     * @param plotState  collects information about the plot (<code>null</code> permitted).
     * 
     * @return the axis state (never <code>null</code>).
     */
    public AxisState draw(Graphics2D g2, 
                          double cursor, 
                          Rectangle2D plotArea, 
                          Rectangle2D dataArea,
                          RectangleEdge edge,
                          PlotRenderingInfo plotState) {
        
        if (logger.isDebugEnabled()) {
            logger.debug("Entering draw() method, cursor = " + cursor);
        }
        
        // if the axis is not visible, don't draw it...
        if (!isVisible()) {
            return new AxisState(cursor);
        }
        
        if (isAxisLineVisible()) {
            drawAxisLine(g2, cursor, dataArea, edge);
        }

        // draw the category labels and axis label
        AxisState state = new AxisState(cursor);
        state = drawCategoryLabels(g2, plotArea, dataArea, edge, state, plotState);
        state = drawLabel(getLabel(), g2, plotArea, dataArea, edge, state);
    
        return state;

    }

    /**
     * Draws the category labels and returns the updated axis state.
     *
     * @param g2  the graphics device (<code>null</code> not permitted).
     * @param plotArea  the plot area (<code>null</code> not permitted).
     * @param dataArea  the area inside the axes (<code>null</code> not permitted).
     * @param edge  the axis location (<code>null</code> not permitted).
     * @param state  the axis state (<code>null</code> not permitted).
     * @param plotState  collects information about the plot (<code>null</code> permitted).
     * 
     * @return the updated axis state (never <code>null</code>).
     */
    protected AxisState drawCategoryLabels(Graphics2D g2,
                                           Rectangle2D plotArea,
                                           Rectangle2D dataArea,
                                           RectangleEdge edge,
                                           AxisState state,
                                           PlotRenderingInfo plotState) {

        if (logger.isDebugEnabled()) {
            logger.debug("Entering drawCategoryLabels() method, cursor = " + state.getCursor());
        }
        if (state == null) {
            throw new IllegalArgumentException("Null 'state' argument.");
        }

        if (isTickLabelsVisible()) {
            g2.setFont(getTickLabelFont());
            g2.setPaint(getTickLabelPaint());
            List ticks = refreshTicks(g2, state, plotArea, dataArea, edge);
            state.setTicks(ticks);        
          
            int categoryIndex = 0;
            Iterator iterator = ticks.iterator();
            while (iterator.hasNext()) {
                
                CategoryTick tick = (CategoryTick) iterator.next();
                // Added by Isaac (14-June-2007)
                if ( categoryIndex % tickLabelsSkip != tickLabelsSkip-1) {
                   categoryIndex ++;
                   continue;
                }
        
                g2.setPaint(getTickLabelPaint());

                CategoryLabelPosition position = this.categoryLabelPositions.getLabelPosition(edge);
                double x0 = 0.0;
                double x1 = 0.0;
                double y0 = 0.0;
                double y1 = 0.0;
                if (edge == RectangleEdge.TOP) {
                    x0 = getCategoryStart(categoryIndex, ticks.size(), dataArea, edge);
                    x1 = getCategoryEnd(categoryIndex, ticks.size(), dataArea, edge);
                    y1 = state.getCursor() - this.categoryLabelPositionOffset;
                    y0 = y1 - state.getMax();
                }
                else if (edge == RectangleEdge.BOTTOM) {
                    x0 = getCategoryStart(categoryIndex, ticks.size(), dataArea, edge);
                    x1 = getCategoryEnd(categoryIndex, ticks.size(), dataArea, edge); 
                    y0 = state.getCursor() + this.categoryLabelPositionOffset;                   
                    y1 = y0 + state.getMax();
                }
                else if (edge == RectangleEdge.LEFT) {
                    y0 = getCategoryStart(categoryIndex, ticks.size(), dataArea, edge);
                    y1 = getCategoryEnd(categoryIndex, ticks.size(), dataArea, edge);
                    x1 = state.getCursor() - this.categoryLabelPositionOffset;
                    x0 = x1 - state.getMax();
                }
                else if (edge == RectangleEdge.RIGHT) {
                    y0 = getCategoryStart(categoryIndex, ticks.size(), dataArea, edge);
                    y1 = getCategoryEnd(categoryIndex, ticks.size(), dataArea, edge);
                    x0 = state.getCursor() + this.categoryLabelPositionOffset;
                    x1 = x0 - state.getMax();
                }
                Rectangle2D area = new Rectangle2D.Double(x0, y0, (x1 - x0), (y1 - y0));
                double[] anchorPoint = RectangleAnchor.coordinates(
                    area, position.getCategoryAnchor()
                );
                TextBlock block = tick.getLabel();
                block.draw(
                    g2, 
                    (float) anchorPoint[0], (float) anchorPoint[1], position.getLabelAnchor(), 
                    (float) anchorPoint[0], (float) anchorPoint[1], position.getAngle()
                );
                Shape bounds = block.calculateBounds(
                    g2, (float) anchorPoint[0], (float) anchorPoint[1], position.getLabelAnchor(), 
                    (float) anchorPoint[0], (float) anchorPoint[1], position.getAngle()
                );
                if (plotState != null) {
                    EntityCollection entities = plotState.getOwner().getEntityCollection();
                    if (entities != null) {
                        String tooltip = (String) this.categoryLabelToolTips.get(
                            tick.getCategory()
                        );
                        entities.addEntity(new TickLabelEntity(bounds, tooltip, null));   
                    }
                }
                categoryIndex++;
            }

            if (edge.equals(RectangleEdge.TOP)) {
                double h = state.getMax();
                state.cursorUp(h);
            }
            else if (edge.equals(RectangleEdge.BOTTOM)) {
                double h = state.getMax();
                state.cursorDown(h);
            }
            else if (edge == RectangleEdge.LEFT) {
                double w = state.getMax();
                state.cursorLeft(w);
            }
            else if (edge == RectangleEdge.RIGHT) {
                double w = state.getMax();
                state.cursorRight(w);
            }
        }
        return state;
    }

    /**
     * Creates a temporary list of ticks that can be used when drawing the axis.
     *
     * @param g2  the graphics device (used to get font measurements).
     * @param state  the axis state.
     * @param plotArea  the area where the plot and axes will be drawn.
     * @param dataArea  the area inside the axes.
     * @param edge  the location of the axis.
     * 
     * @return A list of ticks.
     */
    public List refreshTicks(Graphics2D g2, 
                             AxisState state,
                             Rectangle2D plotArea, 
                             Rectangle2D dataArea,
                             RectangleEdge edge) {

        List ticks = new java.util.ArrayList();
        
        // sanity check for data area...
        if (dataArea.getHeight() <= 0.0 || dataArea.getWidth() < 0.0) {
            return ticks;
        }

        CategoryPlot plot = (CategoryPlot) getPlot();
        List categories = plot.getCategories();
        double max = 0.0;
                
        if (categories != null) {
            CategoryLabelPosition position = this.categoryLabelPositions.getLabelPosition(edge);
            float r = this.maxCategoryLabelWidthRatio;
            if (r <= 0.0) {
                r = position.getWidthRatio();   
            }
                  
            float l = 0.0f;
            if (position.getWidthType() == CategoryLabelWidthType.CATEGORY) {
                l = (float) calculateCategorySize(categories.size(), dataArea, edge);  
            }
            else {
                if (RectangleEdge.isLeftOrRight(edge)) {
                    l = (float) dataArea.getWidth();   
                }
                else {
                    l = (float) dataArea.getHeight();   
                }
            }
            int categoryIndex = 0;
            Iterator iterator = categories.iterator();
            while (iterator.hasNext()) {
                Comparable category = (Comparable) iterator.next();
                TextBlock label = createLabel(category, l * r, edge, g2);
                if (edge == RectangleEdge.TOP || edge == RectangleEdge.BOTTOM) {
                    max = Math.max(max, calculateTextBlockHeight(label, position, g2));
                }
                else if (edge == RectangleEdge.LEFT || edge == RectangleEdge.RIGHT) {
                    max = Math.max(max, calculateTextBlockWidth(label, position, g2));
                }
                Tick tick = new CategoryTick(
                    category, label, 
                    position.getLabelAnchor(), position.getRotationAnchor(), position.getAngle()
                );
                ticks.add(tick);
                categoryIndex = categoryIndex + 1;
            }
        }
        state.setMax(max);
        return ticks;
        
    }

    /**
     * Creates a label.
     *
     * @param category  the category.
     * @param width  the available width. 
     * @param edge  the edge on which the axis appears.
     * @param g2  the graphics device.
     *
     * @return a label.
     */
    protected TextBlock createLabel(Comparable category, float width, 
                                    RectangleEdge edge, Graphics2D g2) {
        TextBlock label = TextUtilities.createTextBlock(
            category.toString(), getTickLabelFont(), getTickLabelPaint(), 
            width, this.maxCategoryLabelLines, new G2TextMeasurer(g2)
        );  
        return label; 
    }
    
    /**
     * A utility method for determining the width of a text block.
     *
     * @param block  the text block.
     * @param position  the position.
     * @param g2  the graphics device.
     *
     * @return the width.
     */
    protected double calculateTextBlockWidth(TextBlock block, 
                                             CategoryLabelPosition position, 
                                             Graphics2D g2) {
                                                    
        Insets insets = getTickLabelInsets();
        Size2D size = block.calculateDimensions(g2);
        Rectangle2D box = new Rectangle2D.Double(0.0, 0.0, size.getWidth(), size.getHeight());
        Shape rotatedBox = ShapeUtils.rotateShape(box, position.getAngle(), 0.0f, 0.0f);
        double w = rotatedBox.getBounds2D().getWidth() + insets.top + insets.bottom;
        return w;
        
    }

    /**
     * A utility method for determining the height of a text block.
     *
     * @param block  the text block.
     * @param position  the label position.
     * @param g2  the graphics device.
     *
     * @return the height.
     */
    protected double calculateTextBlockHeight(TextBlock block, 
                                              CategoryLabelPosition position, 
                                              Graphics2D g2) {
                                                    
        Insets insets = getTickLabelInsets();
        Size2D size = block.calculateDimensions(g2);
        Rectangle2D box = new Rectangle2D.Double(0.0, 0.0, size.getWidth(), size.getHeight());
        Shape rotatedBox = ShapeUtils.rotateShape(box, position.getAngle(), 0.0f, 0.0f);
        double h = rotatedBox.getBounds2D().getHeight() + insets.top + insets.bottom;
        return h;
        
    }

    /**
     * Creates a clone of the axis.
     * 
     * @return a clone.
     * 
     * @throws CloneNotSupportedException if some component of the axis does not support cloning.
     */
    public Object clone() throws CloneNotSupportedException {
    
        Object clone = super.clone();
        return clone;
            
    }
    
    /**
     * Tests this axis for equality with an arbitrary object.
     *
     * @param obj  the object (<code>null</code> permitted).
     *
     * @return a boolean.
     */
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (obj instanceof MyCategoryAxis && super.equals(obj)) {
            MyCategoryAxis axis = (MyCategoryAxis) obj;
            boolean b0 = (axis.lowerMargin == this.lowerMargin);
            boolean b1 = (axis.upperMargin == this.upperMargin);
            boolean b2 = (axis.categoryMargin == this.categoryMargin);
            boolean b3 = (axis.maxCategoryLabelWidthRatio == this.maxCategoryLabelWidthRatio);
            boolean b4 = (axis.categoryLabelPositionOffset == this.categoryLabelPositionOffset);
            boolean b5 = ObjectUtils.equal(
                axis.categoryLabelPositions, this.categoryLabelPositions
            );
            boolean b6 = ObjectUtils.equal(
                axis.categoryLabelToolTips, this.categoryLabelToolTips
            );
            return b0 && b1 && b2 && b3 && b4 && b5 && b6;
        }
        return false;
    }

    /**
     * Provides serialization support.
     *
     * @param stream  the output stream.
     *
     * @throws IOException  if there is an I/O error.
     */
    private void writeObject(ObjectOutputStream stream) throws IOException {
        stream.defaultWriteObject();
    }

    /**
     * Provides serialization support.
     *
     * @param stream  the input stream.
     *
     * @throws IOException  if there is an I/O error.
     * @throws ClassNotFoundException  if there is a classpath problem.
     */
    private void readObject(ObjectInputStream stream) throws IOException, ClassNotFoundException {
        stream.defaultReadObject();
    }
    
    //// DEPRECATED CODE ////////////////////////////////////////////////////////////

    /**
     * Returns a flag indicating whether the category labels are rotated to vertical.
     *
     * @return The flag.
     * 
     * @deprecated Use the get/setXXXCategoryLabelPosition methods.
     */
    public boolean isVerticalCategoryLabels() {
        return false;
    }

    /**
     * Sets the flag that determines whether the category labels are rotated to vertical.
     *
     * @param flag  the flag.
     * 
     * @deprecated Use the get/setXXXCategoryLabelPosition methods.
     */
    public void setVerticalCategoryLabels(boolean flag) {

        if (flag) {
            this.categoryLabelPositions = CategoryLabelPositions.UP_90;
        }
        else {
            this.categoryLabelPositions = CategoryLabelPositions.STANDARD;
        }
        notifyListeners(new AxisChangeEvent(this));

    }

    /**
     * Returns the flag that determines whether the category labels are to be
     * skipped to avoid overlapping.
     *
     * @return The flag.
     * 
     * @deprecated No longer supported.
     */
    public boolean getSkipCategoryLabelsToFit() {
        return false;
    }

    /**
     * Sets the flag that determines whether the category labels are to be
     * skipped to avoid overlapping.
     *
     * @param flag  the new value of the flag.
     * 
     * @deprecated No longer supported.
     */
    public void setSkipCategoryLabelsToFit(boolean flag) {
        // not supported any more
    }

    /**
     * Returns the category label positioning info that applies when the axis is displayed
     * at the top of the plot area.
     * 
     * @return the position info.
     * 
     * @deprecated Use getCategoryLabelPositions().getLabelPosition(RectangleEdge.TOP).
     */
    public CategoryLabelPosition getTopCategoryLabelPosition() {
        return getCategoryLabelPositions().getLabelPosition(RectangleEdge.TOP);
    }
    
    /**
     * Sets the position info that applies when the axis is displayed at the top of the 
     * plot area.
     * 
     * @param position  the position info.
     * 
     * @deprecated Use setCategoryLabelPositions(...).
     */
    public void setTopCategoryLabelPosition(CategoryLabelPosition position) {
        setCategoryLabelPositions(new CategoryLabelPositions(
                position,
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.BOTTOM),
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.LEFT),
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.RIGHT)
        )
        );
    }

    /**
     * Returns the category label positioning info that applies when the axis is displayed
     * at the bottom of the plot area.
     * 
     * @return the position info.
     * 
     * @deprecated Use getCategoryLabelPositions().getLabelPosition(RectangleEdge.BOTTOM).
     */
    public CategoryLabelPosition getBottomCategoryLabelPosition() {
        return getCategoryLabelPositions().getLabelPosition(RectangleEdge.BOTTOM);
    }
    
    /**
     * Sets the position info that applies when the axis is displayed at the bottom of the 
     * plot area.
     * 
     * @param position  the position info.
     * 
     * @deprecated Use setCategoryLabelPositions(...).
     */
    public void setBottomCategoryLabelPosition(CategoryLabelPosition position) {
        setCategoryLabelPositions(new CategoryLabelPositions(
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.TOP),
                position,
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.LEFT),
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.RIGHT)
        )
        );
    }

    /**
     * Returns the category label positioning info that applies when the axis is displayed
     * at the left of the plot area.
     * 
     * @return the position info.
     * 
     * @deprecated Use getCategoryLabelPositions().getLabelPosition(RectangleEdge.LEFT).
     */
    public CategoryLabelPosition getLeftCategoryLabelPosition() {
        return getCategoryLabelPositions().getLabelPosition(RectangleEdge.LEFT);
    }
    
    /**
     * Sets the position info that applies when the axis is displayed at the left of the 
     * plot area.
     * 
     * @param position  the position info.
     * 
     * @deprecated Use setCategoryLabelPositions(...).
     */
    public void setLeftCategoryLabelPosition(CategoryLabelPosition position) {
        setCategoryLabelPositions(new CategoryLabelPositions(
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.TOP),
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.BOTTOM),
                position,
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.RIGHT)
        )
        );
    }

    /**
     * Returns the category label positioning info that applies when the axis is displayed
     * at the right of the plot area.
     * 
     * @return the position info.
     * 
     * @deprecated Use getCategoryLabelPositions().getLabelPosition(RectangleEdge.RIGHT).
     */
    public CategoryLabelPosition getRightCategoryLabelPosition() {
        return getCategoryLabelPositions().getLabelPosition(RectangleEdge.RIGHT);
    }
    
    /**
     * Sets the position info that applies when the axis is displayed at the right of the 
     * plot area.
     * 
     * @param position  the position info.
     * 
     * @deprecated Use setLabelPositions(...).
     */
    public void setRightCategoryLabelPosition(CategoryLabelPosition position) {
        setCategoryLabelPositions(new CategoryLabelPositions(
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.TOP),
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.BOTTOM),
                this.categoryLabelPositions.getLabelPosition(RectangleEdge.LEFT),
                position
        )
        );
    }

}
