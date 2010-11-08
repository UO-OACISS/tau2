package edu.uoregon.tau.paraprof;

import edu.uoregon.tau.paraprof.enums.ValueType;
import edu.uoregon.tau.paraprof.enums.VisType;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Metric;
import edu.uoregon.tau.perfdmf.Thread;
import edu.uoregon.tau.vis.Vec;
import edu.uoregon.tau.vis.Axes.Orientation;

/**
 * Represents the settings of the 3d window/control panels
 * This class is not really that useful since the shapes (plots) themselves
 * do most of the controlling.  The original idea was that this could be
 * cloned such that the user could continue to drag a slider, so if the visualization
 * was slow, it wouldn't do a redraw 5 times inbetween, you would just get the 
 * last setting.  Unfortunately, all the JOGL stuff is forced onto the AWT-EventQueue
 * thread, so my plan got nixed.
 *    
 * TODO : ...
 *
 * <P>CVS $Id: ThreeDeeSettings.java,v 1.3 2009/09/10 00:13:49 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.3 $
 */
public class ThreeDeeSettings implements Cloneable {

    private float plotWidth, plotHeight, plotDepth;

    private Metric heightMetric, colorMetric;
    private ValueType colorValue = ValueType.EXCLUSIVE, heightValue = ValueType.EXCLUSIVE;
    private VisType visType = VisType.TRIANGLE_MESH_PLOT;
    //private VisType visType = VisType.BAR_PLOT;

    private Orientation axisOrientation = Orientation.NW;
    private boolean axesEnabled = true;

    private Metric[] scatterMetrics = { null, null, null, null };
    private ValueType[] scatterValueTypes = { ValueType.EXCLUSIVE, ValueType.EXCLUSIVE, ValueType.EXCLUSIVE, ValueType.EXCLUSIVE };
    private Function[] scatterFunctions = new Function[4];
    //private int[] topoRanges = new int[2];
    private int minTopoRange=0;
    private int maxTopoRange=100;
    
    private int minTopoValue=0;
    private int maxTopoValue=0;
    
    public int getMinTopoValue() {
		return minTopoValue;
	}

	public void setMinTopoValue(int minTopoValue) {
		this.minTopoValue = minTopoValue;
	}

	public int getMaxTopoValue() {
		return maxTopoValue;
	}

	public void setMaxTopoValue(int maxTopoValue) {
		this.maxTopoValue = maxTopoValue;
	}

	private Metric topoMetric=null;
    private ValueType topoValueType = ValueType.EXCLUSIVE;
    private Function topoFunction = null;
    private String topoCart = null;

    public Metric getTopoMetric() {
		return topoMetric;
	}

	public void setTopoMetric(Metric topoMetric) {
		this.topoMetric = topoMetric;
	}

	public ValueType getTopoValueType() {
		return topoValueType;
	}

	public void setTopoValueType(ValueType topoValueType) {
		this.topoValueType = topoValueType;
	}

	public String getTopoCart() {
		return topoCart;
	}

	public void setTopoCart(String topoCart) {
		this.topoCart = topoCart;
	}

//	public void setTopoRanges(int[] topoValues) {
//		this.topoRanges = topoValues;
//	}

	private Thread selectedThread;

    // the function and thread selected by the two scrollbars
    private int[] selections = { -1, 0 };

    private Vec scatterAim = new Vec(7.5f, 7.5f, 7.5f), scatterEye;
    private Vec regularAim, regularEye;

    /**
     * @return Returns the visType.
     */
    public VisType getVisType() {
        return visType;
    }

    /**
     * @param visType The visType to set.
     */
    public void setVisType(VisType visType) {
        this.visType = visType;
    }

    /**
     * @return Returns the colorValue.
     */
    public ValueType getColorValue() {
        return colorValue;
    }

    /**
     * @param colorValue The colorValue to set.
     */
    public void setColorValue(ValueType colorValue) {
        this.colorValue = colorValue;
    }

    /**
     * @return Returns the heightValue.
     */
    public ValueType getHeightValue() {
        return heightValue;
    }

    /**
     * @param heightValue The heightValue to set.
     */
    public void setHeightValue(ValueType heightValue) {
        this.heightValue = heightValue;
    }

    public void setSize(float plotWidth, float plotDepth, float plotHeight) {
        this.plotWidth = plotWidth;
        this.plotDepth = plotDepth;
        this.plotHeight = plotHeight;

    }

    public float getPlotHeight() {
        return plotHeight;
    }

    public float getPlotDepth() {
        return plotDepth;
    }

    public float getPlotWidth() {
        return plotWidth;
    }

    public void setHeightMetric(Metric metric) {
        this.heightMetric = metric;
    }

    public void setColorMetric(Metric metric) {
        this.colorMetric = metric;
    }

    public Metric getHeightMetric() {
        return this.heightMetric;
    }

    public Metric getColorMetric() {
        return this.colorMetric;
    }

    public Object clone() {
        ThreeDeeSettings newSettings = new ThreeDeeSettings();

        newSettings.plotDepth = this.plotDepth;
        newSettings.plotHeight = this.plotHeight;
        newSettings.plotWidth = this.plotWidth;
        newSettings.heightMetric = this.heightMetric;
        newSettings.colorMetric = this.colorMetric;
        newSettings.colorValue = this.colorValue;
        newSettings.heightValue = this.heightValue;

        newSettings.visType = this.visType;
        newSettings.axisOrientation = this.axisOrientation;
        newSettings.axesEnabled = this.axesEnabled;

        newSettings.scatterMetrics = (Metric[]) this.scatterMetrics.clone();
        newSettings.scatterValueTypes = (ValueType[]) this.scatterValueTypes.clone();
        newSettings.scatterFunctions = (Function[]) this.scatterFunctions.clone();
        //newSettings.topoRanges = (int[]) this.topoRanges.clone();
        newSettings.minTopoRange=this.minTopoRange;
        newSettings.maxTopoRange=this.maxTopoRange;
        
        newSettings.minTopoValue=this.minTopoValue;
        newSettings.maxTopoValue=this.maxTopoValue;

        newSettings.topoMetric= this.topoMetric;
        newSettings.topoCart=this.topoCart;
        newSettings.topoValueType=this.topoValueType;
        
        newSettings.regularAim = this.regularAim;
        newSettings.regularEye = this.regularEye;

        newSettings.scatterAim = this.scatterAim;
        newSettings.scatterEye = this.scatterEye;

        newSettings.selections = (int[]) this.selections.clone();

        return newSettings;
    }

    //    public Plot getPlot() {
    //        return plot;
    //    }

    //    public Axes getAxes() {
    //        return axes;
    //    }
    /**
     * @return Returns the axisOrientation.
     */
    public Orientation getAxisOrientation() {
        return axisOrientation;
    }

    /**
     * @param axisOrientation The axisOrientation to set.
     */
    public void setAxisOrientation(Orientation axisOrientation) {
        this.axisOrientation = axisOrientation;
    }

    public boolean isAxesEnabled() {
        return axesEnabled;
    }

    public void setAxesEnabled(boolean axesEnabled) {
        this.axesEnabled = axesEnabled;
    }

//    public int[] getTopoRanges(){
//        return topoRanges;
//    }
//    public void setTopoRanges(int value, int index) {
//        this.topoRanges[index] = value;
//    }
    
    public Function getTopoFunction(){
    	return topoFunction;
    }
    public int getMinTopoRange() {
		return minTopoRange;
	}

	public void setMinTopoRange(int minTopoRange) {
		this.minTopoRange = minTopoRange;
	}

	public int getMaxTopoRange() {
		return maxTopoRange;
	}

	public void setMaxTopoRange(int maxTopoRange) {
		this.maxTopoRange = maxTopoRange;
	}

	public void setTopoFunction(Function f){
    	topoFunction=f;
    }
    
    public Function[] getScatterFunctions() {
        return scatterFunctions;
    }

    public void setScatterFunction(Function function, int index) {
        this.scatterFunctions[index] = function;
    }

    public Metric[] getScatterMetrics() {
        return scatterMetrics;
    }

    public void setScatterMetric(Metric scatterMetric, int index) {
        this.scatterMetrics[index] = scatterMetric;
    }

    public ValueType[] getScatterValueTypes() {
        return scatterValueTypes;
    }

    public void setScatterValueType(ValueType scatterValueType, int index) {
        this.scatterValueTypes[index] = scatterValueType;
    }

    public Vec getRegularAim() {
        return regularAim;
    }

    public void setRegularAim(Vec regularAim) {
        this.regularAim = regularAim;
    }

    public Vec getRegularEye() {
        return regularEye;
    }

    public void setRegularEye(Vec regularEye) {
        this.regularEye = regularEye;
    }

    public Vec getScatterAim() {
        return scatterAim;
    }

    public void setScatterAim(Vec scatterAim) {
        this.scatterAim = scatterAim;
    }

    public Vec getScatterEye() {
        return scatterEye;
    }

    public void setScatterEye(Vec scatterEye) {
        this.scatterEye = scatterEye;
    }

    public int[] getSelections() {
        return this.selections;
    }

    public void setSelection(int index, int value) {
        this.selections[index] = value;
    }

    public Thread getSelectedThread() {
        return selectedThread;
    }

    public void setSelectedThread(Thread selectedThread) {
        this.selectedThread = selectedThread;
    }

}
