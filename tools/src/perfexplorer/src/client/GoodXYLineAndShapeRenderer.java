package edu.uoregon.tau.perfexplorer.client;

import java.awt.Color;
import java.awt.Paint;

import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

public class GoodXYLineAndShapeRenderer extends XYLineAndShapeRenderer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4962881172454711849L;

	public GoodXYLineAndShapeRenderer () {
		super();
	}

    public java.awt.Paint getSeriesPaint(int series) {
        return GoodColors.colors[series%GoodColors.numcolors];
    }

}
