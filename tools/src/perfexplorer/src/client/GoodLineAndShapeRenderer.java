package edu.uoregon.tau.perfexplorer.client;

import java.awt.Color;
import java.awt.Paint;

import org.jfree.chart.renderer.category.LineAndShapeRenderer;

public class GoodLineAndShapeRenderer extends LineAndShapeRenderer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4962881172454711849L;

	public GoodLineAndShapeRenderer () {
		super();
	}

    public java.awt.Paint getSeriesPaint(int series) {
        return GoodColors.colors[series%GoodColors.numcolors];
    }

}
