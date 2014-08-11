package edu.uoregon.tau.perfexplorer.client;

import java.awt.Color;
import java.awt.Paint;

import org.jfree.chart.renderer.category.LineAndShapeRenderer;

public class SpeedupLineAndShapeRenderer extends LineAndShapeRenderer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5002401689897783294L;

	public SpeedupLineAndShapeRenderer (int numRows) {
		super();
	}

	public java.awt.Paint getSeriesPaint(int series) {
		if (series == 0)
			return Color.black;
		else
        	return GoodColors.colors[series%GoodColors.numcolors];
	}
}
