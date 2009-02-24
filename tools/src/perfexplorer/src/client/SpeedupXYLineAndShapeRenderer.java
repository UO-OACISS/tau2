package edu.uoregon.tau.perfexplorer.client;

import java.awt.*;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

public class SpeedupXYLineAndShapeRenderer extends XYLineAndShapeRenderer {
	private int numRows = 0;

	public SpeedupXYLineAndShapeRenderer (int numRows) {
		super();
		this.numRows = numRows;
	}

	public java.awt.Paint getSeriesPaint(int series) {
		if (series == numRows)
			return Color.black;
		else
			return super.getSeriesPaint(series);
	}
}
