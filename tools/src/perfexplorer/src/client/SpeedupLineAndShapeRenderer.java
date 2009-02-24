package edu.uoregon.tau.perfexplorer.client;

import java.awt.*;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;

public class SpeedupLineAndShapeRenderer extends LineAndShapeRenderer {

	public SpeedupLineAndShapeRenderer (int numRows) {
		super();
	}

	public java.awt.Paint getSeriesPaint(int series) {
		if (series == 0)
			return Color.black;
		else
			return super.getSeriesPaint(series);
	}
}
