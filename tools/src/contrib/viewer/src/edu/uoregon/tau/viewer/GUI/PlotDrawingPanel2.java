/*
 * Created on Nov 4, 2003
 *
 */
package edu.uoregon.tau.viewer.GUI;

import javax.swing.*;
import java.awt.*;
import java.awt.color.*;
import java.awt.geom.*;
import java.awt.event.*;

/**
 * @author lili
 *
 */
public class PlotDrawingPanel2 extends JPanel {
	
	int[] xValues, xAxisTicks;
	double[] yValues;
	double[] yValues2;
	double[] yValues3;
	float[] yAxisTicks;
	String title, xLabel, yLabel;
	String[] legends;
	
	int maxX, minX, magOfY = 0; // magOfY is magnititude of y axis values.
	double maxY, minY, yTickIncrement, magOfYValue;
	int xTickNumber, yTickNumber; // yTickNumber counts in original point (0,0)
	
	// final int xAxisLength = 480, yAxisLength = 400, panelWidth = 580, panelHeight = 530, maxYTickNumber = 8,
	final int xAxisLength = 240, yAxisLength = 200, panelWidth = 290, panelHeight = 265, maxYTickNumber = 8,
	tickHeight = 5; // set length, width etc. for various components.
	
	public PlotDrawingPanel2(int[] x, double[] y, double[] y2, double[] y3, String title, String xLabel, String yLabel, String[] legends) {
		super();
		
		// set backgroup color
		setBackground(Color.lightGray);
		
		// set panel size
		setPreferredSize(new Dimension(panelWidth, panelHeight));
		
		// initialization
		this.xValues = x;
		this.xAxisTicks = xValues;	
		this.xTickNumber = xValues.length;	
		this.yAxisTicks = new float[maxYTickNumber];
		
		this.yValues = y;
		this.yValues2 = y2;
		this.yValues3 = y3;
		this.xLabel = xLabel;
		this.yLabel = yLabel;
		this.title = title;
		this.legends = legends;
		
		findMaxValues();			
		setYAxisTickIncrement();			
	}
	
	public void findMaxValues(){
		maxX = minX = xValues[0];
		maxY = minY = yValues[0];
		
		int i = 1;
		
		while (i < xValues.length){
			if (xValues[i] > maxX)
				maxX = xValues[i];
				
			if (xValues[i] < minX)
				minX = xValues[i];
				
			i++;		
		}
		
		i = 1;
		
		while (i < yValues.length){
			if (yValues[i] > maxY)
				maxY = yValues[i];
			if (yValues[i] < minY)
				minY = yValues[i];
			if (yValues2[i] > maxY)
				maxY = yValues2[i];
			if (yValues2[i] < minY)
				minY = yValues2[i];
			if (yValues3[i] > maxY)
				maxY = yValues3[i];
			if (yValues3[i] < minY)
				minY = yValues3[i];
			i++;			
		}
	}
	
	// here assume all y values are positive.
	public void setYAxisTickIncrement(){
				
		// compute magnitude of max y value.
						
		double tmpY, tmpInc, maxYTick, tmpDouble;
		
		tmpY = (maxY == 0.0) ? 1.0 : maxY;

		Double dou; 

		tmpInc = 1;
		
		if (tmpY >=1 ){					
			while (tmpY >= 10 ){
				tmpY = tmpY/10;
				tmpInc *= 10;
				magOfY ++;				
			}
			
			magOfYValue = tmpInc;
			
			tmpInc /= 10;
						
			tmpDouble = maxY/((maxYTickNumber-1)*tmpInc);
						
			dou = new Double(tmpDouble);
			
			//if (tmpDouble > (dou.intValue()+0.5)){
			tmpInc *= dou.intValue() + 1;
			//}
			/*else {			
				tmpInc *= dou.intValue() + 0.5;
			}*/
			
			yTickIncrement = tmpInc;			
		
			yTickNumber = 1; // yTickNumber is the number of ticks which are eventually drawn.
			tmpY = 0.0;
			yAxisTicks[0] = 0;
			while (tmpY <= maxY){
				
				tmpY = yTickIncrement*yTickNumber;
								
				if (magOfY > 3){
					yAxisTicks[yTickNumber] = (new Double(tmpY/magOfYValue)).floatValue();
				}
				else
					yAxisTicks[yTickNumber] = (new Double(tmpY)).floatValue();
								
				
				yTickNumber ++;			
			}
		}		
		else { // 0 < tmpY < 1
			while (tmpY < 1){
				tmpY = tmpY*10.0;
				tmpInc /= 10.0;
				magOfY --;				
			}
			
			magOfYValue = tmpInc;
			
			tmpInc /= 10;
			
			tmpDouble = maxY/((maxYTickNumber-1)*tmpInc);
			
			dou = new Double(tmpDouble);
						
			//if (tmpDouble > (dou.intValue()+0.5)){
				tmpInc *= dou.intValue() + 1;
			/*}
			else {			
				tmpInc *= dou.intValue() + 0.5;
			}*/
			
			yTickIncrement = tmpInc;
								
			yTickNumber = 1; // yTickNumber is the number of ticks which are eventually drawn.
			tmpY = yAxisTicks[0] = 0;
			float ff;
			
			while (tmpY <= maxY){
				
				tmpY = yTickIncrement*yTickNumber;
				
				if (magOfY < -2){
					yAxisTicks[yTickNumber] = (new Double(tmpY/magOfYValue)).floatValue();
				}
				else
					yAxisTicks[yTickNumber] = (new Double(tmpY)).floatValue();
								
				//yAxisTicks[yTickNumber] = (new Double(tmpY/magOfYValue)).floatValue();
				
				yTickNumber ++;							
			}					
		}
	}
	
	public String getToolTipText(MouseEvent evt){
		return "";
	}
	
	public void drawAGlyph(double x, double y){ // create a set of various glyphs for drawing multiple curves 
		
	}
	
	public void paintComponent(Graphics g){	
	   	super.paintComponent(g);	
	   	Graphics2D g2 = (Graphics2D) g;	
		
		// set tick number font
		Font tickFont = new Font(null, Font.BOLD, 12);
		
		// set tick length
		int tickLength = 5, maxTickStringLength = 0;
		
		// set x y label font
		Font labelFont = new Font(null, Font.BOLD, 12);
		
		// set title font
		Font titleFont = new Font(null, Font.BOLD, 16);
		
		// draw coordinates as a rectangle
		g2.setPaint(Color.black);
		
		// x coordinate location
		double xCoor = (panelWidth - xAxisLength)/2 + tickFont.getSize() + labelFont.getSize(); 
		
		// y coordinate location
		double yCoor = (panelHeight - yAxisLength)/2;
		
		Rectangle2D.Double coorRect = new Rectangle2D.Double(xCoor, yCoor, xAxisLength, yAxisLength);		
		g2.draw(coorRect);
		
		// fill the rectangle in white
		g2.setPaint(Color.white);
		g2.fill(new Rectangle2D.Double(xCoor+1, yCoor+1, xAxisLength-1, yAxisLength-1));
				
		// draw x ticks
		g2.setPaint(Color.black);
		g2.setFont(tickFont);
		
		double xPos, yPos;
		String tick;
		for (int i=0; i<xAxisTicks.length; i++){
			
			xPos = xCoor + ((xAxisTicks[i]-minX)*xAxisLength) / (maxX-minX);
			g2.draw(new Line2D.Double(xPos, yCoor+yAxisLength, xPos, yCoor+yAxisLength-tickLength));
			g2.draw(new Line2D.Double(xPos, yCoor, xPos, yCoor+tickLength));
			
			tick = Integer.toString(xAxisTicks[i]);
			g2.drawString(tick, (int) (xPos-g2.getFontMetrics(tickFont).stringWidth(tick)/2), (int) yCoor+yAxisLength+tickFont.getSize());	
		}
		
		// draw y ticks
		int tmpInt;
		for (int j=0; j<yTickNumber; j++){
			yPos = yCoor + yAxisLength - (yAxisTicks[j]*yAxisLength) / yAxisTicks[yTickNumber-1];
			g2.draw(new Line2D.Double(xCoor, yPos, xCoor+tickLength, yPos));
			g2.draw(new Line2D.Double(xCoor+xAxisLength, yPos, xCoor+xAxisLength-tickLength, yPos));
		
			tick = Float.toString(yAxisTicks[j]);
			
			// get rid of superfluous 0
			int tickStrLength;
			while (tick.endsWith("0")){
				tickStrLength = tick.length();
				tick = tick.substring(0, tickStrLength-1);
			}
			
			tickStrLength = tick.length();
			if (tick.endsWith("."))
				tick = tick.substring(0,tickStrLength-1);				
						
			tmpInt = g2.getFontMetrics(tickFont).stringWidth(tick);
			if (tmpInt > maxTickStringLength)
				maxTickStringLength = tmpInt;
				
			g2.drawString(tick, (int) (xCoor-tmpInt-3), (int) yPos+tickFont.getSize()/2);
			
			if (((magOfY > 3) || (magOfY < -2)) && (j==yTickNumber-1)){
				// draw "x 10"	
				tick = "x 10";			
				g2.drawString(tick, (int) xCoor, (int) yCoor-3);
				
				// draw magnitude
				Font tickSubscribeFont = new Font(null, Font.BOLD, 9);
				g2.setFont(tickSubscribeFont);
				g2.drawString(Integer.toString(magOfY), (int) (xCoor+g2.getFontMetrics(tickFont).stringWidth(tick)), (int) yCoor-9);
			}
		}
		
		// draw x label
		
		Font tmpFont = labelFont;
		int f = 1;
		while (g2.getFontMetrics(tmpFont).stringWidth(xLabel) >  xAxisLength){			
			tmpFont = new Font(null, Font.BOLD, (labelFont.getSize()-f));
			f++;			
		}
				
		g2.setFont(tmpFont);						
		g2.drawString(xLabel, (int) (xCoor+(xAxisLength-g2.getFontMetrics(tmpFont).stringWidth(xLabel))/2), (int) (yCoor+yAxisLength+tickFont.getSize()+15));
		
		// draw y label		
		
		// first decide font since the length of label is probably longer than y axis length
		
		tmpFont = labelFont;
		f = 1;
		while (g2.getFontMetrics(tmpFont).stringWidth(yLabel) >  yAxisLength){			
			tmpFont = new Font(null, Font.BOLD, (labelFont.getSize()-f));
			f++;			
		}
			
		g2.setFont(tmpFont);
		
		AffineTransform oldTransform = g2.getTransform();
		AffineTransform ct  = AffineTransform.getTranslateInstance(xCoor-maxTickStringLength-10, yCoor+yAxisLength-(yAxisLength-g2.getFontMetrics(tmpFont).stringWidth(yLabel))/2);
		g2.transform(ct);
		
		g2.transform(AffineTransform.getRotateInstance(Math.toRadians(-90)));		
		g2.drawString(yLabel, 0, 0);
		
		g2.setTransform(oldTransform);
		
		// draw title	
		tmpFont = titleFont;
		f = 1;
		while (g2.getFontMetrics(tmpFont).stringWidth(title) >  panelWidth - xCoor){			
			tmpFont = new Font(null, Font.BOLD, (titleFont.getSize()-f));
			f++;			
		}		
		
		g2.setFont(tmpFont);
		g2.drawString(title, (int) (xCoor+(panelWidth-xCoor-g2.getFontMetrics(tmpFont).stringWidth(title))/2), (int) yCoor-20);

		// draw curve
		Color curveColor;
		Line2D.Double aLine;
		double previousPointX=0.0, previousPointY=0.0;
		Shape previousGlyph = null, plotGlyph;
		
		for (int i =0; i<yValues.length; i++){
			xPos = xCoor + ((xValues[i]-xValues[0])*xAxisLength)/(xValues[xTickNumber-1]-xValues[0]);
			
			if ((magOfY > 3) || (magOfY < -2)){								
				yPos = yCoor + yAxisLength - (yValues[i]*yAxisLength)/(yAxisTicks[yTickNumber-1]*magOfYValue);
			}
			else
				yPos = yCoor + yAxisLength - (yValues[i]*yAxisLength)/yAxisTicks[yTickNumber-1];
			
			// set color for this curve
			curveColor = Color.blue;
					
			plotGlyph = new Rectangle.Double(xPos-2, yPos-2, 4, 4);
									
			if (i>0){
				g2.setColor(curveColor); 
				aLine = new Line2D.Double(previousPointX, previousPointY, xPos, yPos);				
				g2.draw(aLine);				
						
			
			//	fill glyph 
				g2.setColor(Color.white);
				g2.fill(previousGlyph);
			//	draw glyph 
				g2.setColor(curveColor); 
				g2.draw(previousGlyph);
			}
			 	
			if (i==yValues.length-1){
			//	fill glyph 
				g2.setColor(Color.white);
				g2.fill(plotGlyph);
			//	draw glyph 
				g2.setColor(curveColor); 
				g2.draw(plotGlyph);	
			}
			 	
			previousPointX = xPos;
			previousPointY = yPos;
			previousGlyph = plotGlyph;
		}

		// draw curve
		// Color curveColor;
		// Line2D.Double aLine;
		previousPointX=0.0; 
		previousPointY=0.0;
		previousGlyph = null; 
		plotGlyph=null;
		
		for (int i =0; i<yValues2.length; i++){
			xPos = xCoor + ((xValues[i]-xValues[0])*xAxisLength)/(xValues[xTickNumber-1]-xValues[0]);
			
			if ((magOfY > 3) || (magOfY < -2)){								
				yPos = yCoor + yAxisLength - (yValues2[i]*yAxisLength)/(yAxisTicks[yTickNumber-1]*magOfYValue);
			}
			else
				yPos = yCoor + yAxisLength - (yValues2[i]*yAxisLength)/yAxisTicks[yTickNumber-1];
			
			// set color for this curve
			curveColor = Color.red;
					
			plotGlyph = new Rectangle.Double(xPos-2, yPos-2, 4, 4);
									
			if (i>0){
				g2.setColor(curveColor); 
				aLine = new Line2D.Double(previousPointX, previousPointY, xPos, yPos);				
				g2.draw(aLine);				
						
			
			//	fill glyph 
				g2.setColor(Color.white);
				g2.fill(previousGlyph);
			//	draw glyph 
				g2.setColor(curveColor); 
				g2.draw(previousGlyph);
			}
			 	
			if (i==yValues2.length-1){
			//	fill glyph 
				g2.setColor(Color.white);
				g2.fill(plotGlyph);
			//	draw glyph 
				g2.setColor(curveColor); 
				g2.draw(plotGlyph);	
			}
			 	
			previousPointX = xPos;
			previousPointY = yPos;
			previousGlyph = plotGlyph;
		}


		// draw curve
		// Color curveColor;
		// Line2D.Double aLine;
		previousPointX=0.0; 
		previousPointY=0.0;
		previousGlyph = null; 
		plotGlyph=null;
		
		for (int i =0; i<yValues3.length; i++){
			xPos = xCoor + ((xValues[i]-xValues[0])*xAxisLength)/(xValues[xTickNumber-1]-xValues[0]);
			
			if ((magOfY > 3) || (magOfY < -2)){								
				yPos = yCoor + yAxisLength - (yValues3[i]*yAxisLength)/(yAxisTicks[yTickNumber-1]*magOfYValue);
			}
			else
				yPos = yCoor + yAxisLength - (yValues3[i]*yAxisLength)/yAxisTicks[yTickNumber-1];
			
			// set color for this curve
			curveColor = Color.green;
					
			plotGlyph = new Rectangle.Double(xPos-2, yPos-2, 4, 4);
									
			if (i>0){
				g2.setColor(curveColor); 
				aLine = new Line2D.Double(previousPointX, previousPointY, xPos, yPos);				
				g2.draw(aLine);				
						
			
			//	fill glyph 
				g2.setColor(Color.white);
				g2.fill(previousGlyph);
			//	draw glyph 
				g2.setColor(curveColor); 
				g2.draw(previousGlyph);
			}
			 	
			if (i==yValues3.length-1){
			//	fill glyph 
				g2.setColor(Color.white);
				g2.fill(plotGlyph);
			//	draw glyph 
				g2.setColor(curveColor); 
				g2.draw(plotGlyph);	
			}
			 	
			previousPointX = xPos;
			previousPointY = yPos;
			previousGlyph = plotGlyph;
		}

	}

}
