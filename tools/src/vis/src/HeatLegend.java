package edu.uoregon.tau.vis;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Insets;
import java.awt.image.BufferedImage;
import java.awt.image.ImageObserver;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.JPanel;

public class HeatLegend extends JPanel implements ImageObserver {

	int[] pixels = null;
	BufferedImage img = null;
	private static final ColorScale scale = new ColorScale(ColorScale.ColorSet.RAINBOW);

	public HeatLegend () {
	
		// build the image data from the cluster results
		pixels = new int[1000];
		
		// get the size of the image...
		int width = 10;
		int height = 100;
		
		img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		int i = 0;
		for (int x = 0 ; x < width ; x++) {
			for (int y = 0 ; y < height ; y++) {
				img.setRGB(x, y, scale.getColor(((float)(99-y))/100.0f).getRGB());
				i++;
			}
		}
		this.setPreferredSize(new Dimension(10,100));
        this.setSize(10,100);
	}
		
	public void paint(Graphics g) {
		super.paint(g);
		// the size of the component
		Dimension d = getSize();
		// the internal margins of the component
		Insets i = getInsets();
		// draw to fill the entire component
		g.drawImage(img, i.left, i.top, d.width - i.left - i.right, d.height - i.top - i.bottom, this );
		//g.drawImage(img, i.left, i.top, 512, 512, this );
	}

}
