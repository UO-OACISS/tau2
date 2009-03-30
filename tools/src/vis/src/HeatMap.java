package edu.uoregon.tau.vis;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Insets;
import java.awt.image.BufferedImage;
import java.awt.image.ImageObserver;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.JPanel;

public class HeatMap extends JPanel implements ImageObserver {

	int[] pixels = null;
	BufferedImage img = null;
	StringBuffer description = null;
	private static final int idealSize = 128;
	private static final ColorScale scale = new ColorScale(ColorScale.ColorSet.RAINBOW);
	public static final String TMPDIR = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + "tmp" + File.separator;

	public HeatMap (double[][] map, int size, double max, double min, String description) {
		this.description = new StringBuffer();
		this.description.append(description);
		double range = max - min;
	
		// build the image data from the cluster results
		pixels = new int[size*size];
		
		// get the size of the image...
		int width = size;
		int height = size;
		
		// if this is a BIG topology (larger than 128x128), just do one pixel per thread
//		if (idealSize <= width || idealSize <= height) {
			img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			int i = 0;
			for (int x = 0 ; x < width ; x++) {
				for (int y = 0 ; y < height ; y++) {
					if (map[x][y] > 0.0 && range == 0) {
						img.setRGB(x, y, scale.getColor(1f).getRGB());
					} else if (map[x][y] > 0.0) {
						img.setRGB(x, y, scale.getColor((float)((map[x][y]-min)/range)).getRGB());
					}
					i++;
				}
			}
/*		} else {
			img = new BufferedImage(idealSize, idealSize, BufferedImage.TYPE_INT_RGB);

			// otherwise, make each thread "bigger" (multiple pixels) in the final image.
			int cellWidth = idealSize / width;
			int cellHeight = idealSize / height;
			
			int i = 0;
			for (int x = 0 ; x < width ; x++) {
				for (int y = 0 ; y < height ; y++) {
					for (int cellX = x * cellWidth ; cellX < (x+1) * cellWidth ; cellX++) {
						for (int cellY = y * cellHeight ; cellY < (y+1) * cellHeight ; cellY++) {
							
							if (map[x][y] > 0.0 && range == 0) {
								img.setRGB(cellX, cellY, scale.getColor(1f).getRGB());
							} else if (map[x][y] > 0.0) {
								img.setRGB(cellX, cellY, scale.getColor((float)((map[x][y]-min)/range)).getRGB());
							}
						}
					}
					i++;
				}
			}
		}
*/		this.setPreferredSize(new Dimension(600,600));
        this.setSize(600,600);
	}
		
	public String getImage() {
		String filename = TMPDIR + "clusterImage." + description + ".png";
		File outFile = new File(filename);
		try {
			ImageIO.write(img, "PNG", outFile);
		} catch (IOException e) {
			String error = "ERROR: Couldn't write the virtual topology image!";
			System.err.println(error);
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
		return filename;
	}

	public String getThumbnail() {
		String filename = TMPDIR + "clusterImage.thumb." + description + ".png";
		return filename;
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
