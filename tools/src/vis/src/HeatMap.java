package edu.uoregon.tau.vis;

import java.awt.*;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.text.DecimalFormat;

public class HeatMap extends JPanel implements ImageObserver {

	BufferedImage img = null;
	StringBuffer description = null;
	private static final int idealSize = 128;
	private static final ColorScale scale = new ColorScale(ColorScale.ColorSet.RAINBOW);
	public static final String TMPDIR = System.getProperty("user.home") + File.separator + ".ParaProf" + File.separator + "tmp" + File.separator;
	private HeatMapScanner scanner = null; // for tool tips
	private double[][] map = null;
	private HeatMapData mapData = null;
	private DecimalFormat f = new DecimalFormat("0");
	private int size = 128;
	private int index = 0;
	private String path = "";


	public HeatMap (HeatMapData mapData, int index, String path, String description) {
		this.mapData = mapData;
		this.size = mapData.getSize();
		this.description = new StringBuffer();
		this.description.append(description);
		this.index = index;
		this.path = path;
		double max = mapData.getMax(path, index);
		double min = mapData.getMin(path, index);
		double range = max - min;
	
		// get the size of the image...
		int width = size;
		int height = size;
		
		if (img == null) {
			img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		}
		mapData.reset();
		while (mapData.hasNext()) {
			HeatMapData.NextValue next = (HeatMapData.NextValue)mapData.next();
			int x = next.receiver;
			int y = next.sender;
			double value = next.getValue(path, index);
			if (value > 0.0 && range == 0) {
				img.setRGB(x, y, scale.getColor(1f).getRGB());
			} else if (value > 0.0) {
				// this looks inverted, but it is so the sender is on the left, receiver on top
				img.setRGB(x, y, scale.getColor((float)((value-min)/range)).getRGB());
			}
		}
	
		if (HeatMapWindow.maxCells < size) {
			int factor = (size / HeatMapWindow.maxCells) + 1;
			int newSize = size * factor * HeatMapWindow.viewRatio;
			this.setPreferredSize(new Dimension(newSize,newSize));
			this.setSize(newSize,newSize);
		} else {
			this.setPreferredSize(new Dimension(size,size));
			this.setSize(size,size);
		}
		scanner = new HeatMapScanner(this, size);
		this.addMouseListener(scanner);
		this.addMouseMotionListener(scanner);
		this.addMouseMotionListener(scanner);
		this.addMouseWheelListener(scanner);
		this.setFocusable(true);  // enables key listener events
		this.addKeyListener(scanner);
	}

	public String getToolTip(Point p) {
		// adjust to zoom
    	int currentHeight = this.getSize().height;
    	int currentWidth = this.getSize().width;
    	double pixelsPerCell = (double)(Math.max(currentWidth, HeatMapWindow.viewSize)) / (double)size;
		// don't go past the end of the array
		int receiver = Math.min((int)((p.getX()) / pixelsPerCell),size-1);  
    	pixelsPerCell = (double)(Math.max(currentHeight, HeatMapWindow.viewSize)) / (double)size;
		// don't go past the end of the array
		int sender = Math.min((int)((p.getY()) / pixelsPerCell),size-1);  
		// this is inverted - the sender is Y, the receiver is X
		double value = mapData.get(sender, receiver, path, index);
		String s = "<html>sender = " + sender + "<BR>receiver = " + receiver + "<BR>value = " + f.format(value) + "</html>";
		return s;
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
	}

	/**
	 * @return the size
	 */
	public int getMapSize() {
		return size;
	}

	/**
	 * @return the scanner
	 */
	public HeatMapScanner getScanner() {
		return scanner;
	}

	public void clean() {
	}

	public void update(HeatMapData mapData, int index, String path, String filenamePrefix) {
		this.mapData = mapData;
		this.size = mapData.getSize();
		this.description = new StringBuffer();
		this.description.append(description);
		this.index = index;
		this.path = path;
		double max = mapData.getMax(path, index);
		double min = mapData.getMin(path, index);
		double range = max - min;
	
		// get the size of the image...
		int width = size;
		int height = size;
		
		img=null;
		if (img == null) {
			img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		}
		mapData.reset();
		while (mapData.hasNext()) {
			HeatMapData.NextValue next = (HeatMapData.NextValue)mapData.next();
			int x = next.receiver;
			int y = next.sender;
			double value = next.getValue(path, index);
			if (value > 0.0 && range == 0) {
				img.setRGB(x, y, scale.getColor(1f).getRGB());
			} else if (value > 0.0) {
				// this looks inverted, but it is so the sender is on the left, receiver on top
				img.setRGB(x, y, scale.getColor((float)((value-min)/range)).getRGB());
			}
		}
	}
	
	public void goAway() {
		this.img.flush();
	}

}
