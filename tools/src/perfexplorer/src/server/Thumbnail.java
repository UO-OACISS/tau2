package edu.uoregon.tau.perfexplorer.server;


import java.awt.Container;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.MediaTracker;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;

import com.sun.image.codec.jpeg.JPEGCodec;
import com.sun.image.codec.jpeg.JPEGEncodeParam;
import com.sun.image.codec.jpeg.JPEGImageEncoder;

/**
 * This code generates a thumbnail image from a larger JPG or PNG file.
 * This code was taken from http://www.geocities.com/marcoschmidt.geo/java-save-jpeg-thumbnail.html
 *
 * <P>CVS $Id: Thumbnail.java,v 1.4 2009/02/24 00:53:45 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
public class Thumbnail {
	/**
	 * Static method to create a thumbnail from an input image.
	 * 
	 * @param infile
	 * @param outfile
	 * @param big
	 */
	public static void createThumbnail(String infile, String outfile, boolean big) {
	  	try {
			//PerfExplorerOutput.print("Making thumbnail... ");
		    // load image from INFILE
		    Image image = Toolkit.getDefaultToolkit().getImage(infile);
		    MediaTracker mediaTracker = new MediaTracker(new Container());
		    mediaTracker.addImage(image, 0);
		    mediaTracker.waitForID(0);
		    // determine thumbnail size from WIDTH and HEIGHT
		    int thumbWidth = 100;
		    int thumbHeight = 75;
			if (big) {
			    	thumbWidth = 200;
			    	thumbHeight = 150;
			}
		    double thumbRatio = (double)thumbWidth / (double)thumbHeight;
		    int imageWidth = image.getWidth(null);
		    int imageHeight = image.getHeight(null);
		    double imageRatio = (double)imageWidth / (double)imageHeight;
		    if (thumbRatio < imageRatio) {
		    		thumbHeight = (int)(thumbWidth / imageRatio);
		    } else {
		    		thumbWidth = (int)(thumbHeight * imageRatio);
		    }
			// draw original image to thumbnail image object and
			// scale it to the new size on-the-fly
			BufferedImage thumbImage = new BufferedImage(thumbWidth, thumbHeight, BufferedImage.TYPE_INT_RGB);
			Graphics2D graphics2D = thumbImage.createGraphics();
			graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			graphics2D.drawImage(image, 0, 0, thumbWidth, thumbHeight, null);
			// save thumbnail image to OUTFILE
			BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(outfile));
			JPEGImageEncoder encoder = JPEGCodec.createJPEGEncoder(out);
			JPEGEncodeParam param = encoder.getDefaultJPEGEncodeParam(thumbImage);
			int quality = 75;
			quality = Math.max(0, Math.min(quality, 100));
			param.setQuality((float)quality / 100.0f, false);
			encoder.setJPEGEncodeParam(param);
			encoder.encode(thumbImage);
			out.close(); 
			//PerfExplorerOutput.println("Done.");
		} catch (Exception e) {
			System.err.println("\nError making Thumbnail!");
			System.err.println(e.getMessage());
			e.printStackTrace();
		}
	}
}
