package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.image.BufferedImage;

/**
 * Interface for 3D image windows to output graphics (save bmp)
 * 
 * @author amorris
 */
public interface ThreeDeeImageProvider {

    public BufferedImage getImage();

    public Component getComponent();

}
