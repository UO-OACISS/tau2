package edu.uoregon.tau.paraprof.interfaces;

import java.awt.*;

public interface ImageExport {

    public void export(Graphics2D g2D, boolean toScreen, boolean fullWindow, boolean drawHeader);

    public Dimension getImageSize(boolean fullScreen, boolean header);

}
