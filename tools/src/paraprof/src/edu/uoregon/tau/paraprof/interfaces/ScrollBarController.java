package edu.uoregon.tau.paraprof.interfaces;

import java.awt.Dimension;

public interface ScrollBarController {
    public void setVerticalScrollBarPosition(int position);
    public void setHorizontalScrollBarPosition(int position);


    public Dimension getThisViewportSize();
}
