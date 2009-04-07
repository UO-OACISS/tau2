package edu.uoregon.tau.paraprof.interfaces;

import javax.swing.JFrame;


public interface ParaProfWindow {
    public JFrame getFrame();
    public void help(boolean display);
    public void closeThisWindow();
}
