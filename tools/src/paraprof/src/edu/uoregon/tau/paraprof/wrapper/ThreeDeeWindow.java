package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyListener;
import java.awt.image.BufferedImage;
import java.awt.image.ImageObserver;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.util.*;
import javax.swing.*;


public class ThreeDeeWindow extends JFrame {

    public ThreeDeeWindow(ParaProfTrial ppTrial) {
	JOptionPane.showMessageDialog(this, "3D Display not supported with Java 1.3, install 1.4 or higher and reconfigure TAU");
    }
}
