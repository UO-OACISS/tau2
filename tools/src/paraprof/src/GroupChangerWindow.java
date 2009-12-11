package edu.uoregon.tau.paraprof;

import java.awt.Component;
import java.awt.Graphics;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;

import javax.swing.JFrame;

import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;

public class GroupChangerWindow extends JFrame implements ParaProfWindow, ActionListener, Printable {

    private GroupChangerWindow() {
        
    }
    
    static GroupChangerWindow createGroupChangerWindow() {
        GroupChangerWindow gcw = new GroupChangerWindow();
        return gcw;
    }
    
    @Override
    public void closeThisWindow() {
        // TODO Auto-generated method stub
        
    }

    @Override
    public JFrame getFrame() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public void help(boolean display) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public void actionPerformed(ActionEvent arg0) {
        // TODO Auto-generated method stub
        
    }

    @Override
    public int print(Graphics arg0, PageFormat arg1, int arg2) throws PrinterException {
        // TODO Auto-generated method stub
        return 0;
    }

}
