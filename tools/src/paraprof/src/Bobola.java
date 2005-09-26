/*
 * Created on Apr 18, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package edu.uoregon.tau.paraprof;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.*;

/**
 * @author amorris
 *
 * TODO ...
 */
public class Bobola {

    private static class BobFrame extends JFrame {

        public BobFrame() {

            JFrame frame = this;
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            JPanel panel = new JPanel();
            JScrollPane scrollPane = new JScrollPane(panel);

            frame.getContentPane().add(scrollPane);

            frame.addKeyListener(new KeyListener() {

                public void keyPressed(KeyEvent e) {
                    System.err.println(e);
                    // TODO Auto-generated method stub

                }

                public void keyReleased(KeyEvent e) {
                    // TODO Auto-generated method stub
                    System.err.println(e);

                }

                public void keyTyped(KeyEvent e) {
                    // TODO Auto-generated method stub
                    System.err.println(e);

                }

            });

            frame.setSize(500, 500);
            frame.setVisible(true);
        }

    }

    public static void main(String[] args) {

        //Schedule a job for the event-dispatching thread:
        //creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                new BobFrame().show();
            }
        });

    }
}
