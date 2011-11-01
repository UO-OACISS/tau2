package edu.uoregon.tau.paraprof.sourceview;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Point;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextPane;
import javax.swing.text.Document;

/**
 * A class illustrating running line number count on JTextPane. Nothing
 is painted on the pane itself,
 * but a separate JPanel handles painting the line numbers.<br>
 *
 * @author Daniel Sjoblom<br>
 * Created on Mar 3, 2004<br>
 * Copyright (c) 2004<br>
 * @version 1.0<br>
 */
public class LineNumberedTextPanel extends JPanel  {
    /**
	 * 
	 */
	private static final long serialVersionUID = -6759346836912947146L;
	// for this simple experiment, we keep the pane + scrollpane as members.
    JTextPane pane;
    JScrollPane scrollPane;

    public LineNumberedTextPanel() {
        super();
        setMinimumSize(new Dimension(60, 30));
        setPreferredSize(new Dimension(60, 30));
        setMinimumSize(new Dimension(60, 30));
        pane = new NoWrapTextPane() {
            /**
			 * 
			 */
			private static final long serialVersionUID = -6548022852077257728L;

			// we need to override paint so that the linenumbers stay in sync
            public void paint(Graphics g) {
                super.paint(g);
                LineNumberedTextPanel.this.repaint();
            }
        };
        scrollPane = new JScrollPane(pane);
    }

    public JTextPane getJTextPane() {
        return pane;
    }

    public JScrollPane getJScrollPane() {
        return scrollPane;
    }

    public void paint(Graphics g) {
        super.paint(g);
        // We need to properly convert the points to match the viewport
        // Read docs for viewport
        int start = pane.viewToModel(scrollPane.getViewport().getViewPosition()); //starting pos in document
        int end = pane.viewToModel(new Point(scrollPane.getViewport().getViewPosition().x + pane.getWidth(),
                scrollPane.getViewport().getViewPosition().y + pane.getHeight()));
        // end pos in doc

        // translate offsets to lines
        Document doc = pane.getDocument();
        int startline = doc.getDefaultRootElement().getElementIndex(start);
        int endline = doc.getDefaultRootElement().getElementIndex(end);

        int fontHeight = g.getFontMetrics(pane.getFont()).getHeight(); // fontheight

        for (int line = startline, y = 0; line <= endline; line++, y += fontHeight) {
            g.drawString(Integer.toString(line), 0, y);
        }

    }

    // test main
    public static void main(String[] args) {
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().setLayout(new BorderLayout());
        final LineNumberedTextPanel nr = new LineNumberedTextPanel();
        frame.getContentPane().add(nr, BorderLayout.WEST);
        frame.getContentPane().add(nr.scrollPane, BorderLayout.CENTER);
        frame.pack();
        frame.setSize(new Dimension(400, 400));
        frame.setVisible(true);
    }

  
}