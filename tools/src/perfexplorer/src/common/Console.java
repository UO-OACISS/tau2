
package edu.uoregon.tau.perfexplorer.common;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.io.IOException;
import java.io.PrintStream;
import java.net.URL;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTextPane;
import javax.swing.text.AttributeSet;
import javax.swing.text.BadLocationException;
import javax.swing.text.Document;
import javax.swing.text.SimpleAttributeSet;
import javax.swing.text.StyleConstants;

import edu.uoregon.tau.common.Utility;
    
    public class Console extends JFrame {
        /**
		 * 
		 */
		private static final long serialVersionUID = -592613353230989839L;
		JTextPane textPane = new JTextPane();
		private PrintStream out = null;
		private PrintStream err = null;
//		private PrintStream oldOut = null;
//		private PrintStream oldErr = null;
		private Document doc = null;
		private SimpleAttributeSet errorStyle = null;
		private SimpleAttributeSet outputStyle = null;
    
        public Console() throws IOException {
			super("PerfExplorer Console");

			// preserve the old out and err
//			oldOut = System.out;
//			oldErr = System.err;

            // Set up System.out
            ConsoleOutputStream consoleOut = new ConsoleOutputStream(this, false);
            out = new PrintStream(consoleOut, true);
            System.setOut(out);
    
            // Set up System.err
            ConsoleOutputStream consoleErr = new ConsoleOutputStream(this, true);
            err = new PrintStream(consoleErr, true);
            System.setErr(err);
    
            // Add a scrolling text area
            textPane.setEditable(false);
            this.getContentPane().add(new JScrollPane(textPane), BorderLayout.CENTER);
			this.setPreferredSize(new Dimension(800,600));
            this.pack();
            this.setVisible(true);

        	URL url = Utility.getResource("tau32x32.gif");
			if (url != null) 
				this.setIconImage(Toolkit.getDefaultToolkit().getImage(url));

			doc = textPane.getDocument();
			errorStyle = new SimpleAttributeSet();
			StyleConstants.setForeground(errorStyle, Color.red);
			outputStyle = new SimpleAttributeSet();
			StyleConstants.setForeground(outputStyle, Color.black);
        }
    
		public void print(boolean error, String record) {
			AttributeSet attributes = outputStyle;
			if (error) {
				attributes = errorStyle;
			}
			try {
				doc.insertString(doc.getLength(), record, attributes);
			} catch (BadLocationException exp) {
		        exp.printStackTrace();
			}

			// Make sure the last line is always visible
			textPane.setCaretPosition(textPane.getDocument().getLength());
    
		}
    }