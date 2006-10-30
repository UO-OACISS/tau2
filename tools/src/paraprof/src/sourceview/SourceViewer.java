package edu.uoregon.tau.paraprof.sourceview;

import java.awt.Font;
import java.awt.Rectangle;
import java.io.File;
import java.net.URL;

import javax.swing.JEditorPane;
import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.SwingUtilities;
import javax.swing.text.BadLocationException;
import javax.swing.text.Element;

import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.perfdmf.SourceRegion;

public class SourceViewer extends JFrame {

    JScrollPane scrollpane;
    JEditorPane ed = null;

    public int getLineStartOffset(JEditorPane ed, int line) throws BadLocationException {
        Element map = ed.getDocument().getDefaultRootElement();
        Element lineElem = map.getElement(line);
        if (lineElem == null) {
        	return -1;
        }
        return lineElem.getStartOffset();
    }

    public int getLineEndOffset(JEditorPane ed, int line) throws BadLocationException {

        Element map = ed.getDocument().getDefaultRootElement();
        Element lineElem = map.getElement(line);
        int endOffset = lineElem.getEndOffset();
        return endOffset;
        // hide the implicit break at the end of the document
        //return ((line == lineCount - 1) ? (endOffset - 1) : endOffset);
    }

    
    public SourceViewer(File file, SourceRegion region) {
//        String file = "/home/amorris/apps/NPB3.1-MPI/LU/erhs.f";
        this.setTitle("Source Browser: " + file);
        this.setSize(800, 1000);

        try {
        	
        	URL url = new URL("file",null,file.getAbsolutePath());
            final JEditorPane ed = new JEditorPane();
            ed.setPage(url);
            ed.setFont(new Font("Monospaced", ParaProf.preferencesWindow.getFontStyle(), ParaProf.preferencesWindow.getFontSize()));

//            final int startpos = getLineStartOffset(ed, 118 - 1);
//            final int endpos = getLineEndOffset(ed, 227);
//            final int bonus20 = getLineStartOffset(ed, 118 - 1 + 50);
            final int startpos = getLineStartOffset(ed, region.getStartLine()-1);
            final int endpos = getLineEndOffset(ed, region.getEndLine()-1);
            int tempvar = getLineStartOffset(ed, region.getStartLine()+20);
            if (tempvar == -1) {
            	tempvar=endpos;
            }
            
            final int bonus20 = tempvar;
            //final int endpos = getLineEndOffset(ed, 120);
            //System.out.println("startpos = " + startpos);
            //System.out.println("endpos = " + endpos);

            //ed.setSelectionStart(startpos);
            //ed.setSelectionEnd(endpos);
            //ed.setSelectionStart(startpos);
            ed.setCaretPosition(startpos);

            this.add(ed);
            scrollpane = new JScrollPane(ed);
            this.add(scrollpane);

            ed.setEditable(false);
            //ed.setCaretPosition(endpos);
            //ed.moveCaretPosition(startpos);
            ed.setSelectionStart(startpos);
            ed.setSelectionEnd(bonus20);

            SwingUtilities.invokeLater(new Runnable() {
                public void run() {
                    Rectangle prevRect = ed.getVisibleRect();
                    //int value = scrollpane.getVerticalScrollBar().getValue();
                    //System.out.println("value = " + value);

                    ed.setCaretPosition(endpos);
                    ed.moveCaretPosition(startpos);

                    ed.scrollRectToVisible(prevRect);
                }
            });

        } catch (Throwable t) {
            t.printStackTrace();

        }

    }
}
