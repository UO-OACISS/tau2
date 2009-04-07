package edu.uoregon.tau.paraprof.sourceview;

import java.awt.BorderLayout;
import java.awt.Font;
import java.awt.Rectangle;
import java.io.File;
import java.net.URL;

import javax.swing.*;
import javax.swing.text.BadLocationException;
import javax.swing.text.Element;

import edu.uoregon.tau.paraprof.ParaProf;
import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.paraprof.interfaces.ParaProfWindow;
import edu.uoregon.tau.perfdmf.SourceRegion;

public class SourceViewer extends JFrame implements ParaProfWindow {

    LineNumberedTextPanel textPanel;
    JTextPane ed = null;

    public int getLineStartOffset(JEditorPane ed, int line) throws BadLocationException {
        Element map = ed.getDocument().getDefaultRootElement();
        Element lineElem = map.getElement(line);
        if (lineElem == null) {
            return -1;
        }
        return lineElem.getStartOffset();
    }

    public int getLineEndOffset(JTextPane ed, int line) throws BadLocationException {

        Element map = ed.getDocument().getDefaultRootElement();
        Element lineElem = map.getElement(line);
        if (lineElem == null) {
            return -1;
        }
        int endOffset = lineElem.getEndOffset();
        return endOffset;
        // hide the implicit break at the end of the document
        //return ((line == lineCount - 1) ? (endOffset - 1) : endOffset);
    }

    public void highlightRegion(SourceRegion region) {
        try {
            //      final int startpos = getLineStartOffset(ed, 118 - 1);
            //      final int endpos = getLineEndOffset(ed, 227);
            //      final int bonus20 = getLineStartOffset(ed, 118 - 1 + 50);
            final int startpos = Math.max(0, getLineStartOffset(ed, Math.max(0, region.getStartLine() - 1)));
            final int endpos = Math.max(0, getLineEndOffset(ed, Math.max(0, region.getEndLine() - 1)));
            int tempvar = getLineStartOffset(ed, region.getStartLine() + 50);
            if (tempvar == -1) {
                tempvar = endpos;
            }

            final int bonus20 = tempvar;
            //final int endpos = getLineEndOffset(ed, 120);
            //System.out.println("startpos = " + startpos);
            //System.out.println("endpos = " + endpos);

            //ed.setSelectionStart(startpos);
            //ed.setSelectionEnd(endpos);
            //ed.setSelectionStart(startpos);
            ed.setCaretPosition(startpos);

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

        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }
    }

    private void setupMenus() {
        JMenuBar mainMenu = new JMenuBar();
        mainMenu.add(ParaProfUtils.createFileMenu((ParaProfWindow) this, textPanel, textPanel));
        //mainMenu.add(ParaProfUtils.createWindowsMenu(ppTrial, this));
        mainMenu.add(ParaProfUtils.createHelpMenu(this, this));

        setJMenuBar(mainMenu);

    }

    public SourceViewer(File file) {
        //        String file = "/home/amorris/apps/NPB3.1-MPI/LU/erhs.f";
        this.setTitle("TAU: ParaProf: Source Browser: " + file);
        ParaProfUtils.setFrameIcon(this);

        try {

            URL url = new URL("file", null, file.getAbsolutePath());
            textPanel = new LineNumberedTextPanel();

            ed = textPanel.getJTextPane();
            ed.setPage(url);
            ed.setFont(new Font("Monospaced", ParaProf.preferencesWindow.getFontStyle(), ParaProf.preferencesWindow.getFontSize()));

            getContentPane().add(textPanel, BorderLayout.WEST);
            getContentPane().add(textPanel.scrollPane, BorderLayout.CENTER);
            pack();

            setupMenus();

            this.setSize(700, 1000);

        } catch (Exception e) {
            ParaProfUtils.handleException(e);
        }

    }

    public void closeThisWindow() {
        setVisible(false);
        ParaProf.decrementNumWindows();
        dispose();
    }

    public void help(boolean display) {
        ParaProf.getHelpWindow().clearText();
        if (display) {
            ParaProf.getHelpWindow().setVisible(true);
        }
        ParaProf.getHelpWindow().writeText("This is the Source Viewer.\n");
        ParaProf.getHelpWindow().writeText(
                "When you right click on a timer with source location information, you can display it here.");
    }

    public JFrame getFrame() {
        return this;
    }

}
