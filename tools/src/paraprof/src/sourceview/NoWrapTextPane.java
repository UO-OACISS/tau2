package edu.uoregon.tau.paraprof.sourceview;

import java.awt.Dimension;

import javax.swing.JTextPane;
import javax.swing.text.EditorKit;

public class NoWrapTextPane extends JTextPane {

    /**
	 * 
	 */
	private static final long serialVersionUID = 4797000387881016553L;

	public boolean getScrollableTracksViewportWidth() {
        //should not allow text to be wrapped
        return false;
    }

    public void setSize(Dimension d) {
        //  dont let the Textpane get sized smaller than its parent
        Dimension pSize = getParent().getSize();
        super.setSize(pSize.width, d.height);
        if (d.width < pSize.width) {
            super.setSize(pSize.width, d.height);
        } else {
            super.setSize(d);
        }
    }

    protected EditorKit createDefaultEditorKit() {
        return new NoWrapEditorKit();
    }

}
