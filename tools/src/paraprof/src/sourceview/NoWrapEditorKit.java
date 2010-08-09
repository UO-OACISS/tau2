package edu.uoregon.tau.paraprof.sourceview;

import javax.swing.text.AbstractDocument;
import javax.swing.text.BoxView;
import javax.swing.text.ComponentView;
import javax.swing.text.Element;
import javax.swing.text.IconView;
import javax.swing.text.LabelView;
import javax.swing.text.ParagraphView;
import javax.swing.text.StyleConstants;
import javax.swing.text.StyledEditorKit;
import javax.swing.text.TabExpander;
import javax.swing.text.View;
import javax.swing.text.ViewFactory;
public class NoWrapEditorKit extends StyledEditorKit {
    /**
	 * 
	 */
	private static final long serialVersionUID = 2502198775876837363L;
	private static final ViewFactory defaultFactory = new MyViewFactory();

    public ViewFactory getViewFactory() {
        return defaultFactory;
    }

    public static class MyViewFactory implements ViewFactory {
        public View create(Element elem) {
            String kind = elem.getName();
            if (kind != null) {
                if (kind.equals(AbstractDocument.ContentElementName)) {
                    return new NoWrapView(elem);
                } else if (kind.equals(AbstractDocument.ParagraphElementName)) {
                    return new ParagraphView(elem);
                } else if (kind.equals(AbstractDocument.SectionElementName)) {
                    return new BoxView(elem, View.Y_AXIS);
                } else if (kind.equals(StyleConstants.ComponentElementName)) {
                    return new ComponentView(elem);
                } else if (kind.equals(StyleConstants.IconElementName)) {
                    return new IconView(elem);
                }
            }

            // default to text display
            return new NoWrapView(elem);
        }
    }

    public static class NoWrapView extends LabelView {
        public NoWrapView(Element e) {
            super(e);
        }

        public float getTabbedSpan(float x, TabExpander e) {
            float result = super.getTabbedSpan(x, e);
            this.preferenceChanged(this, true, false);
            return result;
        }

    }
}