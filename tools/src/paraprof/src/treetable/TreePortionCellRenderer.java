package edu.uoregon.tau.paraprof.treetable;

import java.awt.Color;
import java.awt.Component;

import javax.swing.JTree;
import javax.swing.tree.DefaultTreeCellRenderer;

import edu.uoregon.tau.paraprof.treetable.TreeTableColumn.ColorIcon;

/**
 * Renderer for the tree portion of the treetable
 *    
 * TODO : ...
 *
 * <P>CVS $Id: TreePortionCellRenderer.java,v 1.2 2006/12/28 03:14:42 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class TreePortionCellRenderer extends DefaultTreeCellRenderer {

    /**
	 * 
	 */
	private static final long serialVersionUID = 7914281444748384284L;
	private static ColorIcon colorIcon = new ColorIcon();

    public Component getTreeCellRendererComponent(JTree tree, Object value, boolean selected, boolean expanded, boolean leaf,
            int row, boolean hasFocus) {
        super.getTreeCellRendererComponent(tree, value, selected, expanded, leaf, row, hasFocus);

        // set the icon
        if (value instanceof TreeTableNode) {
            TreeTableNode node = (TreeTableNode) value;
            colorIcon.setColor(node.getColor(node.getModel().getColorMetric()));
            if (node.getColor(0) != null) {
                this.setIcon(colorIcon);
            } else {
                this.setIcon(null);
            }
        }

        // shade every other row
        setBackgroundNonSelectionColor(null);
        if (row % 2 == 0) {
            setBackgroundNonSelectionColor(new Color(235, 235, 235));
        } else {
            this.setBackground(tree.getBackground());
        }

        return this;
    }

}
