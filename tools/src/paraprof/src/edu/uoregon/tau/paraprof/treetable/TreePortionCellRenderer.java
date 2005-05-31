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
 * <P>CVS $Id: TreePortionCellRenderer.java,v 1.1 2005/05/31 23:21:51 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public class TreePortionCellRenderer extends DefaultTreeCellRenderer {

    private static ColorIcon colorIcon = new ColorIcon();
    
    public Component getTreeCellRendererComponent(JTree tree, Object value, boolean selected, boolean expanded,
            boolean leaf, int row, boolean hasFocus) {
        super.getTreeCellRendererComponent(tree, value, selected, expanded, leaf, row, hasFocus);

        // set the icon
        if (value instanceof TreeTableNode) {
            TreeTableNode node = (TreeTableNode) value;
            if (node.getModel().getPPTrial().getNumberOfMetrics() == 1) {
                colorIcon.setColor(node.getColor(0));
                if (node.getColor(0) != null) {
                    this.setIcon(colorIcon);
                } else {
                    this.setIcon(null);
                }
            } else {
                this.setIcon(null);
            }
        }

        // shade every other row
        setBackgroundNonSelectionColor(null);
        if (row % 2 == 0) {
            setBackgroundNonSelectionColor(new Color(235,235,235));
        } else {
            this.setBackground(tree.getBackground());
        }

        return this;
    }

}
