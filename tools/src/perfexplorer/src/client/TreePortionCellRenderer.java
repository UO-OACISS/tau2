package edu.uoregon.tau.perfexplorer.client;

import java.awt.Color;
import java.awt.Component;
import javax.swing.JTree;
import javax.swing.tree.DefaultTreeCellRenderer;

import org.jfree.chart.ChartColor;

/**
 * Renderer for the tree portion of the treetable
 *    
 * TODO : ...
 *
 * <P>CVS $Id: TreePortionCellRenderer.java,v 1.3 2009/02/24 00:53:34 khuck Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.3 $
 */
public class TreePortionCellRenderer extends DefaultTreeCellRenderer {

    public Component getTreeCellRendererComponent(JTree tree, Object value, boolean selected, boolean expanded, boolean leaf,
            int row, boolean hasFocus) {
        super.getTreeCellRendererComponent(tree, value, selected, expanded, leaf, row, hasFocus);

        // set the text color?
        if (value instanceof XMLTAUAttributeElementNode) {
            this.setForeground(ChartColor.VERY_DARK_GREEN);
        } else if (value instanceof XMLElementNode) {
            this.setForeground(ChartColor.DARK_BLUE);
        } else if (value instanceof XMLCommentNode) {
            this.setForeground(ChartColor.VERY_DARK_RED);
        } else if (value instanceof XMLAttributeNode) {
            this.setForeground(ChartColor.VERY_DARK_GREEN);
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
