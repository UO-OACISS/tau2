package edu.uoregon.tau.common.treetable;

import java.awt.Component;

import javax.swing.Icon;
import javax.swing.JTable;
import javax.swing.UIManager;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.JTableHeader;


/***
 * Renders a small arrow next to the title in the column header<br/>
 * <br/>
 * Original code by Claude Duguay for "Java Pro"<br/>
 * JSortTable can be found here:<br/>
 * <a href="http://www.fawcette.com/javapro/2002_08/magazine/columns/visualcomponents/default_pf.aspx">
 * http://www.fawcette.com/javapro/2002_08/magazine/columns/visualcomponents/default_pf.aspx
 * </a><br/>
 * <br/>
 * - Changed to make the 'sortedColumnIndex' reflect the model's index
 *   instead of the 'visible' index. When you dragged a column, the arrow
 *   stuck with the 'x-th' column, even though you dragged it to another place.
 *   Now the renderer compares the index of the column in the model and it
 *   'sticks' to the column being dragged.<br/>
 * 
 * @author Claude Duguay
 * @author Vincent Vollers
 */
public class SortHeaderRenderer extends DefaultTableCellRenderer {
    /**
	 * 
	 */
	private static final long serialVersionUID = -7236758094424816775L;

	/***
     * The icon which is put next to an unsorted column (empty)
     */
    public static Icon NONSORTED = new ArrowIcon(ArrowIcon.NONE);

    /***
     * The icon which is put next to a column sorted ascending
     */
    public static Icon ASCENDING = new ArrowIcon(ArrowIcon.UP);

    /***
     * The icon which is put next to a column sorted descending
     */
    public static Icon DESCENDING = new ArrowIcon(ArrowIcon.DOWN);

    /***
     * Creates a new SortTreeRenderer, basicly sets up the text-alignment.
     */
    public SortHeaderRenderer() {
        setHorizontalTextPosition(LEFT);
        setHorizontalAlignment(CENTER);
    }

    /***
     * Updates the component to reflect the current header
     */
    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus,
            int row, int col) {

        // we'll be using these to determine if the current header
        // is the header by which everything is sorted
        int index = -1;

        // we'll use the modelIndex instead of 'col', since 'col' is the
        // index of the column on screen and 'modelCol' is the index of
        // the column in the model.
        int modelCol = table.getColumnModel().getColumn(col).getModelIndex();

        boolean ascending = true;

        // I would be surprised if you use this renderer with a non-JSortTable
        // but anyway, this code gets the current selected column-index and order.
        if (table instanceof JTreeTable) {
            JTreeTable sortTable = (JTreeTable) table;
            index = sortTable.getSortedColumnIndex();
            ascending = sortTable.getSortedColumnAscending();
        }

        // set the header font and colors
        if (table != null) {
            JTableHeader header = table.getTableHeader();
            if (header != null) {
                setForeground(header.getForeground());
                setBackground(header.getBackground());
                setFont(header.getFont());
            }
        }

        // set the icon for this column (notice: modelCol is the
        // index of the column in the model for the current column,
        // 'index' is the selected column)
        Icon icon = ascending ? ASCENDING : DESCENDING;
        setIcon(modelCol == index ? icon : NONSORTED);

        // set the text of the header
        setText((value == null) ? "" : value.toString());

        // set the border of the header
        setBorder(UIManager.getBorder("TableHeader.cellBorder"));

        return this;
    }
}
