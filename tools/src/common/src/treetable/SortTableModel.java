package edu.uoregon.tau.common.treetable;

import javax.swing.table.TableModel;

/**
 * Extends the TableModel to provide sorting
 *
 * <P>CVS $Id: SortTableModel.java,v 1.1 2007/01/31 22:18:15 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public interface SortTableModel extends TableModel {
    public boolean isSortable(int col);
    public void sortColumn(int col, boolean ascending);
    public void updateTreeTable();
}
