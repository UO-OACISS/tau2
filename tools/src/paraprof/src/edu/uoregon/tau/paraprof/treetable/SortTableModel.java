package edu.uoregon.tau.paraprof.treetable;

import javax.swing.table.TableModel;

/**
 * Extends the TableModel to provide sorting
 *
 * <P>CVS $Id: SortTableModel.java,v 1.1 2005/05/31 23:21:51 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public interface SortTableModel extends TableModel {
    public boolean isSortable(int col);
    public void sortColumn(int col, boolean ascending);
    public void updateTreeTable();
}
