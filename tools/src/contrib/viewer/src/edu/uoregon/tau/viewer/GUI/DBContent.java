/***** DBContent class extends AbstractTableModel. Performance data will be presented in a DBContent
object.
*****/
 
package edu.uoregon.tau.viewer.GUI;

import java.util.*;
import javax.swing.table.AbstractTableModel;
import javax.swing.JTable;
import javax.swing.table.JTableHeader;
import javax.swing.table.TableColumnModel;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import javax.swing.event.*;


public class DBContent extends AbstractTableModel {

    String[] columnNames = {};
    Vector rowsVec = null;
    Object[][] rows = null;
    int[] indexes;
	boolean ascending; // sort a column in ascending order.
	int column; 
	
    public DBContent(Object[][] rows, String[] columnNames ){
	 	super();
		this.rows = rows;
		this.columnNames = columnNames;
		initIndexes();	
    }

    public DBContent(Vector rowsVec, String[] columnNames){
    	super();
		this.rowsVec = rowsVec;
		this.columnNames = columnNames;
		initIndexes();
    }
    
    public void initIndexes(){
		int rowCount = getRowCount();
		indexes = new int[rowCount];	
		for (int row = 0; row < rowCount; row++) {
			indexes[row] = row;
		}
    }
    
    public String getColumnName(int column) {
	if (columnNames[column] != null) {
            return columnNames[column];
        } else{
            return "";
        }
    }    
    
    public boolean isCellEditable(int row, int column) {
        return true;
    }

    public int getColumnCount() {
        return columnNames.length;
    }
   
    public int getRowCount() {
	if (rows != null)
	    return rows.length;
	else if (rowsVec != null)
	    return rowsVec.size();
	else return 0;
    }

    public Object getValueAt(int aRow, int aColumn) {
	if (rows != null)
	    return rows[indexes[aRow]][aColumn];
	else if (rowsVec != null)
	    return ((Vector)(rowsVec.elementAt(indexes[aRow]))).elementAt(aColumn);
	else 
	    return null;
    }
	
	public void setValueAt(Object aValue, int aRow, int aColumn) {
		super.setValueAt(aValue, indexes[aRow], aColumn);
	}
	
	// clicking on a column will sort values in the column in descending order
	public void addMouseListenerToTableHeader(JTable table) { 
			final DBContent sorter = this; 
			final JTable tableView = table; 
			tableView.setColumnSelectionAllowed(false); 
			MouseAdapter listMouseListener = new MouseAdapter() {
				public void mouseClicked(MouseEvent e) {
					TableColumnModel columnModel = tableView.getColumnModel();
					int selectedColumn = columnModel.getColumnIndexAtX(e.getX()); 
					int column = tableView.convertColumnIndexToModel(selectedColumn); 
					if (e.getClickCount() == 1 && column != -1) {						
						boolean ascending = false; 
						sorter.sortByColumn(column, ascending); 
					}
				}
			};
			JTableHeader header = tableView.getTableHeader(); 
			header.addMouseListener(listMouseListener); 
	}
	
	public void sortByColumn(int col, boolean order){
		this.ascending = order;
		this.column = col;
		shuttlesort((int[])indexes.clone(), indexes, 0, indexes.length);
		fireTableChanged(new TableModelEvent(this));
	}	
	
	// shuttlesort 
	public void shuttlesort(int from[], int to[], int low, int high) {
			if (high - low < 2) {
				return;
			}
			int middle = (low + high)/2;
			shuttlesort(to, from, low, middle);
			shuttlesort(to, from, middle, high);

			int p = low;
			int q = middle;			

			if (high - low >= 4 && compare(from[middle-1], from[middle]) <= 0) {
				for (int i = low; i < high; i++) {
					to[i] = from[i];
				}
				return;
			}

			//merge. 

			for (int i = low; i < high; i++) {
				if (q >= high || (p < middle && compare(from[p], from[q]) <= 0)) {
					to[i] = from[p++];
				}
				else {
					to[i] = from[q++];
				}
			}
	}
	
	public int compare(int row1, int row2) {
					
			int result = compareRowsByColumn(row1, row2, column);
			if (result != 0) {
				return ascending ? result : -result;
			}
		
			return 0;
	}
	
	// compare two rows according to their values at a column.
	public int compareRowsByColumn(int row1, int row2, int column) {
						
			Object o1,o2;
						
			if (rows != null){			
				o1 = rows[row1][column];
				o2 = rows[row2][column];
			}
			else {// rowsVec != null
				o1 = ((Vector)(rowsVec.elementAt(row1))).elementAt(column);
				o2 = ((Vector)(rowsVec.elementAt(row2))).elementAt(column);
			}

			// If both values are null, return 0.
			if (o1 == null && o2 == null) {
				return 0; 
			} else if (o1 == null) { // null is less than everything. 
				return -1; 
			} else if (o2 == null) { 
				return 1; 
			}			

			if (o1 instanceof Integer) {
				
				int i1 = ((Integer) o1).intValue();
				int i2 = ((Integer) o2).intValue();

				if (i1 < i2) {
					return -1;
				} else if (i1 > i2) {
					return 1;
				} else {
					return 0;
				}
			} 
			else if (o1 instanceof Double){
				double d1 = ((Double) o1).doubleValue();
				double d2 = ((Double) o2).doubleValue();

				if (d1 < d2) {
					return -1;
				} else if (d1 > d2) {
					return 1;
				} else {
					return 0;
				}
			}		
			else if (o1 instanceof String) {
				String s1 = (String) o1;
				String s2 = (String) o2;
				int result = s1.compareTo(s2);

				if (result < 0) {
					return -1;
				} else if (result > 0) {
					return 1;
				} else {
					return 0;
				}
			} else if (o1 instanceof Boolean) {
				Boolean bool1 = (Boolean) o1;
				boolean b1 = bool1.booleanValue();
				Boolean bool2 = (Boolean) o2;
				boolean b2 = bool2.booleanValue();

				if (b1 == b2) {
					return 0;
				} else if (b1) { // Define false < true
					return 1;
				} else {
					return -1;
				}
			} else {				
				String s1 = o1.toString();				
				String s2 = o2.toString();
				int result = s1.compareTo(s2);

				if (result < 0) {
					return -1;
				} 
				else if (result > 0) {
					return 1;
				}
				else {
					return 0;
				}
			}
		}
}
