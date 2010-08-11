package edu.uoregon.tau.paraprof.tablemodel;

import java.awt.Color;
import java.awt.Component;
import java.util.Map;

import javax.swing.JTable;
import javax.swing.table.DefaultTableCellRenderer;

public class TrialCellRenderer extends DefaultTableCellRenderer {

    /**
	 * 
	 */
	private static final long serialVersionUID = -7147446002751695053L;
	private Color grey = new Color(235, 235, 235);
    private Color green = new Color(0, 185, 0);
    private Color red = new Color(215, 0, 0);

    private Map<String, String> common, other;

    public TrialCellRenderer(Map<String, String> common, Map<String, String> other) {
        this.common = common;
        this.other = other;
    }

    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row,
            int column) {
        Component cell = super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);
        //if (value instanceof Integer) {
        if (row % 2 == 0) {
            cell.setBackground(grey);
            // You can also customize the Font and Foreground this way
            // cell.setForeground();
            // cell.setFont();
        } else {
            cell.setBackground(Color.white);
        }
        //}

        
     //   green = new Color(0, 175, 0);
        
        if (column == 0) {
            if (common.get(value) != null) {
                cell.setForeground(green);
            } else if (other.get(value) != null) {
                cell.setForeground(red);
            } else {
                cell.setForeground(Color.black);
            }
        } else {
            cell.setForeground(Color.black);
        }

        return cell;
    }

}
