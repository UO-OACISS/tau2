/**
 * 
 */
package edu.uoregon.tau.perfexplorer.client;

import edu.uoregon.tau.common.treetable.AbstractTreeTableModel;
import edu.uoregon.tau.common.treetable.TreeTableModel;

public class XMLModel extends AbstractTreeTableModel implements TreeTableModel {

    // Names of the columns.
    String[] cNames = { "Element", "Value" };

    // Types of the columns.
    Class<?>[] cTypes = { TreeTableModel.class, String.class };

    // The the returned file length for directories. 
    final Integer ZERO = new Integer(0);
    
    public XMLModel(XMLNode root) {
        super(root);
    }

    protected String getValue(Object node) {
        XMLNode XMLNode = ((XMLNode) node);
        return XMLNode.getValue();
    }

    //
    // The TreeModel interface
    //

    public int getChildCount(Object obj) {
        XMLNode node = (XMLNode)obj;
        return node.getChildCount();
    }

    public Object getChild(Object obj, int i) {
        XMLNode node = (XMLNode)obj;
        return node.getChildAt(i);
    }

    public boolean isLeaf(Object obj) {
        XMLNode node = (XMLNode)obj;
        return node.isLeaf();
    }

    //
    //  The TreeTableNode interface. 
    //

    public int getColumnCount() {
        return cNames.length;
    }

    public String getColumnName(int column) {
        return cNames[column];
    }

    @SuppressWarnings({ "rawtypes", "unchecked" })
	public Class getColumnClass(int column) {
        return cTypes[column];
    }

    public Object getValueAt(Object obj, int column) {
    	XMLNode node = (XMLNode)obj;
        try {
            switch (column) {
            case 0:
                return node.toString();
            case 1:
                return node.getValue();
            }
        } catch (SecurityException se) {
        }

        return null;
    }

    public int getColorMetric() {
        // TODO Auto-generated method stub
        return 0;
    }
}