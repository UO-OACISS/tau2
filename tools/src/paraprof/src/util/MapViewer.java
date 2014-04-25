package edu.uoregon.tau.paraprof.util;

import java.awt.Font;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.Iterator;
import java.util.Map;

import javax.swing.JFrame;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JTable;

import edu.uoregon.tau.common.MetaDataMap;
import edu.uoregon.tau.common.MetaDataMap.MetaDataKey;
import edu.uoregon.tau.common.MetaDataMap.MetaDataValue;
import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.ParaProfUtils;

public class MapViewer extends JFrame {

    /**
	 * 
	 */
	private static final long serialVersionUID = -3534612098086026146L;

	String[][] items=null;
	
	public MapViewer(String title, MetaDataMap map, Font f) {
        JTable table;

        int num = map.size();
        items = new String[num][2];

        int index = 0;
        for (Iterator<MetaDataKey> it = map.keySet().iterator(); it.hasNext();) {
            MetaDataKey key = it.next();
            MetaDataValue value = map.get(key);
            items[index][0] = key.name;
            items[index][1] = value.value.toString();
            index++;
        }

        String columns[] = new String[2];
        columns[0] = "Name";
        columns[1] = "Value";

        table = new JTable(items, columns);
        
        Utility.setTableFontHeight(table, f);

        JScrollPane scrollpane = new JScrollPane(table);
        getContentPane().add(scrollpane);
        setTitle(title);
        pack();
        ParaProfUtils.setFrameIcon(this);
        table.addMouseListener(getMouseListener(table));
    }
	
	 public MouseListener getMouseListener(final JTable table) {
	        return new MouseListener() {

	            public void mouseClicked(MouseEvent e) {
	                if (ParaProfUtils.rightClick(e)) {
	                    int row = table.rowAtPoint(e.getPoint());
	                    int column = table.columnAtPoint(e.getPoint());
	                    String item = items[row][0];
	                    if(item.startsWith("BACKTRACE"))
	                    {
	                    
	                    	//System.out.println("you clicked on (" + column + "," + row + ") = " + getValueAt(row, column));
	                    	
	                    	
	                    	JPopupMenu popup = ParaProfUtils.createMetadataClickPopUp(items[row][column].toString(), table);
	                        
	                    	if(popup!=null)
	                    		popup.show(table, e.getX(), e.getY());

	                    }
	                    
	                    
	                }
	            }

	            public void mouseEntered(MouseEvent e) {

	            }

	            public void mouseExited(MouseEvent e) {
	            }

	            public void mousePressed(MouseEvent e) {
	            }

	            public void mouseReleased(MouseEvent e) {
	            }
	        };
	    }
	 
	    

}
