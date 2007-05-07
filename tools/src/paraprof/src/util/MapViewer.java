package edu.uoregon.tau.paraprof.util;

import java.util.Iterator;
import java.util.Map;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTable;

import edu.uoregon.tau.paraprof.ParaProfUtils;

public class MapViewer extends JFrame {

    public MapViewer(String title, Map map) {
        JTable table;

        int num = map.size();
        String[][] items = new String[num][2];

        int index = 0;
        for (Iterator it = map.keySet().iterator(); it.hasNext();) {
            String key = (String) it.next();
            String value = (String) map.get(key);
            items[index][0] = key;
            items[index][1] = value;
            index++;
        }

        String columns[] = new String[2];
        columns[0] = "Name";
        columns[1] = "Value";

        table = new JTable(items, columns);

        JScrollPane scrollpane = new JScrollPane(table);
        getContentPane().add(scrollpane);
        setTitle(title);
        pack();
        ParaProfUtils.setFrameIcon(this);
    }

}
