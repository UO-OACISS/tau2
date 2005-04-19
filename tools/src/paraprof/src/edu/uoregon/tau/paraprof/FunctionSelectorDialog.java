package edu.uoregon.tau.paraprof;

import java.awt.Container;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Iterator;
import java.util.Vector;

import javax.swing.*;

/**
 * Function selector dialog.  Nothing in it is "function" specific except the title.
 * Other than that this could be used as a generic "object" selector.
 *    
 * This dialog works best as a 'modal' dialog
 *   
 * TODO: nothing
 *
 * <P>CVS $Id: FunctionSelectorDialog.java,v 1.3 2005/04/19 23:25:19 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.3 $
 */
public class FunctionSelectorDialog extends JDialog {

    private Vector items = new Vector();
    private JList list;
    private boolean selected;
    private Object selectedObject;
    

    // center the frame in the owner 
    private void center(JFrame owner) {
      
        int centerOwnerX = owner.getX() + (owner.getWidth() / 2);
        int centerOwnerY = owner.getY() + (owner.getHeight() / 2);
        
        
        int posX = centerOwnerX-(this.getWidth()/2);
        int posY = centerOwnerY-(this.getHeight()/2);
        
        posX = Math.max(posX,0);
        posY = Math.max(posY,0);
        
        this.setLocation(posX, posY );
    }
    
    public boolean choose() {
        this.show();

        if (!selected)
            return false;
        
        if (list.getSelectedIndex() == 0) {
            selectedObject = null;
        } else {
            selectedObject = items.get(list.getSelectedIndex());
        }
        
        return true;
    }

    
    public FunctionSelectorDialog(JFrame owner, boolean modal, Iterator functions, Object initialSelection) {
        
        super(owner, modal);
        this.setTitle("Select a Function");
        this.setSize(600,600);
      
        center(owner);
        int selectedIndex = 0;
        
        items.add("   <none>");
        int index = 1;
        for (Iterator it = functions; it.hasNext();) {
            Object object = it.next();
            if (object == initialSelection) {
                selectedIndex = index;
            }
            items.add(object);
            index++;
        }

        list = new JList(items);
        list.setSelectedIndex(selectedIndex);
        list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        JScrollPane sp = new JScrollPane(list);
        
        Container panel = this.getContentPane();
        
        panel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        
        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;

        JButton okButton = new JButton ("select");
        okButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    selected = true;
                    dispose();
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }
        });
        JButton cancelButton = new JButton ("cancel");
        
        cancelButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    dispose();
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }
        });
        
        
        
        JPanel buttonPanel = new JPanel();
        buttonPanel.add(cancelButton);
        buttonPanel.add(okButton);
        ParaProfUtils.addCompItem(panel, sp, gbc, 0, 0, 1, 1);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        ParaProfUtils.addCompItem(panel, buttonPanel, gbc, 0, 1, 1, 1);
    }
    
    public Object getSelectedObject() {
        return selectedObject;
    }
}
