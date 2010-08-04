package edu.uoregon.tau.paraprof;

import java.awt.Container;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.ListSelectionModel;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.perfdmf.Function;

/**
 * Function selector dialog.  Nothing in it is "function" specific except the title.
 * Other than that this could be used as a generic "object" selector.
 *    
 * This dialog works best as a 'modal' dialog
 *   
 * TODO: nothing
 *
 * <P>CVS $Id: FunctionSelectorDialog.java,v 1.4 2009/11/12 22:50:27 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.4 $
 */
public class FunctionSelectorDialog extends JDialog {

    /**
	 * 
	 */
	private static final long serialVersionUID = -4743553292070955890L;
	private Vector<Object> items = new Vector<Object>();
    private JList list;
    private boolean selected;
    private Object selectedObject;
    private List<Object> selectedObjects;
    private boolean allowNone;
    private boolean allowMultiple;

    // center the frame in the owner 
    private void center(JFrame owner) {

        int centerOwnerX = owner.getX() + (owner.getWidth() / 2);
        int centerOwnerY = owner.getY() + (owner.getHeight() / 2);

        int posX = centerOwnerX - (this.getWidth() / 2);
        int posY = centerOwnerY - (this.getHeight() / 2);

        posX = Math.max(posX, 0);
        posY = Math.max(posY, 0);

        this.setLocation(posX, posY);
    }

    public boolean choose() {
        this.setVisible(true);

        if (!selected) {
            return false;
        }

        if (allowMultiple) {
            list.getSelectedIndices();
            selectedObjects = new ArrayList<Object>();
            for (int i = 0; i < list.getSelectedIndices().length; i++) {
                selectedObjects.add(items.get(list.getSelectedIndices()[i]));
            }
        } else {

            if (list.getSelectedIndex() == 0 && allowNone) {
                selectedObject = null;
            } else {
                selectedObject = items.get(list.getSelectedIndex());
            }
        }
        return true;
    }

    public FunctionSelectorDialog(JFrame owner, boolean modal, Iterator functions, Object initialSelection, boolean allowNone,
            boolean allowMultiple) {

        super(owner, modal);

        this.allowNone = allowNone;
        this.allowMultiple = allowMultiple;
        this.setTitle("Select a Function");
        this.setSize(600, 600);

        center(owner);
        int selectedIndex = 0;

        int index = 0;
        if (allowNone) {
            items.add("   <none>");
            index++;
        }
        for (Iterator<Function> it = functions; it.hasNext();) {
            Object object = it.next();
            if (object == initialSelection) {
                selectedIndex = index;
            }
            items.add(object);
            index++;
        }

        list = new JList(items);
        list.setSelectedIndex(selectedIndex);
        if (allowMultiple) {
            list.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
        } else {
            list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        }
        JScrollPane sp = new JScrollPane(list);

        Container panel = this.getContentPane();

        panel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;

        JButton okButton = new JButton("select");
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
        JButton cancelButton = new JButton("cancel");

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
        buttonPanel.add(okButton);
        buttonPanel.add(cancelButton);
        Utility.addCompItem(panel, sp, gbc, 0, 0, 1, 1);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.weightx = 0;
        gbc.weighty = 0;
        Utility.addCompItem(panel, buttonPanel, gbc, 0, 1, 1, 1);
    }

    public Object getSelectedObject() {
        return selectedObject;
    }

    public List<Object> getSelectedObjects() {
        return selectedObjects;
    }
}
