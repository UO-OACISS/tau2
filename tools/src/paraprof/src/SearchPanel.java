package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import edu.uoregon.tau.common.Utility;
import edu.uoregon.tau.paraprof.interfaces.Searchable;
import edu.uoregon.tau.paraprof.interfaces.SearchableOwner;

/**
 * Search panel for ParaProf windows, similar to FireFox search panel
 *    
 * TODO : ...
 *
 * <P>CVS $Id: SearchPanel.java,v 1.2 2009/11/12 22:50:28 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class SearchPanel extends JPanel {

    /**
	 * 
	 */
	private static final long serialVersionUID = 8136473488409647519L;
	//private Searchable searchable;
    private JTextField searchField;
    
    public SearchPanel(final SearchableOwner owner, final Searchable searchable, boolean showHighlight, boolean showPrevious){
    	createSearchPanel(owner, searchable, showHighlight, showPrevious);
    }
    
    public SearchPanel(final SearchableOwner owner, final Searchable searchable){
    	createSearchPanel(owner, searchable, true, true);
    }

    private void createSearchPanel(final SearchableOwner owner, final Searchable searchable, boolean showHighlight, boolean showPrevious) {
        //this.searchable = searchable;

        searchable.setSearchHighlight(false);
        searchable.setSearchMatchCase(false);

        searchField = new JTextField();

        searchField.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {

            }

        });

        searchField.addKeyListener(new KeyListener() {

            public void keyPressed(KeyEvent e) {

            }

            public void keyReleased(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_ENTER) {
                    searchable.searchNext();
                } else {
                    if (searchable.setSearchString(searchField.getText())) {
                        searchField.setBackground(Color.white);
                        searchField.setForeground(Color.black);
                    } else {
                        searchField.setBackground(new Color(255, 102, 102));
                        searchField.setForeground(Color.white);
                    }
                }
            }

            public void keyTyped(KeyEvent e) {

            }

        });

        this.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.EAST;
        gbc.insets = new Insets(0, 5, 0, 5);
        gbc.weightx = 0;
        gbc.weighty = 0;

        JButton closeButton = new JButton("Close");
        closeButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                searchable.setSearchString("");
                owner.showSearchPanel(false);
            }
        });

        final JButton nextButton = new JButton("Next");
        nextButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                searchable.searchNext();
            }
        });
        JButton tmpPrevious=null;
        if(showPrevious){
        final JButton prevButton = new JButton("Previous");
        prevButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                searchable.searchPrevious();
            }
        });
        tmpPrevious=prevButton;
        }
        JCheckBox tmpHighlight=null;
        if(showHighlight){
        final JCheckBox highlightBox = new JCheckBox("Highlight");
        highlightBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                searchable.setSearchHighlight(highlightBox.isSelected());
            }
        });
        tmpHighlight=highlightBox;
        }
        
        final JCheckBox matchCaseBox = new JCheckBox("Match Case");
        matchCaseBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                searchable.setSearchMatchCase(matchCaseBox.isSelected());
                if (searchable.setSearchString(searchField.getText())) {
                    searchField.setBackground(Color.white);
                    searchField.setForeground(Color.black);
                } else {
                    searchField.setBackground(new Color(255, 102, 102));
                    searchField.setForeground(Color.white);
                }
            }
        });

        Utility.addCompItem(this, closeButton, gbc, 0, 0, 1, 1);
        Utility.addCompItem(this, new JLabel("Find:"), gbc, 1, 0, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 1;
        gbc.weighty = 1;
        Utility.addCompItem(this, searchField, gbc, 2, 0, 1, 1);

        gbc.weightx = 0;
        gbc.weighty = 0;
        Utility.addCompItem(this, nextButton, gbc, 3, 0, 1, 1);
        int xDex=4;
        if(showPrevious)
        {
        	Utility.addCompItem(this, tmpPrevious, gbc, xDex, 0, 1, 1);
        	xDex++;
        }
        if(showHighlight){
        	Utility.addCompItem(this, tmpHighlight, gbc, xDex, 0, 1, 1);
        	xDex++;
        }
        Utility.addCompItem(this, matchCaseBox, gbc, xDex, 0, 1, 1);

        searchField.requestFocus();
    }

    public void setFocus() {
        searchField.requestFocus();
    }

}
