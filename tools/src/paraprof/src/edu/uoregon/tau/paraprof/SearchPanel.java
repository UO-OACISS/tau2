package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;

import javax.swing.*;

import edu.uoregon.tau.paraprof.interfaces.Searchable;
import edu.uoregon.tau.paraprof.interfaces.SearchableOwner;


public class SearchPanel extends JPanel {

    private Searchable searchable;
    
    private JTextField searchField;
    
    SearchPanel(final SearchableOwner owner, final Searchable searchable) {
        this.searchable = searchable;
        
        
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
                        searchField.setBackground(Color.WHITE);
                        searchField.setForeground(Color.BLACK);
                    } else {
                        searchField.setBackground(new Color(255,102,102));
                        searchField.setForeground(Color.WHITE);
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
        
        final JButton prevButton = new JButton("Previous");
        prevButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                searchable.searchPrevious();
            }
        });
        
        
        final JCheckBox highlightBox = new JCheckBox("Highlight");
        highlightBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                searchable.setSearchHighlight(highlightBox.isSelected());
            }
        });
        
        final JCheckBox matchCaseBox = new JCheckBox("Match Case");
        matchCaseBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                searchable.setSearchMatchCase(matchCaseBox.isSelected());
                if (searchable.setSearchString(searchField.getText())) {
                    searchField.setBackground(Color.WHITE);
                    searchField.setForeground(Color.BLACK);
                } else {
                    searchField.setBackground(new Color(255,102,102));
                    searchField.setForeground(Color.WHITE);
                }
            }
        });

        
        
        
        
        ParaProfUtils.addCompItem(this, closeButton, gbc, 0, 0, 1, 1);

        ParaProfUtils.addCompItem(this, new JLabel("Find:"), gbc, 1, 0, 1, 1);
        
        
        
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 1;
        gbc.weighty = 1;
        ParaProfUtils.addCompItem(this, searchField, gbc, 2, 0, 1, 1);
        
        gbc.weightx = 0;
        gbc.weighty = 0;
        ParaProfUtils.addCompItem(this, nextButton, gbc, 3, 0, 1, 1);
        ParaProfUtils.addCompItem(this, prevButton, gbc, 4, 0, 1, 1);
        ParaProfUtils.addCompItem(this, highlightBox, gbc, 5, 0, 1, 1);
        ParaProfUtils.addCompItem(this, matchCaseBox, gbc, 6, 0, 1, 1);

        
        searchField.requestFocus();
        
        
    }
    
    
    
    public void setFocus() {
        searchField.requestFocus();
        
    }
    
    
}
