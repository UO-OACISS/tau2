package edu.uoregon.tau.common;

import java.awt.event.ActionListener;
import java.awt.event.InputEvent;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;

public class TreeUI {
	
	
	public static JPopupMenu getTreePopupMenu(List<String> items, ActionListener l){
		JPopupMenu popUp = new JPopupMenu();
		JMenuItem jMenuItem;
		for(int i=0;i<items.size();i++){
			jMenuItem=new JMenuItem(items.get(i));
			jMenuItem.addActionListener(l);
			popUp.add(jMenuItem);
		}
		
		return popUp;
	}
	
	public static JPopupMenu getDatabasePopup(ActionListener l){
		ArrayList<String> items = new ArrayList<String>(3);
		items.add("Add Application");
		items.add("Add Experiment");
		items.add("Add Trial");
		
		return getTreePopupMenu(items,l);
	}
	
	public static JPopupMenu getTauDBPopUp(ActionListener l){
		ArrayList<String> items = new ArrayList<String>(2);
		items.add("Add Trial");
		items.add("Add View");
		return getTreePopupMenu(items,l);
	}
	
	public static JPopupMenu getViewPopUp(ActionListener l){
		ArrayList<String> items = new ArrayList<String>(3);
		items.add("Add Sub-View");
		items.add("Add Metadata Field To All Trials");
		items.add("Remove Metadata Field From All Trials");
		items.add("Edit");
		items.add("Delete");
		items.add("Rename");
		return getTreePopupMenu(items,l);
	}
	
	public static JPopupMenu getStdAppPopUp(ActionListener l){
		ArrayList<String> items = new ArrayList<String>(5);
		items.add("Add Experiment");
		items.add("Add Trial");
		items.add("Upload Application to DB");
		items.add("Delete");
		items.add("Rename");
		return getTreePopupMenu(items,l);
	}
	
	public static JPopupMenu getDbAppPopUp(ActionListener l){
		ArrayList<String> items = new ArrayList<String>(4);
		items.add("Add Experiment");
		items.add("Add Trial");
		items.add("Delete");
		items.add("Rename");
		return getTreePopupMenu(items,l);
	}
	
	public static JPopupMenu getMultiPopUp(ActionListener l){
		ArrayList<String> items = new ArrayList<String>(1);
		
		// jMenuItem = new JMenuItem("Copy");
		// jMenuItem.addActionListener(this);
		// multiPopup.add(jMenuItem);
		//
		// jMenuItem = new JMenuItem("Cut");
		// jMenuItem.addActionListener(this);
		// multiPopup.add(jMenuItem);
		//
		// jMenuItem = new JMenuItem("Paste");
		// jMenuItem.addActionListener(this);
		// multiPopup.add(jMenuItem);
		
		items.add("Delete");
		return getTreePopupMenu(items,l);
	}
	
	public static JPopupMenu getPerfExPopUp(ActionListener l){
		ArrayList<String> items = new ArrayList<String>(4);
		items.add("Delete");
		items.add("Rename");
		return getTreePopupMenu(items,l);
	}

    public static boolean rightClick(MouseEvent evt) {
        if ((evt.getModifiers() & InputEvent.BUTTON3_MASK) != 0) {
            return true;
        }
        return false;
    }
	
}
