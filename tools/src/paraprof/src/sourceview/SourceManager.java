package edu.uoregon.tau.paraprof.sourceview;

import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JScrollPane;
import javax.swing.ListSelectionModel;

import edu.uoregon.tau.paraprof.ParaProfUtils;
import edu.uoregon.tau.paraprof.WindowPlacer;
import edu.uoregon.tau.perfdmf.SourceRegion;

public class SourceManager extends JFrame {

    /**
	 * 
	 */
	private static final long serialVersionUID = -2806416317097108083L;
	private DefaultListModel listModel;
    private JList dirList;
    private SourceRegion toFind;

    private Map<File, SourceViewer> sourceViewers = new TreeMap<File, SourceViewer>();

    public ArrayList<Object> getCurrentElements() {
        ArrayList<Object> list = new ArrayList<Object>();
        for (int i = 0; i < listModel.getSize(); i++) {
            list.add(listModel.getElementAt(i));
        }
        return list;
    }

    private boolean matchFiles(String s1, String s2) {
        //System.out.println("comparing " + s1 + " to " + s2);
    	
    	if(s1==null || s2==null)
    		return false;
    	
    	if(s1.equals(s2))
    		return true;
    	
    	File f1=new File(s1);
    	File f2=new File(s2);
    	
    	if(f1.exists()&&f2.exists())
    	{
        try {
			if (f1.getCanonicalFile().equals(f2.getCanonicalFile())) {
			    return true;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
    	}
        return false;
    }

    private void launchSourceViewer(File sourceFile,SourceRegion region){
        SourceViewer sourceViewer = sourceViewers.get(sourceFile);
        if (sourceViewer == null) {
            sourceViewer = new SourceViewer(sourceFile);
            sourceViewers.put(sourceFile, sourceViewer);
        }
        sourceViewer.highlightRegion(region);
        sourceViewer.setVisible(true);
    }
    
    private boolean searchLocations(SourceRegion region, File[] list, boolean recurse) {
    	if(list==null)
    		return false;
    	String name = region.getFilename();
        for (int j = 0; j < list.length; j++) {
            if (matchFiles(name, list[j].getName())) {
                //System.out.println("found it");
            	launchSourceViewer(list[j],region);
                return true;
            }
        }
        /*
         * Check if we use unix or windows separators
         */
        char separator = '/';
        if(!name.startsWith("/")){
        	separator='\\';
        }
        /*
         *Check the file name by itself instead of the full path. We don't continue if we get 0 because it already failed to find that. 
         */
        int last = name.lastIndexOf(separator)+1;
        if(last>1&&last<name.length()){
        name=region.getFilename().substring(last);
        for (int j = 0; j < list.length; j++) {
            if (matchFiles(name, list[j].getName())) {
                //System.out.println("found it");
            	launchSourceViewer(list[j],region);
                return true;
            }
        }
        }

        if (recurse) {
            for (int j = 0; j < list.length; j++) {
                if (list[j].isDirectory()) {
                    if (searchLocations(region, list[j].listFiles(), recurse)) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    public void showSourceCode(SourceRegion region) {

        String filename = region.getFilename();

        
        File rfile = new File(filename);
        if(rfile.exists())
        {
        	this.launchSourceViewer(rfile, region);
        	return;
        }
        
        File cwd = new File(".");

        if (searchLocations(region, cwd.listFiles(), false)) {
            return;
        }

        for (int i = 0; i < listModel.getSize(); i++) {
            String directory = (String) listModel.getElementAt(i);
            File file = new File(directory);
            File[] children = file.listFiles();
            if (children != null) {
                if (searchLocations(region, children, true)) {
                    return;
                }
            }
        }

        //        JOptionPane.showMessageDialog(this, "ParaProf could not find \"" + filename
        //                + "\", please add the containing directory, or a parent to the search list.");
        int hr = JOptionPane.showOptionDialog(this, "ParaProf could not find \"" + filename
                + "\", would you like to add the containing directory to the search list?", "Looking for \"" + filename + "\"",
                JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE, null, null, null);

        toFind = null;
        if (hr == JOptionPane.YES_OPTION) {
            toFind = region;
            display(null);
        }
    }

    public SourceManager(List<Object> initialElements) {

        Container contentPane = getContentPane();
        contentPane.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0;
        gbc.weighty = 0;

        // First add the label.
        JLabel titleLabel = new JLabel("Source Code Directories (searched recursively from top to bottom)");
        titleLabel.setFont(new Font("SansSerif", Font.PLAIN, 14));
        addCompItem(titleLabel, gbc, 0, 0, 1, 1);

        gbc.fill = GridBagConstraints.BOTH;
        gbc.anchor = GridBagConstraints.WEST;
        gbc.weightx = 0.1;
        gbc.weighty = 0.1;

        // Create and add color list.
        listModel = new DefaultListModel();

        if (initialElements != null) {
            for (int i = 0; i < initialElements.size(); i++) {
                listModel.addElement(initialElements.get(i));
            }
        }

        dirList = new JList(listModel);
        dirList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        dirList.setSize(500, 300);
        // colorList.addMouseListener(this);
        JScrollPane sp = new JScrollPane(dirList);
        addCompItem(sp, gbc, 0, 1, 1, 5);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        JButton button = new JButton("Add...");
        button.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent evt) {
                JFileChooser jFileChooser = new JFileChooser();
                jFileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                jFileChooser.setMultiSelectionEnabled(false);
                jFileChooser.setDialogTitle("Select Directory");
                jFileChooser.setApproveButtonText("Select");
                if ((jFileChooser.showOpenDialog(SourceManager.this)) == JFileChooser.APPROVE_OPTION) {
                  try {
                    Object path = (jFileChooser.getSelectedFile().getCanonicalPath());
                    if (!listModel.contains(path)) {
                        listModel.addElement(path);
                    }
                  } catch (Exception e) {
                      ParaProfUtils.handleException(e);
                  }
                }
            }
        });
        addCompItem(button, gbc, 1, 1, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        button = new JButton("Move Up");
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
              int index = dirList.getSelectedIndex();
              if (index > 0) {
                Object obj = listModel.getElementAt(index-1);
                listModel.insertElementAt(obj, index+1);
                listModel.removeElementAt(index-1);
                dirList.setSelectedIndex(index-1);
                dirList.ensureIndexIsVisible(index-1);
              }
            }
        });
        addCompItem(button, gbc, 1, 2, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        button = new JButton("Move Down");
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
              int index = dirList.getSelectedIndex();
              if (index != -1) {
                int count = listModel.getSize();
                if (index != count-1) {
                  Object obj = listModel.getElementAt(index);
                  listModel.insertElementAt(obj, index+2);
                  listModel.removeElementAt(index);
                  dirList.setSelectedIndex(index+1);
                  dirList.ensureIndexIsVisible(index+1);
                }
              }
            }
        });
        addCompItem(button, gbc, 1, 3, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.NORTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        button = new JButton("Remove");
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                int index = dirList.getSelectedIndex();
                if (index >= 0) {
                    listModel.removeElementAt(index);
                }
            }
        });
        addCompItem(button, gbc, 1, 4, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.SOUTH;
        gbc.weightx = 0;
        gbc.weighty = 0;
        button = new JButton("Close");
        button.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                if (toFind != null) {
                    showSourceCode(toFind);
                }
                SourceManager.this.setVisible(false);
            }
        });
        addCompItem(button, gbc, 1, 5, 1, 1);

    }

    public void display(Component invoker) {
        setSize(new Dimension(855, 450));
        setLocation(WindowPlacer.getNewLocation(this, invoker));
        setTitle("TAU: ParaProf: Source Directory Manager");
        ParaProfUtils.setFrameIcon(this);
        setVisible(true);
    }

    private void addCompItem(Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        getContentPane().add(c, gbc);
    }

}
