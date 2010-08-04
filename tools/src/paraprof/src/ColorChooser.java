/*
 * ParaProf.java
 * 
 * Title: ParaProf 
 * Author: Robert Bell 
 * Description:
 */

package edu.uoregon.tau.paraprof;

import java.awt.Color;
import java.awt.Component;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

import edu.uoregon.tau.paraprof.enums.SortType;
import edu.uoregon.tau.perfdmf.Function;
import edu.uoregon.tau.perfdmf.Group;
import edu.uoregon.tau.perfdmf.UserEvent;

public class ColorChooser implements WindowListener {

    private Vector<Color> colors = new Vector<Color>();
    private Vector<Color> groupColors = new Vector<Color>();
    private Color functionHighlightColor = Color.red;
    private Color groupHighlightColor = new Color(0, 255, 255);
    private Color userEventHighlightColor = new Color(255, 255, 0);
    private Color miscFunctionColor = Color.black;

    private boolean clrChooserFrameShowing = false; //For determining whether the clrChooserFrame is showing.
    private ColorDefaultsWindow clrChooserFrame;

    public ColorChooser(Preferences savedPreferences) {

        if (savedPreferences != null) {
            colors = savedPreferences.getColors();
            groupColors = savedPreferences.getGroupColors();
            functionHighlightColor = savedPreferences.getHighlightColor();
            groupHighlightColor = savedPreferences.getGroupHighlightColor();
            userEventHighlightColor = savedPreferences.getUserEventHighlightColor();
            miscFunctionColor = savedPreferences.getMiscFunctionColor();

            if (functionHighlightColor == null) {
                functionHighlightColor = Color.red;
            }

            if (groupHighlightColor == null) {
                groupHighlightColor = new Color(0, 255, 255);
            }

            if (userEventHighlightColor == null) {
                userEventHighlightColor = new Color(255, 255, 0);
            }

            if (miscFunctionColor == null) {
                miscFunctionColor = Color.black;
            }

        } else {
            //Set the default colors.
            this.setDefaultColors();
            this.setDefaultGroupColors();
        }
    }

    public void showColorChooser(Component invoker) {
        if (!clrChooserFrameShowing) {
            //Bring up the color chooser frame.
            clrChooserFrame = new ColorDefaultsWindow(this, invoker);
            clrChooserFrame.addWindowListener(this);
            clrChooserFrame.setVisible(true);
            clrChooserFrameShowing = true;
        } else {
            //Just bring it to the foreground.
            clrChooserFrame.setVisible(true);
        }
    }

    public void setSavedColors() {
        ParaProf.preferences.setColors(colors);
        ParaProf.preferences.setGroupColors(groupColors);
        ParaProf.preferences.setHighlightColor(functionHighlightColor);
        ParaProf.preferences.setGroupHighlightColor(groupHighlightColor);
        ParaProf.preferences.setMiscFunctionColor(miscFunctionColor);
    }

    public int getNumberOfColors() {
        return colors.size();
    }

    public int getNumberOfGroupColors() {
        return groupColors.size();
    }

    public void addColor(Color color) {
        colors.add(color);
    }

    public void setColor(Color color, int location) {
        colors.setElementAt(color, location);
    }

    public Color getColor(int location) {
        return colors.get(location);
    }

    public Vector<Color> getColors() {
        return colors;
    }

    public void addGroupColor(Color color) {
        groupColors.add(color);
    }

    public void setGroupColor(Color color, int location) {
        groupColors.setElementAt(color, location);
    }

    public Color getGroupColor(int location) {
        return groupColors.elementAt(location);
    }

    public Vector<Color> getGroupColors() {
        return groupColors;
    }

    public void setHighlightColor(Color highlightColor) {
        this.functionHighlightColor = highlightColor;
    }

    public Color getHighlightColor() {
        return functionHighlightColor;
    }

    public void setGroupHighlightColor(Color groupHighlightColor) {
        this.groupHighlightColor = groupHighlightColor;
    }

    public Color getGroupHighlightColor() {
        return groupHighlightColor;
    }

    // User Event Colors
    public void setUserEventHighlightColor(Color userEventHighlightColor) {
        this.userEventHighlightColor = userEventHighlightColor;
    }

    public Color getUserEventHighlightColor() {
        return userEventHighlightColor;
    }

    // User Event Colors

    public void setMiscFunctionColor(Color miscFunctionColor) {
        this.miscFunctionColor = miscFunctionColor;
    }

    public Color getMiscFunctionColor() {
        return miscFunctionColor;
    }

    //A function which sets the colors vector to be the default set.
    public void setDefaultColors() {
        colors.clear();

        addColor(new Color(115, 165, 255));
        addColor(new Color(215, 0, 0));
        addColor(new Color(70, 215, 70));
        addColor(new Color(105, 17, 169));
        addColor(new Color(255, 153, 0));
        addColor(new Color(0, 0, 255));
        addColor(new Color(204, 255, 51));
        addColor(new Color(0, 227, 154));
        addColor(new Color(233, 107, 53));
        addColor(new Color(204, 194, 220));
        addColor(new Color(44, 238, 202));
        addColor(new Color(30, 175, 141));
        addColor(new Color(132, 127, 121));
        addColor(new Color(246, 211, 47));

        for (int i = 0; i < 50; i++) {
            //addColor(new Color((i*i*777 + 85) % 255, ((i*i*i * 333) + 125) % 255, ((i * 666) + 205) % 255));
            addColor(new Color((i * i * 525) % 255, ((i * i * i * 33) + 125) % 255, ((i * 666) + 205) % 255));
            //addColor(new Color((float)Math.random(),(float)Math.random(),(float)Math.random()));

        }
        //        
        //        //Add the default colors.
        //        addColor(new Color(70, 156, 168));
        //        addColor(new Color(255, 153, 0));
        //
        //        addColor(new Color(0, 51, 255));
        //
        //        addColor(new Color(102, 0, 51));
        //        addColor(new Color(221, 232, 30));
        //        addColor(new Color(0, 255, 0));
        //        addColor(new Color(121, 196, 144));
        //        addColor(new Color(86, 88, 112));
        //
        //        addColor(new Color(151, 204, 255));
        //        addColor(new Color(102, 102, 255));
        //        addColor(new Color(0, 102, 102));
        //        addColor(new Color(204, 255, 51));
        //        addColor(new Color(102, 132, 25));
        //        addColor(new Color(255, 204, 153));
        //        addColor(new Color(204, 0, 204));
        //        addColor(new Color(0, 102, 102));
        //        addColor(new Color(204, 204, 255));
        //        addColor(new Color(61, 104, 63));
        //        addColor(new Color(102, 255, 255));
        //        addColor(new Color(255, 102, 102));
        //        addColor(new Color(119, 71, 145));
        //        addColor(new Color(255, 204, 204));
        //        addColor(new Color(240, 97, 159));
        //        addColor(new Color(0, 102, 153));
    }

    //A function which sets the groupColors vector to be the default set.
    public void setDefaultGroupColors() {
        //Clear the globalColors vector.
        groupColors.clear();

        //Add the default colors.
        addGroupColor(new Color(102, 0, 102));
        addGroupColor(new Color(51, 51, 0));
        addGroupColor(new Color(204, 0, 51));
        addGroupColor(new Color(0, 102, 102));
        addGroupColor(new Color(255, 255, 102));
        addGroupColor(new Color(0, 0, 102));
        addGroupColor(new Color(153, 153, 255));
        addGroupColor(new Color(255, 51, 0));
        addGroupColor(new Color(255, 153, 0));
        addGroupColor(new Color(255, 102, 102));
        addGroupColor(new Color(51, 0, 51));
        addGroupColor(new Color(255, 255, 102));
    }

    public Color getColorLocation(int i) {
        return getColor(i % getNumberOfColors());
    }

    //Sets the colors of the given TrialData.
    //If the selection is equal to -1, then set the colors in all the
    // sets, otherwise, just set the ones for the specified set.
    public void setColors(ParaProfTrial ppTrial, int selection) {

        if ((selection == -1) || (selection == 0)) {
            int numberOfColors = this.getNumberOfColors();

            DataSorter dataSorter = new DataSorter(ppTrial);
            dataSorter.setSortType(SortType.MEAN_VALUE);
            dataSorter.setDescendingOrder(true);
            List<Comparable> list = dataSorter.getFunctionProfiles(ppTrial.getDataSource().getMeanData());

            for (int i = 0; i < list.size(); i++) {
                Function func = ((PPFunctionProfile) list.get(i)).getFunction();

                Color color = ParaProf.colorMap.getColor(func);
                if (color != null) {
                    func.setSpecificColor(color);
                    func.setColorFlag(true);
                } else {
                    func.setColorFlag(false);
                    func.setSpecificColor(null);
                }

                // we could be doing runtime analysis, don't reassign colors
                if (func.getColor() == null) {
                    func.setColor(this.getColor(i % numberOfColors));
                }
            }
        }

        if ((selection == -1) || (selection == 1)) {
            int numberOfColors = this.getNumberOfGroupColors();
            for (Iterator i = ppTrial.getDataSource().getGroups(); i.hasNext();) {
                Group group = (Group) i.next();
                group.setColor(this.getGroupColor((group.getID()) % numberOfColors));
            }
        }

        if ((selection == -1) || (selection == 2)) {
            int numberOfColors = this.getNumberOfColors();
            for (Iterator i = ppTrial.getDataSource().getUserEvents(); i.hasNext();) {
                UserEvent userEvent = (UserEvent) i.next();
                userEvent.setColor(this.getColor((userEvent.getID()) % numberOfColors));
            }
        }
    }

    //Window Listener code.
    public void windowClosed(WindowEvent winevt) {
    }

    public void windowIconified(WindowEvent winevt) {
    }

    public void windowOpened(WindowEvent winevt) {
    }

    public void windowClosing(WindowEvent winevt) {
        if (winevt.getSource() == clrChooserFrame) {
            clrChooserFrameShowing = false;
        }
    }

    public void windowDeiconified(WindowEvent winevt) {
    }

    public void windowActivated(WindowEvent winevt) {
    }

    public void windowDeactivated(WindowEvent winevt) {
    }

}
