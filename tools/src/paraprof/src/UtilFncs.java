 /* 
   UtilFncs.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  Some useful functions for the system.
*/

package paraprof;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.lang.*;
import java.io.*;
import java.text.*;

public class UtilFncs{
  
    public static double adjustDoublePresision(double d, int precision){
	String result = null;
 	try{
	    String formatString = "##0.0";
	    for(int i=0;i<(precision-1);i++){
		formatString = formatString+"0";
	    }
	    if(d < 0.001){
		for(int i=0;i<4;i++){
		    formatString = formatString+"0";
		}
	    }
        
	    DecimalFormat dF = new DecimalFormat(formatString);
	    result = dF.format(d);
	}
	catch(Exception e){
		ParaProf.systemError(e, null, "UF01");
	}
	return Double.parseDouble(result);
    }
    
    public static String getTestString(double inDouble, int precision){
      
	//This method comes up with a rough estimation.  The drawing windows do not
	//need to be absolutely accurate.
      
	String returnString = "";
	for(int i=0;i<precision;i++){
	    returnString = returnString + " ";
	}
      
	long tmpLong = Math.round(inDouble);
	returnString = Long.toString(tmpLong) + returnString;
      
	return returnString;
    }

    //This method is used in a number of windows to determine the actual output string
    //displayed. Current types are:
    //0 - microseconds
    //1 - milliseconds
    //2 - seconds
    //3 - hr:min:sec
    //At present, the passed in double value is assumed to be in microseconds.
    public static String getOutputString(int type, double d){
	switch(type){
	case 0:
	    return (Double.toString(UtilFncs.adjustDoublePresision(d, ParaProf.defaultNumberPrecision)));
	case 1:
	    return (Double.toString(UtilFncs.adjustDoublePresision((d/1000), ParaProf.defaultNumberPrecision)));
	case 2:
	    return (Double.toString(UtilFncs.adjustDoublePresision((d/1000000), ParaProf.defaultNumberPrecision)));
	case 3:
	    int hr = 0;
	    int min = 0;
	    hr = (int) (d/3600000000.00);
	    //Calculate the number of microseconds left after hours are subtracted.
	    d = d-hr*3600000000.00;
	    min = (int) (d/60000000.00);
	    //Calculate the number of microseconds left after minutess are subtracted.
	    d = d-min*60000000.00;
	    return (Integer.toString(hr)+":"+Integer.toString(min)+":"+Double.toString(UtilFncs.adjustDoublePresision((d/1000000), ParaProf.defaultNumberPrecision)));
	default:
	    ParaProf.systemError(null, null, "Unexpected string type - UF02 value: " + type);
	}
	return null;
    }

    public static String getUnitsString(int type, boolean time){
	
	if(!time)
	    return "counts";

	switch(type){
	case 0:
	    return "microseconds";
	case 1:
	    return "milliseconds";
	case 2:
	    return "seconds";
	case 3:
	    return "hour:minute:seconds";
	default:
	    ParaProf.systemError(null, null, "Unexpected string type - UF02 value: " + type);
	}
	return null;
    }

    public static int exists(int[] ref, int i){
	if(ref == null)
	    return -1;
	int test = ref.length;
	for(int j=0;j<test;j++){
	    if(ref[j]==i)
		return j;
	}
	return -1;
    }

    public static int exists(Vector ref, int i){
	//Assuming a vector of Integers.
	if(ref == null)
	    return -1;
	Integer current = null;
	int test = ref.size();
	for(int j=0;j<test;j++){
	    current = (Integer) ref.elementAt(j);
	    if((current.intValue())==i)
		return j;
	}
	return -1;
    }

    public static void fileMenuItems(JMenu jMenu, ActionListener actionListener){
	JMenu subMenu = null;
	JMenuItem menuItem = null;

	//######
	//Open menu.
	//######
	subMenu = new JMenu("Open ...");
	
	menuItem = new JMenuItem("ParaProf Manager");
	menuItem.addActionListener(actionListener);
	subMenu.add(menuItem);
	
	menuItem = new JMenuItem("Bin Window");
	menuItem.addActionListener(actionListener);
	subMenu.add(menuItem);
	
	jMenu.add(subMenu);
	//######
	//End - Open menu.
	//######
	
	//######
	//Save menu.
	//######
	subMenu = new JMenu("Save ...");
	
	menuItem = new JMenuItem("ParaProf Preferrences");
	menuItem.addActionListener(actionListener);
	subMenu.add(menuItem);

	menuItem = new JMenuItem("Save Image");
	menuItem.addActionListener(actionListener);
	subMenu.add(menuItem);
	
	jMenu.add(subMenu);
	//######
	//End - Save menu.
	//######
	
	menuItem = new JMenuItem("Edit ParaProf Preferences!");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);

	menuItem = new JMenuItem("Print");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);
	
	menuItem = new JMenuItem("Close This Window");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);
	
	menuItem = new JMenuItem("Exit ParaProf!");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);
    }

    public static void optionMenuItems(JMenu jMenu, ActionListener actionListener){
	JMenu subMenu = null;
	ButtonGroup group = null;
	JCheckBoxMenuItem box = null;
	JRadioButtonMenuItem button = null;

	box = new JCheckBoxMenuItem("Sort By Name", false);
	box.addActionListener(actionListener);
	jMenu.add(box);

	box = new JCheckBoxMenuItem("Decending Order", true);
	box.addActionListener(actionListener);
	jMenu.add(box);

	box = new JCheckBoxMenuItem("Show Values as Percent", true);
	box.addActionListener(actionListener);
	jMenu.add(box);

	//######
	//Units submenu.
	//######
	subMenu = new JMenu("Select Units");
	group = new ButtonGroup();
	
	button = new JRadioButtonMenuItem("hr:min:sec", false);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	button = new JRadioButtonMenuItem("Seconds", false);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	button = new JRadioButtonMenuItem("Milliseconds", false);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	button = new JRadioButtonMenuItem("Microseconds", true);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	jMenu.add(subMenu);
	//######
	//End - Units submenu.
	//######

	//######
	//Set the metric options.
	//######
	subMenu = new JMenu("Select Metric");
	group = new ButtonGroup();

	button = new JRadioButtonMenuItem("Exclusive", true);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	button = new JRadioButtonMenuItem("Inclusive", false);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	button = new JRadioButtonMenuItem("Number of Calls", false);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	button = new JRadioButtonMenuItem("Number of Subroutines", false);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	button = new JRadioButtonMenuItem("Per Call Value", false);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	jMenu.add(subMenu);
	//######
	//End - Set the metric options.
	//######

	box = new JCheckBoxMenuItem("Display Sliders", false);
	box.addActionListener(actionListener);
	jMenu.add(box);
    }

    public static void usereventOptionMenuItems(JMenu jMenu, ActionListener actionListener){
	JMenu subMenu = null;
	ButtonGroup group = null;
	JCheckBoxMenuItem box = null;
	JRadioButtonMenuItem button = null;

	box = new JCheckBoxMenuItem("Decending Order", true);
	box.addActionListener(actionListener);
	jMenu.add(box);

	//######
	//Set the metric options.
	//######
	subMenu = new JMenu("Select Metric");
	group = new ButtonGroup();

	button = new JRadioButtonMenuItem("Number of Userevents", true);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	button = new JRadioButtonMenuItem("Min. Value", false);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	button = new JRadioButtonMenuItem("Max. Value", false);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);

	button = new JRadioButtonMenuItem("Mean Value", false);
	button.addActionListener(actionListener);
	group.add(button);
	subMenu.add(button);
	jMenu.add(subMenu);
	//######
	//End - Set the metric options.
	//######

	box = new JCheckBoxMenuItem("Display Sliders", false);
	box.addActionListener(actionListener);
	jMenu.add(box);
    }

    public static void windowMenuItems(JMenu jMenu, ActionListener actionListener){
	JMenuItem menuItem = null;

	menuItem = new JMenuItem("Show Function Ledger");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);
	
	menuItem = new JMenuItem("Show Group Ledger");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);
	
	menuItem = new JMenuItem("Show User Event Ledger");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);
	
	menuItem = new JMenuItem("Show Call Path Relations");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);
	
	menuItem = new JMenuItem("Close All Sub-Windows");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);
    }

    public static void helpMenuItems(JMenu jMenu, ActionListener actionListener){
	JMenuItem menuItem = null;

	menuItem = new JMenuItem("Show Help Window");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);
	
	menuItem = new JMenuItem("About ParaProf");
	menuItem.addActionListener(actionListener);
	jMenu.add(menuItem);
    }
}
