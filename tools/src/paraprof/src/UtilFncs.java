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
		UtilFncs.systemError(e, null, "UF01");
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
    public static String getOutputString(int type, double d, int precision){
	switch(type){
	case 0:
	    return (Double.toString(UtilFncs.adjustDoublePresision(d, precision)));
	case 1:
	    return (Double.toString(UtilFncs.adjustDoublePresision((d/1000), precision)));
	case 2:
	    return (Double.toString(UtilFncs.adjustDoublePresision((d/1000000), precision)));
	case 3:
	    int hr = 0;
	    int min = 0;
	    hr = (int) (d/3600000000.00);
	    //Calculate the number of microseconds left after hours are subtracted.
	    d = d-hr*3600000000.00;
	    min = (int) (d/60000000.00);
	    //Calculate the number of microseconds left after minutess are subtracted.
	    d = d-min*60000000.00;
	    return (Integer.toString(hr)+":"+Integer.toString(min)+":"+Double.toString(UtilFncs.adjustDoublePresision((d/1000000), precision)));
	default:
	    UtilFncs.systemError(null, null, "Unexpected string type - UF02 value: " + type);
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
	    UtilFncs.systemError(null, null, "Unexpected string type - UF03 value: " + type);
	}
	return null;
    }

    public static String getValueTypeString(int type){
	switch(type){
	case 2:
	    return "exclusive";
	case 4:
	    return "inclusive";
	case 6:
	    return "number of calls";
	case 8:
	    return "number of subroutines";
	case 10:
	    return "per call value";
	case 12:
	    return "number of userevents";
	case 14:
	    return "min";
	case 16:
	    return "max";
	case 18:
	    return "mean";
	default:
	    UtilFncs.systemError(null, null, "Unexpected string type - UF04 value: " + type);
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
	//Set the value type options.
	//######
	subMenu = new JMenu("Select Value Type");
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
	//End - Set the value type options.
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
	//Set the value type options.
	//######
	subMenu = new JMenu("Select Value Type");
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
	//End - Set the value type options.
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

    //The component argument is passed to the showOpenDialog method.  It is ok if it is null.
    //The functionality that should be expected is as follows: Multiple files in a single directory
    //may be chosen, however, if a directory is selected, then no other files (or directories) may 
    //be chosen. Thus, if files.length > 1, it must contain only files.  If files.length == 1 then
    //it can be a file or a directory.
    //Returns a MetricFileList array of length 0 if no files are obtained.
    public static Vector getFileList(Component component, int type){
	try{
	    File[] files = new File[0];
	    File  file = null;
	    Vector result = new Vector();

	    if(type>3)
		UtilFncs.systemError(new ParaProfError("UF05","File Selection Error!",
						       "Internal error: Unexpected file type - value: " + type, null,
						       component, true), null, null);

	    JFileChooser jFileChooser = new JFileChooser(System.getProperty("user.dir"));
	    if((jFileChooser.showOpenDialog(component)) == JFileChooser.APPROVE_OPTION){
		files = jFileChooser.getSelectedFiles();
		if(files != null){
		    //Validate the selection.
		    if(files.length == 0){
			UtilFncs.systemError(new ParaProfError("UF06","File Selection Error!",
							       "No files selected.", null,
							       component, false), null, null);
			return result;
		    }
		    else if(files.length > 1){
			for(int i=0;i<files.length;i++){
			    if(files[i].isDirectory()){
				UtilFncs.systemError(new ParaProfError("UF07","File Selection Error!",
							       "Please chose one or more files OR a single directory.", null,
							       component, false), null, null);
				return result;
			    }
			}
		    }
		    
		    //If hear, selection is valid.
		    if(files.length == 1){
			if(files[0].isDirectory()){
			    //First try to find a pprof.dat file or profile.*.*.* files in this directory.
			    switch(type){
			    case 0:
				files = UtilFncs.fileListHelper(files[0], type);
				if(files!=null)
				    result.add(files);
				break;
			    case 1:
				files = files[0].listFiles();
				Vector v = new Vector();
				for(int i = 0;i<files.length;i++){
				    if(files[i] != null){
					if((files[i].isDirectory())&&(files[i].getName().indexOf("MULTI__") != -1))
					    v.add(files[i]);
				    }
				}
				int length = v.size();
				if(length!=0){
				    for(int i=0;i<length;i++){
					file = (File)(v.elementAt(i));
					files = UtilFncs.fileListHelper(file, type);
					if(files!=null)
					    result.add(files);
				    }
				}
				break;
			    default:
				UtilFncs.systemError(new ParaProfError("UF08","File Selection Error!",
							       "File selection/File Type mismatch", null,
							       component, false), null, null);
				break;
			    }
			}
			else{
			    switch(type){
			    case 0:
				result.add(files);
				break;
			    case 1:
				result.add(files);
				break;
			    default:
				UtilFncs.systemError(new ParaProfError("UF09","File Selection Error!",
							       "File selection/File Type mismatch", null,
							       component, false), null, null);
				break;
			    }
			}
		    }
		    else{ //More than one file in selection (already checked for zero).
			switch(type){
			case 0:
			    break;
			case 1:
			    result.add(files);
			    break;
			default:
			    UtilFncs.systemError(new ParaProfError("UF10","File Selection Error!",
							       "File selection/File Type mismatch", null,
							       component, false), null, null);
			    break;
			}
		    }
		}
	    }
	    UtilFncs.systemError(new ParaProfError("UF11", "File selection cancelled by user",null),null,null);
		return result;
	    }
	catch(Exception e){
	    //Did not find anything that could be used.
	    ParaProfError paraProfError = new ParaProfError();
	    paraProfError.location = "UF08";
	    paraProfError.popupString = "File Selection Error!";
	    paraProfError.s0 = "An error has been detected.";
	    paraProfError.exp = e;
	    paraProfError.component = component;
	    paraProfError.showPopup = true;
	    paraProfError.showContactString = true;
	    paraProfError.quit = true;
	    UtilFncs.systemError(paraProfError, null, null);
	}
	return new File[0];
    }

    //This function helps the getFileList function above. It looks in the given directory
    //for a pprof.dat file, or for a list of profile.*.*.* files (in that order).
    //If nothing is found, it returns null.
    public static File[] fileListHelper(File directory, int type){
	if(directory.isDirectory()){
	    File[] files = new File[0];
	    File  file = null;
	    String directoryPath = directory.getCanonicalPath();
	    String fileSeparator = System.getProperty("file.separator");
	    
	    switch(type){
	    case 0:
		file = new File(directoryPath + fileSeparator + "pprof.dat");
		if(file.exists()){
		    System.out.println("Found pprof.dat!");
		    files = new File[1];
		    files[0] = file;
		    return files;
		}
		break;
	    case 1:
		files = directory.listFiles();
		Vector v = new Vector();
		for(int i = 0;i<files.length;i++){
		    if(files[i] != null){
			if(files[i].getName().indexOf("profile.") != -1)
			    v.add(files[i]);
		    }
		}
		int length = v.size();
		if(length!=0){
		    files = new File[length];
		    for(int i=0;i<length;i++){
			files[i] = (File) v.elementAt(i);
		    }
		    return files;
		}
		break;
	    default:
		return null;
	    }
	}
	return null;
    }

    public static void systemError(Object obj, Component component, String string, boolean debug){ 
	System.out.println("####################################");
	boolean quit = true; //Quit by default.
	if(obj != null){
	    if(obj instanceof Exception){
		Exception exception = (Exception) obj;
		if(debug){
		    System.out.println(exception.toString());
		    exception.printStackTrace();
		    System.out.println("\n");
		}
		System.out.println("An error was detected: " + string);
		System.out.println(ParaProfError.contactString);
	    }
	    if(obj instanceof ParaProfError){
		ParaProfError paraProfError = (ParaProfError) obj;
		if(debug){
		    if((paraProfError.showPopup)&&(paraProfError.popupString!=null))
			JOptionPane.showMessageDialog(paraProfError.component,
						      "ParaProf Error", paraProfError.popupString, JOptionPane.ERROR_MESSAGE);
		    if(paraProfError.exp!=null){
			System.out.println(paraProfError.exp.toString());
			paraProfError.exp.printStackTrace();
			System.out.println("\n");
		    }
		    if(paraProfError.location!=null)
			System.out.println("Location: " + paraProfError.location);
		    if(paraProfError.s0!=null)
			System.out.println(paraProfError.s0);
		    if(paraProfError.s1!=null)
			System.out.println(paraProfError.s1);
		    if(paraProfError.showContactString)
			System.out.println(ParaProfError.contactString);
		}
		else{
		    if((paraProfError.showPopup)&&(paraProfError.popupString!=null))
			JOptionPane.showMessageDialog(paraProfError.component,
						      "ParaProf Error", paraProfError.popupString, JOptionPane.ERROR_MESSAGE);
		    if(paraProfError.location!=null)
			System.out.println("Location: " + paraProfError.location);
		    if(paraProfError.s0!=null)
			System.out.println(paraProfError.s0);
		    if(paraProfError.s1!=null)
			System.out.println(paraProfError.s1);
		    if(paraProfError.showContactString)
			System.out.println(ParaProfError.contactString);
		}
		quit = paraProfError.quit;
	    }
	    else{
		System.out.println("An error has been detected: " + string);
	    }
	}
	else{
	    System.out.println("An error was detected at " + string);
	}
	System.out.println("####################################");
	if(quit)
	    System.exit(0);
    }
}
