 /* 
   FileList.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  Some useful functions for the system.
*/

package paraprof;

import java.util.*;
import java.awt.*;
import javax.swing.*;
import java.io.*;

public class FileList{

    public FileList(){}
  
    //The component argument is passed to the showOpenDialog method.  It is ok if it is null.
    //The functionality that should be expected is as follows: Multiple files in a single directory
    //may be chosen, however, if a directory is selected, then no other files (or directories) may 
    //be chosen. Thus, if files.length > 1, it must contain only files.  If files.length == 1 then
    //it can be a file or a directory.
    //Returns a MetricFileList array of length 0 if no files are obtained.
    public Vector getFileList(Component component, int type, boolean debug){
	Vector result = new Vector();

	//Check to see if type is valid.
	if(type>3){
	    System.out.println("Unexpected Type -  " + type + ":");
	    System.out.println("Location - ParaProfManager.getFileList(...) 0");
	    return new Vector();
	}

	try{
	    File[] files = new File[0];
	    File  file = null;

	    JFileChooser jFileChooser = new JFileChooser(System.getProperty("user.dir"));
	    jFileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
	    jFileChooser.setMultiSelectionEnabled(true);
	    if((jFileChooser.showOpenDialog(component)) == JFileChooser.APPROVE_OPTION){
		//User clicked the approve option.  Grab the selection.
		files = jFileChooser.getSelectedFiles(); //Note that multiple selection must have been enabled.
		//Validate the selection.  See above method description for an explanation.
		if(files.length == 0){
		    System.out.println("No files selected!");
		    return result;
		}
		else if(files.length > 1){
		    for(int i=0;i<files.length;i++){
			if(files[i].isDirectory()){
			    JOptionPane.showMessageDialog(component,"Chose one or more files OR a single directory",
							  "File Selection Error",
							  JOptionPane.ERROR_MESSAGE);
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
			    files = this.helperGetFileList(files[0], type, debug);
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
				    files = this.helperGetFileList(file, type, debug);
				    if(files!=null)
					result.add(files);
				}
			    }
			    break;
			default:
			    System.out.println("Unexpected Type -  " + type + ":");
			    System.out.println("Location - ParaProfManager.getFileList(...) 1");
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
			    System.out.println("Unexpected Type -  " + type + ":");
			    System.out.println("Location - ParaProfManager.getFileList(...) 2");
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
			System.out.println("Unexpected Type -  " + type + ":");
			System.out.println("Location - ParaProfManager.getFileList(...) 3");
			break;
		    }
		}
	    }
	    else
		System.out.println("File selection cancelled by user.");
	    return result;
	}
	catch(NullPointerException e){
	    System.out.println("An error occurred getting file list:");
            System.out.println("Location - ParaProfManager.getFileList(...)");
            if(debug)
                e.printStackTrace();
	    return new Vector();
        }
	catch(SecurityException e){
            System.out.println("An error occurred getting file list:");
            System.out.println("Location - ParaProfManager.getFileList(...)");
            if(debug)
                e.printStackTrace();
            return new Vector();
        }
	catch(IllegalArgumentException e){
	    System.out.println("An error occurred getting file list:");
            System.out.println("Location - ParaProfManager.getFileList(...)");
            if(debug)
                e.printStackTrace();
	    return new Vector();
        }
	catch(ArrayIndexOutOfBoundsException e){
            System.out.println("An error occurred getting file list:");
            System.out.println("Location - ParaProfManager.getFileList(...)");
            if(debug)
                e.printStackTrace();
            return new Vector();
        }
	catch(HeadlessException e){
	    System.out.println("An error occurred getting file list:");
            System.out.println("Location - ParaProfManager.getFileList(...)");
	    if(debug)
                e.printStackTrace();
	    return new Vector();
	}
    }

    //This function helps the getFileList function above. It looks in the given directory
    //for a pprof.dat file, or for a list of profile.*.*.* files (in that order).
    //If nothing is found, it returns null.
    public File[] helperGetFileList(File directory, int type, boolean debug){
	File[] files = new File[0];

	if(directory.isDirectory()){
	    File  file = null;
	    String fileSeparator = null;
	    String directoryPath = null;
	    
	    try{
		fileSeparator = System.getProperty("file.separator");
		directoryPath = directory.getCanonicalPath();
		switch(type){
		case 0:
		    file = new File(directoryPath + fileSeparator + "pprof.dat");
		    if(file.exists()){
			files = new File[1];
			files[0] = file;
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
		    }
		    break;
		default:
		    System.out.println("Unexpected Type -  " + type + ":");
		    System.out.println("Location - ParaProfManager.helperGetFileList(...)");
		}
		return files;
	    }
	    catch(NullPointerException e){
		System.out.println("An error occurred getting file list:");
		System.out.println("Location - ParaProfManager.helperGetFileList(...)");
		if(debug)
		    e.printStackTrace();
		return new File[0];
	    }
	    catch(SecurityException e){
                System.out.println("An error occurred getting file list:");
                System.out.println("Location - ParaProfManager.helperGetFileList(...)");
		if(debug)
                    e.printStackTrace();
		return new File[0];
            }
	    catch(IllegalArgumentException e){
		System.out.println("An error occurred getting file list:");
                System.out.println("Location - ParaProfManager.helperGetFileList(...)");
                if(debug)
                    e.printStackTrace();
		return new File[0];
            }
	    catch(IOException e){
		System.out.println("An error occurred getting file list:");
                System.out.println("Location - ParaProfManager.helperGetFileList(...)");
		if(debug)
		    e.printStackTrace();
		return new File[0];
	    }
	}
	else
	    return files;
    }
    
    //For testing purposes.
    public static void main(String args[]){
	boolean debug = false;
	int type = 0; //Pass in a vild type by default. This type represents: "Pprof -d File".

	//Process command line arguments.
	try{
	    int position = 0;
	    String argument = null;
	    //Deal with help and debug individually, then the rest.
	    //Help
	    while (position < args.length) {
		argument = args[position++];
		if (argument.equalsIgnoreCase("HELP")) {
                    System.out.println("paraprof/FileList filetype [0-9]+ | help | debug");
                    System.exit(0);
                }
	    }
	    //Debug
	    position = 0;
	    while (position < args.length) {
                argument = args[position++];
                if (argument.equalsIgnoreCase("DEBUG")) {
                    debug = true;
                }
            }
	    //Now the rest.
	    position = 0;
	    while (position < args.length) {
		argument = args[position++];
		if (argument.equalsIgnoreCase("FILETYPE")){
			argument = args[position++];
			type = Integer.parseInt(argument);
		}
	    }
	}
	catch(NullPointerException e){
	    System.out.println("An error occurred processing command line arguments:");
	    System.out.println("Location - FileList.main(...)");
	    System.err.println("paraprof/FileList filetype [0-9]+ | help | debug");
	    if(debug)
		e.printStackTrace();
	    System.exit(-1);
	}
	catch(ArrayIndexOutOfBoundsException e){
	    System.out.println("An error occurred processing command line arguments:");
            System.out.println("Location - FileList.main(...)");
	    System.err.println("paraprof/FileList filetype [0-9]+ | help | debug");
            if(debug)
                e.printStackTrace();
            System.exit(-1);
	}
	catch(NumberFormatException e){
            System.out.println("An error occurred processing command line arguments:");
            System.out.println("Location - FileList.main(...)");
	    System.err.println("paraprof/FileList filetype [0-9]+ | help | debug");
            if(debug)
                e.printStackTrace();
            System.exit(-1);
        }
	    
	
	FileList fl = new FileList();
	File[] files = null;
	Vector v = fl.getFileList(null,type,debug);
	for(Enumeration e = v.elements(); e.hasMoreElements() ;){
            files = (File[]) e.nextElement();
	    for(int i=0;i<files.length;i++){
		System.out.println(files[i].getName());
	    }
	}
	System.exit(0);
    }
}
