 /* 
   FileList.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  Some useful functions for the system.
*/

package edu.uoregon.tau.dms.dss;

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
    public Vector getFileList(File f, Component component, int type, String filePrefix, boolean debug){
	Vector result = new Vector();

	//Check to see if type is valid.
	if(type>6 && (type <= 100 || type > 101)){
	    System.out.println("Unexpected Type -  " + type + ":");
	    System.out.println("Location - ParaProfManager.getFileList(...) 0");
	    return new Vector();
	}

	try{
	    File[] selection = null;
	    File[] files = new File[0];
	    File  file = null;
	    JFileChooser jFileChooser = null;

	    //If a file was not passed in, prompt user for a selection.
	    if(f==null){
		jFileChooser = new JFileChooser(System.getProperty("user.dir"));
		jFileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
		jFileChooser.setMultiSelectionEnabled(true);
		if((jFileChooser.showOpenDialog(component)) != JFileChooser.APPROVE_OPTION){
		    System.out.println("File selection cancelled by user!");
		    return new Vector();
		}
		//User clicked the approve option.  Grab the selection.
		selection = jFileChooser.getSelectedFiles(); //Note that multiple selection must have been enabled.
		//Validate the selection.  See above method description for an explanation.
		if(selection.length == 0){
		    System.out.println("No files selected!");
		    return result;
		}
		else if(selection.length > 1){
		    for(int i=0;i<selection.length;i++){
			if(selection[i].isDirectory()){
			    JOptionPane.showMessageDialog(component,"Choose one or more files OR a single directory",
							  "File Selection Error",
							  JOptionPane.ERROR_MESSAGE);
			    return result;
			}
		    }
		}
	    }
	    else{
		selection = new File[1];
		selection[0] = f;
	    }

	    path = selection[0].getPath();

	    //If here, selection is valid.
	    if(selection.length == 1){
		if(selection[0].isDirectory()){
		    if(filePrefix==null){
			//Get prefix.
			switch(type){
			case 0:
			    filePrefix = JOptionPane.showInputDialog("Enter file prefix", "pprof");
			    if((filePrefix == null) || "".equals(filePrefix))
				filePrefix = "pprof";
			    break;
			case 1:
			    filePrefix = JOptionPane.showInputDialog("Enter file prefix", "profile");
			    if((filePrefix == null) || "".equals(filePrefix))
				filePrefix = "profile";
			    break;
			case 2:
			    filePrefix = JOptionPane.showInputDialog("Enter file prefix");
			    if((filePrefix == null) || "".equals(filePrefix)){
				System.out.println("No prefix given!");
				return new Vector();
			    }
			    break;
			case 3:
			case 4:
			case 5:
			case 6:
			case 101:
			    filePrefix = JOptionPane.showInputDialog("Enter file prefix");
			    if((filePrefix == null) || "".equals(filePrefix)){
				System.out.println("No prefix given!");
				return new Vector();
			    }
			    break;
			default:
			    break;
			}
		    }

		    if(type==0 || type==1){
			//First try and find a .dat file in the selected directory.
			files = this.helperGetFileList(selection[0], type, filePrefix, debug);
			if(files.length > 0)
			    result.add(files);
			else{
			    files = selection[0].listFiles();
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
				    files = this.helperGetFileList(file, type, filePrefix, debug);
				    if(files.length > 0)
					result.add(files);
				}
			    }
			}
		    }
		    else if(type==2 || type==3 || type==4 || type== 5 || type==6 || type==101){
			files = this.helperGetFileList(selection[0], type, filePrefix, debug);
			if(files.length > 0)
			    result.add(files);
		    }
		}
		else{
		    result.add(selection);
		}
	    }
	    else{ //More than one file in selection (already checked for zero).
		result.add(selection);
	    }
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

    //This function helps the getFileList function above. 
    //Currently, It looks in the given directory for a pprof.dat file,
    //or for a list of profile.*.*.* files (switching based on the type argument).
    //If nothing is found, it returns an empty File[].
    public File[] helperGetFileList(File directory, int type, String filePrefix, boolean debug){
	File[] files = new File[0];

	if(directory.isDirectory()){
	    File  file = null;
	    String fileSeparator = null;
	    String directoryPath = null;
	    
	    try{
		fileSeparator = System.getProperty("file.separator");
		directoryPath = directory.getCanonicalPath();

		if(debug){
		    System.out.println("####################################");
		    System.out.println("DEBUG MESSAGE");
		    System.out.println("FileList.helperGetFileList(...):");
		    System.out.println("type: " + type);
		    System.out.println("directry:" + directoryPath);
		    System.out.println("End - DEBUG MESSAGE");
		    System.out.println("####################################");
		}

		if(type==0){
		    file = new File(directoryPath + fileSeparator + filePrefix + ".dat");
		    if(file.exists()){
			files = new File[1];
			files[0] = file;
		    }
		}
		else if(type==1||type==2||type==3||type==4||type==5||type==6){
		    files = directory.listFiles();
		    Vector v = new Vector();
		    for(int i = 0;i<files.length;i++){
			if(files[i] != null){
				if(files[i].getName().indexOf(filePrefix) == 0) {
					v.add(files[i]);
				}
			}
		    }
		    int length = v.size();
		    files = new File[length]; //Important to reset files here.
		    if(length!=0){
			for(int i=0;i<length;i++){
			    files[i] = (File) v.elementAt(i);
			}
		    }
		}
		else if(type==101){
			int index = 0;
		    Vector v = new Vector();
			file = new File (directoryPath + fileSeparator + filePrefix + fileSeparator + index + fileSeparator + "output");
			while (file.exists()) {
				v.add(file);
				index++;
				file = new File (directoryPath + fileSeparator + filePrefix + fileSeparator + index + fileSeparator + "output");
			}
		    int length = v.size();
		    files = new File[length]; //Important to reset files here.
		    if(length!=0){
				for(int i=0;i<length;i++){
			    	files[i] = (File) v.elementAt(i);
				}
		    }
		}
		else{
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

    public String getPath(){
	return path;
    }

    public static String getPathReverse(String string){
	String fileSeparator = System.getProperty("file.separator");
	String reverse = "";

	StringTokenizer st = new StringTokenizer(string, fileSeparator);
	while(st.hasMoreTokens()){
	    String token = st.nextToken();
	    reverse = token+fileSeparator+reverse;
	}
	return reverse;
    }

    //For testing purposes.
    public static void main(String args[]){
	boolean debug = false;
	int type = 0; //Pass in a valid type by default. This type represents: "Pprof -d File".
	String filePrefix = null;

	//Process command line arguments.
	try{
	    int position = 0;
	    String argument = null;
	    //Deal with help and debug individually, then the rest.
	    //Help
	    while (position < args.length) {
		argument = args[position++];
		if (argument.equalsIgnoreCase("HELP")) {
                    System.out.println("paraprof/FileList filetype [0-9]+ | prefix \"filename prefix\" | help | debug");
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
		else if (argument.equalsIgnoreCase("PREFIX")){
			argument = args[position++];
			filePrefix = argument;
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
	    
	try{
	    FileList fl = new FileList();
	    File[] files = null;
	    Vector v = fl.getFileList(null, null,type,filePrefix,debug);
	    System.out.println("####################################");
	    System.out.println("Files found:");
	    for(Enumeration e = v.elements(); e.hasMoreElements() ;){
		files = (File[]) e.nextElement();
		for(int i=0;i<files.length;i++){
		    System.out.println(files[i].getCanonicalPath());
		}
	    }
	    System.out.println("####################################");
	}
	catch(NullPointerException e){
	    System.out.println("An error occurred getting file list:");
	    System.out.println("Location - FileList.main(...)");	    
	    if(debug)
		e.printStackTrace();
	}
	catch(SecurityException e){
	    System.out.println("An error occurred getting file list:");
	    System.out.println("Location - ParaProfManager.helperGetFileList(...)");
	    if(debug)
		e.printStackTrace();
	}
	catch(ArrayIndexOutOfBoundsException e){
	    System.out.println("An error occurred getting file list:");
            System.out.println("Location - FileList.main(...)");
            if(debug)
                e.printStackTrace();
        }
	catch(IllegalArgumentException e){
	    System.out.println("An error occurred getting file list:");
	    System.out.println("Location - ParaProfManager.helperGetFileList(...)");
	    if(debug)
		e.printStackTrace();
	}
	catch(IOException e){
	    System.out.println("An error occurred getting file list:");
	    System.out.println("Location - ParaProfManager.helperGetFileList(...)");
	    if(debug)
		e.printStackTrace();
	}

	System.exit(0);
    }

    //####################################
    //Instance data.
    //####################################
    //This stores the path to the last file list
    //obtained by a call to FileList.getFileList(...).
    String path = null;
    //####################################
    //End - Instance data.
    //####################################
}
