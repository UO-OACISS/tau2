/* 
   ParaProfImageFormatFileFilter.java

   Title:      ParaProf
   Author:     Robert Bell
   
   Description: A file filter for the different image format types Pararof supports.

   Things to do: Class is complete.
*/

package paraprof;

import java.io.*;

public class ParaProfImageFormatFileFilter extends javax.swing.filechooser.FileFilter{

    public ParaProfImageFormatFileFilter(String extension){
	super();
	this.extension = extension;
    }

    public boolean accept(File f){
	boolean accept = f.isDirectory();
	if(!accept){
	    String extension = ParaProfImageFormatFileFilter.getExtension(f);
	    if(extension!=null)
		accept = this.extension.equals(extension);
	}
	return accept;
    }

    public String getDescription(){
	if(extension.equals("jpg"))
	    return "JPEG File (*.jpg)";
	else if(extension.equals("png"))
	    return "PNG File (*.png)";
	else
	    return "Unknown Extension (*.";
    }

    public String toString(){
	return this.getDescription();}
    
    public String getExtension(){
	return extension;}

    public static String getExtension(File f){
	String s = f.getPath();
	String extension = null;

	int i = s.lastIndexOf('.');
	if(i>0 && i<s.length()-1)
	    extension = s.substring(i+1).toLowerCase();
	
	return extension;
    }

    //####################################
    //Instance Data.
    //####################################
    String extension = null;
    //####################################
    //End - Instance Data.
    //####################################
}
