/* 
  
  ParaProfImageOutput.java
  
  Title:       ParaProfImageOutput.java
  Author:      Robert Bell
  Description: Handles the output of the various panels to image files.
*/

package edu.uoregon.tau.paraprof;

import java.util.*;
import java.awt.*;
import java.awt.image.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import edu.uoregon.tau.dms.dss.*;

public class ParaProfImageOutput{

    public ParaProfImageOutput(){
    }

    public void saveImage(ParaProfImageInterface ref){
	try{
	    JOptionPane.showMessageDialog(null, "Jar compiled for jdk-ver < 1.4. Image support not available.",
						      "Image Error!",
						      JOptionPane.ERROR_MESSAGE);
	}
	catch(Exception e){
	}
    }
}
