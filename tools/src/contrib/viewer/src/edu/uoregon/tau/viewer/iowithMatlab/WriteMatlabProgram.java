/** This class writes two types of Matlab .m file, drawing a curve with x-axis and y-axis 
** values (e.g. a scalability curve) , and drawing a group of curves (e.g. three scalability 
** curves with varied problem size.
**/ 

package edu.uoregon.tau.viewer.iowithMatlab;

import java.io.*;
import java.util.*;

public class WriteMatlabProgram{

    public static String drawACurve(String filename, int[] x, double[] y, String title, String xlabel, String ylabel ) {
		
	int i;
	String s;

	// write x vector
	s = "x=[";
	for(i = 0; i < x.length; i++){
	    s += x[i];
	    s += " ";
	}
	s += "]; ";
	
	// write y vector
	s += "y=[";
	for(i = 0; i < y.length; i++){
	    s += y[i];
	    s += " ";
	}
	s += "]; ";
	
	// plot 
	s += "plot(x,y,'-s','MarkerSize',4,'MarkerFaceColor','auto');  ";
	
	// title
	s += "title('";
	s += title;
	s += "');  ";
	
	// xlabel
	s += "xlabel('";
	s += xlabel;
	s += "'); ";
	
	// ylabel
	s += "ylabel('";
	s += ylabel;
	s += "'); ";
	
	// xtick
	s += "set(gca, 'xtick',x); ";
	
	// axis 
	s += "v = axis; ";
	
	s += "ymax = v(4);  ";
	
	s += "axis([";
	s += x[0];
	s += " ";
	s += x[x.length - 1];
	s += " 0 ymax]);  ";
	
	// create .m file if necessary
	if (filename != null){
	    File mfile;	
	    mfile = new File(filename);
	    if (!mfile.exists()){
		try {
		    if (mfile.createNewFile()){
			System.out.println("Create " + filename + " !");
		    }
		}
		catch(Exception e){
		    e.printStackTrace();	
		}
	    }

	    try{
		BufferedWriter writer = new BufferedWriter(new FileWriter(mfile));
		writer.write(s, 0, s.length());
		writer.close();
	    }
	    catch(Exception e){
		e.printStackTrace();	
	    }

	    return null;
	}

	return s;
    }

    public static String drawACurveGroup(String filename, int[] x, double[][] y, String title, String xlabel, String ylabel, String[] legends) {
	
	int i, k;
	String s;

	// write x vector
	s = "x=[";
	for(i = 0; i < x.length; i++){
	    s += x[i];
	    s += " ";
	}
	s += "]; ";
	
	// write y vector
	s += "y=[";
	for (k =0; k<y[0].length; k++){
	   for(i = 0; i < y.length; i++){
		s += y[i][k];
		s += " ";
	    }
	    if (i!= y.length-1)
		s += ";";
	}
       
	s += "]; ";
	
	// plot 
	s += "plot(x,y,'-s','MarkerSize',4,'MarkerFaceColor','auto');  ";
	
	// title
	s += "title('";
	s += title;
	s += "'); ";
	
	// xlabel
	s += "xlabel('";
	s += xlabel;
	s += "');  ";
	
	// ylabel
	s += "ylabel('";
	s += ylabel;
	s += " (usec)');  ";
	
	// xtick
	s += "set(gca, 'xtick',x);  ";
	
	// axis 
	s += "v = axis;  ";
	
	s += "ymax = v(4); ";
	
	s += "axis([";
	s += x[0];
	s += " ";
	s += x[x.length - 1];
	s += " 0 ymax]);  ";
		
	s += "legend('";
	for (i=0; i<legends.length; i++){
	    s += legends[i] + "'";
	    if (i!=legends.length-1)
		s += ", '";
	}

	s += ");  ";
	
	// create .m file if necessary
	if (filename != null){
	    File mfile;
	
	    mfile = new File(filename);
	    if (!mfile.exists()){
		try {
		    if (mfile.createNewFile()){
			System.out.println("Create " + filename + " !");
		    }
		}
		catch(Exception e){
		    e.printStackTrace();	
		}
	    }

	    try{
		BufferedWriter writer = new BufferedWriter(new FileWriter(mfile));
		writer.write(s, 0, s.length());	
		writer.close();
	    }catch(Exception e){
		e.printStackTrace();	
	    }

	    return null;
	}

	return s;
    }

}
