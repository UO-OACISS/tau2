/* 
   Name: ParaProfError.java
   Author:     Robert Bell
   
   Provides a more extensive set of fields for better error output.

   Things to do: Class is complete.
*/

package edu.uoregon.tau.dms.dss;

import java.awt.*;

public class ParaProfError{
    
    public ParaProfError(){}

    //Provide some useful constructors.
    //Everything constructor.
    public ParaProfError(String location, String popupString,
			 String s0, String s1,
			 Exception exp, Component component,
			 Object obj0, Object obj1, boolean showPopup,
			 boolean showContactString, boolean quit){
	this.location = location;
	this.popupString = popupString;
	this.s0 = s0;
	this.s1 = s1;
	this.exp = exp;
	this.component = component;
	this.obj0 = obj0;
	this.obj1 = obj1;
	this.showPopup = showPopup;
	this.showContactString = showContactString;
	this.quit =  quit;
    }
    
    //Show the popup, a message, and an optional quit.
    public ParaProfError(String location, String popupString,
			 String s0, String s1,
			 Component component, boolean quit){
	this.location = location;
	this.popupString = popupString;
	this.s0 = s0;
	this.s1 = s1;
	this.component = component;
	this.showPopup = true;
	if(!quit){
	    this.showContactString = false;
	    this.quit =  false;
	}
    }

    //Just an informational message to the console.
    public ParaProfError(String location, String s0, String s1){
	this.location = location;
	this.s0 = s0;
	this.s1 = s1;
	this.showContactString = false;
	this.quit =  false;
    }

    //####################################
    //Instance Data.
    //####################################
    public String location = null; //code location string.
    public String popupString = null;
    public static String contactString =
	"@@@@@@@@@\n"+
	"@@@ Please email us at: tau-bugs@cs.uoregon.edu\n"+
	"@@@ If possible, include the profile files that caused this error,\n"+
	"@@@ and a brief desciption your sequence of operation.\n"+
	"@@@ Also email this error message,as it will tell us where the error occured.\n"+
	"@@@ Thank you for your help!\n"+
	"@@@@@@@@@";

    public String s0 = null;
    public String s1 = null;

    public Exception exp  = null;
    public Component component = null;
    
    public Object obj0 = null; //Additional use.
    public Object obj1 = null; //Additional use.

    public boolean showPopup = false; //Indicates whether it is safe to show a popup window.
                                      //Some methods do not seem to like being interupted - 
                                      //paintComponent methods for example.
    public boolean showContactString = true; //Indicates whether the contact string is printed.
    public boolean quit = true; //Indicates whether this error is serious
                                //enough to quit the system or not.
    //####################################
    //End - Instance Data.
    //####################################
}
