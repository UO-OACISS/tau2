/* 
   ParaProfError.java

   Title:      ParaProf
   Author:     Robert Bell
   
   Provides a more extensive set of fields for better error output.

   Things to do: Class is complete.
*/

package paraprof;

import java.awt.*;

public class ParaProfError{
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
    public boolean showContactString = false; //Indicates whether the contact string is printed.
    public boolean quit = true; //Indicates whether this error is serious
                                 //enough to quit the system or not.
    //####################################
    //End - Instance Data.
    //####################################
}
