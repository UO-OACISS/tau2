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
    public String s0 = null; //Being used as a code location string.
    public String s1 = null; //Being used as the primary descriptor.
    public String s2 = null;
    public String s3 = null;

    public Exception exp  = null;
    public Component component = null;
    
    public Object obj0 = null; //Additional use.
    public Object obj1 = null; //Additional use.

    public boolean showPopup = false; //Indicates whether it is safe to show a popup window.
                                      //Some methods do not seem to like being interupted - 
                                      //paintComponent methods for example.
    public boolean quit = false; //Indicates whether this error is serious
                                 //enough to quit the system or not.
    //####################################
    //End - Instance Data.
    //####################################
}
