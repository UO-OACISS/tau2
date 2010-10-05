package edu.uoregon.tau.common;

import org.python.util.PythonInterpreter;

/**
 * TauScriptor.java
 * A simple wrapper around any scripting interpreter.
 *       
 * <P>CVS $Id: TauScripter.java,v 1.2 2006/03/03 02:47:03 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class TauScripter {


    private static PythonInterpreter interpreter = new PythonInterpreter();

    
    public static void execfile(String string) {
        interpreter.execfile(string);
    }
    
    public static void exec(String string) {
        interpreter.exec(string);
    }
    
    
}
