package edu.uoregon.tau.common;

import org.python.core.PyException;
import org.python.core.PyInteger;
import org.python.core.PyObject;
import org.python.util.PythonInterpreter;


public class TauScripter {


    private static PythonInterpreter interpreter = new PythonInterpreter();

    
    public static void execfile(String string) {
        interpreter.execfile(string);
    }
    
    public static void exec(String string) {
        interpreter.exec(string);
    }
    
    
}
