package common;


/**
 * JPythonInterpreterDriver.java
 *
 *
 * Created: Wed Dec 23 16:03:41 1998
 *
 * This code was taken from a Java World example located here:
 * http://www.javaworld.com/javaworld/jw-10-1999/jw-10-script_p.html
 * 
 * @author Ramnivas Laddad
 * @version
 */


import java.io.*;
import org.python.core.*;
import org.python.util.PythonInterpreter; 


/**
 * @author  khuck
 */
public class JPythonInterpreterDriver implements InterpreterDriver {
    private static JPythonInterpreterDriver _instance;

    private PythonInterpreter _interpreter = new PythonInterpreter();

    static {
	_instance = new JPythonInterpreterDriver();
	InterpreterDriverManager.registerDriver(_instance);
    }

    public void executeScript(String script)
	throws InterpreterDriver.InterpreterException {
	try {
	    _interpreter.exec(script);
	} catch (PyException ex) {
	    throw new InterpreterDriver.InterpreterException(ex);
	}
     }

    public void executeScriptFile(String scriptFile) 
	throws InterpreterDriver.InterpreterException {

	try {
	    _interpreter.execfile(scriptFile);
	} catch (PyException ex) {
	    throw new InterpreterDriver.InterpreterException(ex);
	}
    }

    public String[] getSupportedExtensions() {
	return new String[]{"py"};
    }
    
    public String[] getSupportedLanguages() {
	return new String[]{"Python", "JPython"};
    }

    public static void main(String[] args) {
	try {
	    _instance.executeScript("print \"Hello\"");
	    _instance.executeScriptFile("test.py");
	} catch (Exception ex) {
	    System.out.println(ex);
	}
    }
}
