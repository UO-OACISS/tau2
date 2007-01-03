package common;

import java.util.*;

/**
 * Class for managing interpreter drivers.
 * This manager is responsible for keeping track of loaded
 * driver. Interpreter drivers are required to register instance
 * of themselves with this manager when they are loaded.
 * This code was taken from a Java World example located here:
 * http://www.javaworld.com/javaworld/jw-10-1999/jw-10-script_p.html
 * 
 * @author Ramnivas Laddad
 */
public class InterpreterDriverManager {
    private static Map _extensionDriverMap = new HashMap();
    private static Map _languageDriverMap = new HashMap();

    private static final char EXTENSION_SEPARATOR = '.';

    /**
     * Private constructor
     * There is no need for instantiating this class as all methods
     * are private. This private constructor is to disallow creating
     * instances of this class.
     */
    private InterpreterDriverManager() {
    }
	
    /**
     * Register a driver.
     * Interpreter drivers call this method when they are loaded.
     * @param driver the driver to be registered
     */
    public static void registerDriver(InterpreterDriver driver) {
	String[] extensions = driver.getSupportedExtensions();
	for (int size = extensions.length, i = 0; i < size; i++) {
	    _extensionDriverMap.put(extensions[i], driver);
	}
	String[] languages = driver.getSupportedLanguages();
	for (int size = languages.length, i = 0; i < size; i++) {
	    _languageDriverMap.put(languages[i], driver);
	}
    }

    /**
     * Execute a script string
     * Execute the string supplied according to the langauge specified
     * @param script script to be executed
     * @param language language for interpreting the script string
     */
    public static void executeScript(String script, String language) 
	throws InterpreterDriver.InterpreterException {
	InterpreterDriver driver 
	    = (InterpreterDriver)_languageDriverMap.get(language);
	if (driver == null) {
	    System.out.println("No driver installed to handle language " 
			       + language);
	    return;
	}
	driver.executeScript(script);
    }

    /**
     * Exceute a script file.
     * The interpreter driver supporting the language for this file
     * is deduced from file name extension
     * @param scriptFile file name containing script
     */
    public static void executeScriptFile(String scriptFile)
	throws InterpreterDriver.InterpreterException {
	String extension
	    = scriptFile.substring(scriptFile
				   .lastIndexOf(EXTENSION_SEPARATOR)+1);
	InterpreterDriver driver 
	    = (InterpreterDriver)_extensionDriverMap.get(extension);
	if (driver == null) {
	    System.out.println("No driver installed to handle extension " 
			       + extension);
	    return;
	}
	driver.executeScriptFile(scriptFile);
    }

    /**
     * The main function which exercises the basic functionality.
     * Useful for unit testing.
     */ 
    public static void main (String[] args) {
	try {
	    Class.forName("scripting.FESIInterpreterDriver");
	    Class.forName("scripting.JPythonInterpreterDriver");
	    Class.forName("scripting.JaclInterpreterDriver");
	    Class.forName("scripting.SkijInterpreterDriver");

	    executeScriptFile("test.js");
	    executeScriptFile("test.py");
	    executeScriptFile("test.tcl");
	    executeScriptFile("test.scm");
	} catch (Exception ex) {
	    System.out.println(ex);
	}
    }
}


