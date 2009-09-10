package edu.uoregon.tau.common;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.python.core.Py;
import org.python.core.PyException;
import org.python.core.PyInteger;
import org.python.core.PyModule;
import org.python.core.PyObject;
import org.python.core.PySystemState;
import org.python.core.imp;
import org.python.util.PythonInterpreter;

/** Loads Jython interpreting environments and preloads them with the needed
 * information.
 * 
 * This Jython environment creation indirection is especially needed due to the
 * problems with class path loading in Web Started applications.
 * <p>
 * The factory can be instantiated. There is, however, a publically visible
 * default factory in it which should be sufficient for most setups. Normally,
 * you will just have to use that one.
 * 
 * @author cantamen/Dirk Hillbrecht. The code in this class is in the Public Domain. Use it and/or put it under whatever license fits your needs.
 * @see http://forum.java.sun.com/thread.jspa?threadID=786457&messageID=4471253
 */
public class PythonInterpreterFactory {

    /** Default implementation which should be enough for most usage cases. */
    public static PythonInterpreterFactory defaultfactory = new PythonInterpreterFactory();

    /** List of Java packages to preload into the delivered Jython interpreter
     * environments. */
    private List packages;

    /** Create a new interpreter factory with an empty list. */
    public PythonInterpreterFactory() {
        packages = new LinkedList();
    }

    /** Add a single java package name into the internal package list.
     * 
     * @param packagename Name of a Java package (like "javax.swing").
     */
    public void addPackage(String packagename) {
        packages.add(packagename);
    }

    /** Add a list of java packages.
     * 
     * The names of the packages to be added are expected line by line in the
     * passed stream.
     * <p>
     * This method is tailored for being called with the output of 
     * <tt>{@link ClassLoader#getResourceAsStream(String)}</tt>. The resource is
     * a text file of package names which is put into a JAR archive file. This
     * file can be loaded via Java Web Start. So, the Jython environment can
     * obtain knowledge about the Java packages within the Web Start application
     * without reading actually reading the JAR files.  
     * 
     * @param packagenamestream InputStream which reads a text file, possibly obtained via ClassLoader. 
     */
    public void addPackagesFromStream(InputStream packagenamestream) {
        if (packagenamestream != null) {
            BufferedReader br = new BufferedReader(new InputStreamReader(packagenamestream));
            try {
                String line;
                while ((line = br.readLine()) != null)
                    addPackage(line);
                br.close();
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }
        }
    }

    public void addPackagesFromList(List packages) {
        if (packages == null) {
            return;
        }
        for (Iterator i = packages.iterator(); i.hasNext();) {
            this.packages.add(i.next());
        }
    }

    /** Return a jython environment which knows about the additional Java
     * packages.
     * 
     * The method can be used as a drop-in replacement for the instantiation in
     * code like "<tt>PythonInterpreter pi=new PythonInterpreter()</tt>".
     * The returned interpreter will know about the additional Java packages.
     * 
     * @return A freshly instantiated PythonInterpreter instance which knows about the additional Java packages.
     */
    public PythonInterpreter getPythonInterpreter() {
        PythonInterpreter pythoninterpreter = new PythonInterpreter();
        PyModule mainmodule = imp.addModule("__main__");
        pythoninterpreter.setLocals(mainmodule.__dict__);
        PySystemState sys = Py.getSystemState();
        for (Iterator i = packages.iterator(); i.hasNext();) {
            PySystemState.add_package((String) i.next());
        }
        // set standard out and standard error for the interpreter
        pythoninterpreter.setErr(System.err);
        pythoninterpreter.setOut(System.out);
        return pythoninterpreter;
    }

}
