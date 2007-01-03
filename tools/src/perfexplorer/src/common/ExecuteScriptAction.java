package common;

/**
 * ExecuteScriptAction.java
 *
 *
 * Created: Wed Dec 23 15:22:01 1998
 *
 * This code was taken from a Java World example located here:
 * http://www.javaworld.com/javaworld/jw-10-1999/jw-10-script_p.html
 * 
 * @author Ramnivas Laddad
 * @version
 */

import javax.swing.AbstractAction;
import java.awt.event.ActionEvent;

public class ExecuteScriptAction extends AbstractAction {
    String _scriptFile;
    public ExecuteScriptAction(String name, String scriptFile) {
	super(name);
	_scriptFile = scriptFile;
    }
    
    public void actionPerformed(ActionEvent e) {
	System.out.println("Invoking " + _scriptFile);

	try {
	    InterpreterDriverManager.executeScriptFile(_scriptFile);
	} catch (InterpreterDriver.InterpreterException ex) {
	    System.out.println(ex);
	    ex.printStackTrace();
	}
    }
}
