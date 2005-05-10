package edu.uoregon.tau.paraprof;

import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.ClipboardOwner;
import java.awt.datatransfer.StringSelection;

public class JVMDependent {

    public static final String version = "1.4";
    
    public static void main(String[] args) {
        System.out.println("I was compiled with Java 1.4");
    }

    public static void setClipboardContents(String contents, ClipboardOwner owner) {
        if (contents == null || contents == "")
            return;
        Toolkit tk = Toolkit.getDefaultToolkit();
        StringSelection st = new StringSelection(contents);
        Clipboard cp = tk.getSystemSelection();
        if (cp != null) { // some systems (e.g. windows) don't have a system selection clipboard
            cp.setContents(st, owner);
        }
        cp = tk.getSystemClipboard();
        if (cp != null) {
            cp.setContents(st, owner);
        }
    }
    
    
}
