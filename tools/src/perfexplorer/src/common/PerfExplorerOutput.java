package edu.uoregon.tau.perfexplorer.common;

/**
 * Wrapper class for supporting console output.
 *
 * <P>CVS $Id: PerfExplorerOutput.java,v 1.3 2009/02/24 00:53:37 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class PerfExplorerOutput {
    private static boolean _quiet = false;

    /**
     * Private Constructor. 
     * 
     */
    private PerfExplorerOutput () {
    }

    /**
     * Static accessor method.
     * 
     */
    public static void setQuiet(boolean quiet)  {
        _quiet = quiet;
    }

    /**
     * Method to selectively print out the message, with no newline.
     * 
     * @param message
     */
    public static void print(String message) {
        if (!_quiet)
            System.out.print(message);
    }

    /**
     * Method to selectively print out the message, with a newline.
     * 
     * @param message
     */
    public static void println(String message) {
        if (!_quiet)
            System.out.println(message);
    }

    /**
     * Method to selectively print out a newline.
     * 
     * @param message
     */
    public static void println() {
        if (!_quiet)
            System.out.println();
    }
    
    /**
     * Method to selectively print out an int without a newline.
     * 
     * @param message
     */
    public static void print(int x) {
        if (!_quiet)
            System.out.print(x);
    }

    /**
     * Method to selectively print out an int with a newline.
     * 
     * @param message
     */
    public static void println(int x) {
        if (!_quiet)
            System.out.println(x);
    }
}

