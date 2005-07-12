package edu.uoregon.tau.vis;

import java.awt.*;

import net.java.games.jogl.GL;

/**
 * Various static tools for the vis package
 * 
 * <P>CVS $Id: VisTools.java,v 1.1 2005/07/12 18:02:17 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public class VisTools {

    // Exception handler for Swing elements
    private static ExceptionHandler exceptionHandler;

    /**
     * Helper function to call glColor3f.
     * @param gl GL object to use.
     * @param c Color to use.
     */
    public static void glSetColor(GL gl, Color c) {
        // no alpha for now
        gl.glColor3f(c.getRed() / 255.0f, c.getGreen() / 255.0f, c.getBlue() / 255.0f);
    }

    /**
     * calls glSetColor on the given VisRenderer with the given Color, 
     * or it's inverse, depending on the VisRenderer's state
     * @param visRenderer 
     * @param c color
     */
    public static void glApplyInvertableColor(VisRenderer visRenderer, Color c) {
        if (visRenderer.getReverseVideo()) {
            glSetColor(visRenderer.getGLDrawable().getGL(), invert(c));
        } else {
            glSetColor(visRenderer.getGLDrawable().getGL(), c);
        }
    }

    /**
     * Inverts a color
     * @param c color to invert
     * @return inverted color
     */
    public static Color invert(Color c) {
        return new Color(255 - c.getRed(), 255 - c.getGreen(), 255 - c.getBlue());
    }

    /**
     * Outputs the given string on stderr if the 'vis.verbose' property is set
     * @param string string to output
     */
    public static void verr(String string) {
        if (System.getProperty("vis.verbose") != null) {
            System.err.println(string);
        }
    }

    /**
     * Outputs the given string on stdout if the 'vis.verbose' property is set
     * @param string string to output
     */
    public static void vout(String string) {
        if (System.getProperty("vis.verbose") != null) {
            System.out.println(string);
        }
    }

    /**
     * Outputs the given string along with the given object's class name
     * on stdout if the 'vis.verbose' property is set
     * @param obj the object 
     * @param string string to output
     */
    public static void vout(Object obj, String string) {
        if (System.getProperty("vis.verbose") != null) {
            String className = obj.getClass().getName();
            int lastDot = className.lastIndexOf('.');
            if (lastDot != -1) {
                className = className.substring(lastDot + 1);
            }
            System.out.println(className + ": " + string);
        }
    }

    /**
     * Outputs the given string along with the given object's class name
     * on stderr if the 'vis.verbose' property is set
     * @param obj the object 
     * @param string string to output
     */
    public static void verr(Object obj, String string) {
        if (System.getProperty("vis.verbose") != null) {
            String className = obj.getClass().getName();
            int lastDot = className.lastIndexOf('.');
            if (lastDot != -1) {
                className = className.substring(lastDot + 1);
            }
            System.err.println(className + ": " + string);
        }
    }

    /**
     * Helper method to add Components to JPanels
     */
    static void addCompItem(Container jPanel, Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        jPanel.add(c, gbc);
    }

    
    /**
     * Performs and epsilon comparison of two doubles.  Returns true if x0 and x1 are within epsilon.
     * @param x0 the first double.
     * @param x1 the second double.
     * @param epsilon the epsilon to use.
     * @return whether or not x0 and x1 differ by less than epsilon.
     */
    public static boolean isSufficientlyEqual(double x0, double x1, double epsilon) {
        return (Math.abs(x0 - x1) < epsilon);
    }

    /**
     * Calls isSufficienctly equal with and epsilon of 1e-2.
     * @see #isSufficientlyEqual(double, double, double)
     */
    public static boolean isSufficientlyEqual(double x0, double x1) {
        return (isSufficientlyEqual(x0, x1, 1e-2));
    }
    
    /**
     * Sets the exception handler for the ActionListeners of the Swing control panels.
     * If you're not using the control panels, then you needn't use this.
     * @param eh    ExceptionHandler to use
     * @see         VisTools
     */
    public static void setSwingExceptionHandler(ExceptionHandler eh) {
        exceptionHandler = eh;
    }

    /**
     * Called internally by Swing Listeners when an exception occurrs
     * @param e     Exception
     */
    public static void handleException(Exception e) {
        if (exceptionHandler != null) {
            exceptionHandler.handleException(e);
        } else {
            throw new RuntimeException(e);
        }
    }
}
