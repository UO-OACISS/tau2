package edu.uoregon.tau.paraprof.vis;

import java.awt.*;

import net.java.games.jogl.GL;

/**
 * Various static tools for the vis package
 * 
 * <P>CVS $Id: VisTools.java,v 1.3 2005/07/11 22:59:53 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.3 $
 */
public class VisTools {

    // Exception handler for Swing elements
    private static ExceptionHandler exceptionHandler;

    // no alpha for now
    public static void glSetColor(GL gl, Color c) {
        gl.glColor3f(c.getRed() / 255.0f, c.getGreen() / 255.0f, c.getBlue() / 255.0f);
    }

    public static void glSetInvertableColor(VisRenderer visRenderer, Color c) {
        if (visRenderer.getReverseVideo()) {
            glSetColor(visRenderer.getGLDrawable().getGL(), invert(c));
        } else {
            glSetColor(visRenderer.getGLDrawable().getGL(), c);
        }
    }

    public static Color invert(Color c) {
        return new Color(255 - c.getRed(), 255 - c.getGreen(), 255 - c.getBlue());
    }

    public static void verr(String string) {
        if (System.getProperty("vis.verbose") != null) {
            System.err.println(string);
        }
    }

    public static void vout(String string) {
        if (System.getProperty("vis.verbose") != null) {
            System.out.println(string);
        }
    }

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

    static void addCompItem(Container jPanel, Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        jPanel.add(c, gbc);
    }

    
    /**
     * Sets the exception handler for the ActionListeners of the Swing control panels
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
