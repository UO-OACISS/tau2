package edu.uoregon.tau.paraprof.vis;

import java.awt.Color;

import net.java.games.jogl.GL;

public class VisTools {

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
        return new Color(255-c.getRed(), 255-c.getGreen(), 255-c.getBlue());
    }
    
}
