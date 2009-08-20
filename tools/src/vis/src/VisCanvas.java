/*
 * VisCanvas.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

import java.awt.Canvas;
import java.awt.event.KeyListener;

import javax.media.opengl.GLCanvas;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLException;

/**
 * This class is merely a wrapper over GLCanvas which allows users of the Vis
 * package to build against vis alone (not jogl). 
 *
 * <P>CVS $Id: VisCanvas.java,v 1.12 2009/08/20 22:09:35 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.12 $
 */
public class VisCanvas {

    private GLCanvas glCanvas;

    /**
     * Creates a <tt>VisCanvas</tt> with the given <tt>VisRenderer</tt>.
     * 
     * @param visRenderer <tt>VisRenderer</tt> to use.
     */
    public VisCanvas(VisRenderer visRenderer) {

        GLCapabilities caps = new GLCapabilities();

        //FSAA (Full Screen Anti-Aliasing)
        if (visRenderer.getFSAA()) {
            caps.setSampleBuffers(true);
            caps.setNumSamples(4);
            caps.setHardwareAccelerated(true);
        }

        boolean tryStereo = true;
        // Fisher Price machines segfault if you try to use stereo 
        String os = System.getProperty("os.name").toLowerCase();
        if (os.startsWith("mac os x")) {
            tryStereo = false;
        }

        // ask for stereo, if available
        try {
            caps.setStereo(tryStereo);
            glCanvas = new GLCanvas(caps);
        } catch (GLException gle) {
            caps.setStereo(false);
            glCanvas = new GLCanvas(caps);
        }

        glCanvas.setSize(200, 200);
        glCanvas.addGLEventListener(visRenderer);

        // for testing
        //canvas.addGLEventListener(new Gears.GearRenderer());

        //visRenderer.set

    }

    /**
     * Adds a key listener to the underlying glCanvas.
     * @param keyListener the key listener to add.
     */
    public void addKeyListener(KeyListener keyListener) {
        glCanvas.addKeyListener(keyListener);
    }

    /**
     * Returns the actual GLCanvas as a Canvas.
     * @return the actual GLCanvas as a Canvas.
     */
    public Canvas getActualCanvas() {
        return glCanvas;
    }

    /**
     * Returns the height of the <tt>Canvas</tt>.
     * @return the height of the <tt>Canvas</tt>.
     */
    public int getHeight() {
        return glCanvas.getHeight();
    }

    /**
     * Returns the width of the <tt>Canvas</tt>.
     * @return the width of the <tt>Canvas</tt>.
     */
    public int getWidth() {
        return glCanvas.getWidth();
    }

}
