/*
 * VisCanvas.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

import java.awt.Canvas;
import java.awt.event.KeyListener;

import net.java.games.jogl.*;

/**
 * This class is merely a wrapper over GLCanvas which allows users of the Vis
 * package to build against vis alone (not jogl). 
 *
 * <P>CVS $Id: VisCanvas.java,v 1.6 2006/10/31 03:16:28 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.6 $
 */
public class VisCanvas {

    private GLCanvas glCanvas;

    
    /**
     * Creates a <tt>VisCanvas</tt> with the given <tt>VisRenderer</tt>.
     * 
     * @param visRenderer <tt>VisRenderer</tt> to use.
     */
    public VisCanvas(VisRenderer visRenderer) {
     
        GLCapabilities glCapabilities = new GLCapabilities();

        //FSAA (Full Screen Anti-Aliasing)
        //glCapabilities.setSampleBuffers(true);
        //glCapabilities.setNumSamples(4);
        
        //glCapabilities.setHardwareAccelerated(true);

        glCanvas = GLDrawableFactory.getFactory().createGLCanvas(glCapabilities);

        glCanvas.setSize(200, 200);
        glCanvas.addGLEventListener(visRenderer);
        
        // for testing
        //canvas.addGLEventListener(new Gears.GearRenderer());

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
