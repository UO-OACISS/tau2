/*
 * VisRenderer.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.glu.GLU;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import com.sun.opengl.util.BufferUtil;

/**
 * This object manages the JOGL interface.
 *    
 * <P>CVS $Id: VisRenderer.java,v 1.11 2009/08/20 23:11:24 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.11 $
 */
public class VisRenderer implements GLEventListener, MouseListener, MouseMotionListener, MouseWheelListener {

    private class VisAnimator extends Thread {

        private volatile boolean stop = false;

        public void run() {
            stop = false;
            while (!stop) {
                try {
                    if (rotateSpeed == 0) {
                        Thread.sleep(250);
                    } else {
                        VisRenderer.this.rotate(rotateSpeed, 0);
                    }
                } catch (Exception e) {
                    // Who cares if we were interrupted
                }
            }
        }

        public void end() {
            stop = true;
        }
    }

    // OpenGL objects
    private GL gl;
    private GLU glu;
    private GLAutoDrawable glDrawable;

    // conversion to/from degress/radians
    private final static float DTOR = 0.0174532925f;
    private final static float RTOD = 57.2957795f;

    // sensitivity settings
    private final static float lateralSense = 1 * DTOR;
    private final static float verticalSense = 1 * DTOR;

    // camera information
    private double viewAltitude = -30 * DTOR; // The angle from the x-y plane that the eye is placed 
    private double viewAzimuth = -135 * DTOR; // The angle on the x-y plane that the eye is placed
    private double viewDistance = 50.0; // The distance from the eye to the aim
    private Vec eye; // The location of the eye
    private Vec aim = new Vec(0, 0, 0); // Where the eye is focused at
    private Vec vup; // The canonical V-up vector
    private float camera_aperture = 45.0f;
    private float camera_near = 1.0f;
    private float camera_far = 500.0f;
    private float camera_focallength = 50.0f;

    private List shapes = new ArrayList(); // The list of shapes to draw

    private float fps; // Frames per Second
    private int framesRendered;

    // dimensionality of the viewing frame (pixels on the screen)
    private int width, height;

    // for screenshot capability
    private boolean makeScreenShot;
    private BufferedImage screenShot;

    // auto-rotation capability
    private VisAnimator visAnimator;
    private volatile float rotateSpeed = 0.5f;

    // GL info
    private String glInfo_Vendor;
    private String glInfo_Renderer;
    private String glInfo_Version;
    private boolean stereo_available;

    // settings
    private boolean stereo;
    private JCheckBox stereoCheckBox;
    private boolean antiAliasedLines;
    private boolean fsaa;
    private boolean reverseVideo;

    private int prevMouseX, prevMouseY;
    private boolean mouseRButtonDown;

    public static final int CAMERA_PLOT = 0;
    public static final int CAMERA_STICK = 1;

    private int cameraMode = CAMERA_PLOT;

    private VisCanvasListener visCanvasListener;

    public VisRenderer() {}

    /**
     * Set the handler for VisCanvas events.  When a canvas change is needed, 
     * for example when FSAA is set/unset, the parent needs to place the new canvas
     * into their JFrame.
     * @param visCanvasListener
     */
    public void setVisCanvasListener(VisCanvasListener visCanvasListener) {
        this.visCanvasListener = visCanvasListener;
    }

    /**
     * Add a shape to the list of shapes to be drawn
     * @param shape		the shape to add
     */
    public void addShape(Shape shape) {
        shapes.add(shape);
    }

    /**
     * Remove a shape from the list of shapes to be drawn
     * @param shape		the shape to remove
     */
    public void removeShape(Shape shape) {
        shapes.remove(shape);
    }

    private void setLighting() {
        float lightPosition[] = { 0.0f, 0.0f, 1.0f, 0.0f };

        //float lightPosition[] = { 0.7f, 1.0f, 0.6f, 0.0f };
        //float lightPosition[] = { 5.7f, 5.0f, 5.6f, 1.0f };
        //float lightPosition[] = { -1.0f, 0.0f, 1.0f, 0.0f };

        //float lightPosition[] = { 0f, 10.0f, 10.0f, 1.0f };

        //float lightPosition2[] = { 0.7f, -1.0f, 0.6f, 0.0f };
        float whiteLight[] = { 0.75f, 0.75f, 0.75f, 1.0f };
        //float ambientLight[] = { 0.15f, 0.15f, 0.15f, 1.0f };
        float ambientLight[] = { 0.15f, 0.15f, 0.15f, 1.0f };

        gl.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, lightPosition, 0);
        gl.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, whiteLight, 0);
        //gl.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, whiteLight, 0);
        gl.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, ambientLight, 0);

        //        float mat_shininess[] = { 50.0f };
        //        float mat_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
        //        gl.glMaterialfv(GL.GL_FRONT, GL.GL_SHININESS, mat_shininess, 0);
        //        gl.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, mat_specular, 0);

        //gl.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, lightPosition2);
        //gl.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, whiteLight);
        //gl.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, whiteLight);
        //gl.glLightfv(GL.GL_LIGHT1, GL.GL_AMBIENT, ambientLight);

        gl.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT, ambientLight, 0);
        gl.glEnable(GL.GL_COLOR_MATERIAL);
        gl.glShadeModel(GL.GL_FLAT);
        gl.glEnable(GL.GL_LIGHTING);
        gl.glEnable(GL.GL_LIGHT0);

        //        float fogColor[] = {0.5f, 0.5f, 0.5f, 1.0f}; //set the for color to grey
        //        float density = 0.015f;
        //        gl.glEnable(GL.GL_FOG);
        //        gl.glFogi (GL.GL_FOG_MODE, GL.GL_EXP2); //set the fog mode to GL_EXP2
        //        gl.glFogfv (GL.GL_FOG_COLOR, fogColor, 0); //set the fog color to our color chosen above
        //        gl.glFogf (GL.GL_FOG_DENSITY, density); //set the density to the value above
        //        gl.glHint (GL.GL_FOG_HINT, GL.GL_DONT_CARE); // set the fog to look the nicest, may slow down on older cards

        //gl.glEnable(GL.GL_LIGHT1);
        //gl.glEnable(GL.GL_BLEND);

    }

    /**
     * This method is called by JOGL, do not use.
     *
     * @see net.java.games.jogl.GLEventListener#init(net.java.games.jogl.GLAutoDrawable)
     * 
     * @param drawable	The drawable provided by JOGL
     */
    public void init(GLAutoDrawable drawable) {
        gl = drawable.getGL();
        glu = new GLU();
        this.glDrawable = drawable;

        VisTools.verr(this, "Initializing OpenGL (JOGL)");
        VisTools.verr(this, "JOGL Class: " + gl.getClass().getName());
        VisTools.verr(this, "GL_VENDOR: " + gl.glGetString(GL.GL_VENDOR));
        VisTools.verr(this, "GL_RENDERER: " + gl.glGetString(GL.GL_RENDERER));
        VisTools.verr(this, "GL_VERSION: " + gl.glGetString(GL.GL_VERSION));
        glInfo_Vendor = gl.glGetString(GL.GL_VENDOR);
        glInfo_Renderer = gl.glGetString(GL.GL_RENDERER);
        glInfo_Version = gl.glGetString(GL.GL_VERSION);

        byte[] bytes = new byte[1];
        gl.glGetBooleanv(GL.GL_STEREO, bytes, 0);
        if (bytes[0] != 0) {
            stereo_available = true;
            VisTools.verr(this, "OpenGL Stereo is available");
        } else {
            stereo_available = false;
            VisTools.verr(this, "OpenGL Stereo is not available");
        }

        if (stereoCheckBox != null) {
            stereoCheckBox.setEnabled(stereo_available);
        }

        gl.glEnable(GL.GL_CULL_FACE);
        gl.glEnable(GL.GL_DEPTH_TEST);

        setLighting();

        computeEye();

        if (System.getProperty("vis.polyline") != null) {
            gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE);
        }

        if (System.getProperty("vis.polyfill") != null) {
            gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL);
        }

        if (System.getProperty("vis.polypoint") != null) {
            gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_POINT);
        }

        drawable.addMouseListener(this);
        drawable.addMouseMotionListener(this);
        drawable.addMouseWheelListener(this);
    }

    // computes the eye's position based on the viewAngle, viewDirection and aim vector
    private void computeEye() {

        if (aim == null)
            return;

        //System.out.println ("viewAltitude now " + viewAltitude);
        //System.out.println ("viewAzimuth now " + viewAzimuth);

        Matrix rotateY = Matrix.createRotateY(-viewAltitude);
        Matrix rotateZ = Matrix.createRotateZ(viewAzimuth);

        eye = rotateZ.transform(rotateY.transform(new Vec(1, 0, 0)));
        eye.normalize();

        // set the canonical v-up vector
        vup = rotateZ.transform(rotateY.transform(new Vec(0, 0, 1)));

        eye.setx(eye.x() * viewDistance);
        eye.sety(eye.y() * viewDistance);
        eye.setz(eye.z() * viewDistance);

        if (cameraMode == CAMERA_STICK) {
            aim.setx(0);
            aim.sety(0);
        }

        eye = eye.add(aim);
    }

    /**
     * Cleans up resources used by this object.
     */
    public void cleanUp() {
        if (visAnimator != null) {
            visAnimator.end();
        }
    }

    /**
     * Rotate the camera
     */
    private void rotate(float x, float y) {
        final float limit = 90.0f;

        viewAltitude += verticalSense * y;
        if (viewAltitude < -DTOR * limit)
            viewAltitude = -DTOR * limit;
        if (viewAltitude > DTOR * limit)
            viewAltitude = DTOR * limit;

        viewAzimuth += lateralSense * x;
        if (viewAzimuth >= 2 * Math.PI)
            viewAzimuth -= 2 * Math.PI;
        if (viewAzimuth >= 2 * Math.PI)
            viewAzimuth -= 2 * Math.PI;

        computeEye();
        redraw();
    }

    /**
     * Translate the camera
     */
    private void translate(float x, float y) {

        double oldViewAltitude = viewAltitude;
        float oldz = aim.z();

        viewAltitude = 45.0;

        computeEye();

        Vec VPN = aim.subtract(eye);
        Vec n = new Vec(VPN);
        n.normalize();
        Vec u = vup.cross(n);
        u.normalize();
        Vec v = n.cross(u);
        v.normalize();

        Matrix translate = new Matrix();
        translate.setToTranslate(-eye.x(), -eye.y(), -eye.z());

        Matrix rotate = new Matrix();
        rotate.setOrthRotate(u, v, n);

        Matrix M = rotate.multiply(translate);

        Vec diff = eye.subtract(aim);

        double scaleFactor = Math.sqrt(diff.length()) / 50;

        translate.setToTranslate(scaleFactor * x, 0, scaleFactor * y);
        M = translate.multiply(M);

        rotate.transpose();
        M = rotate.multiply(M);

        translate.setToTranslate(eye.x(), eye.y(), eye.z());

        M = translate.multiply(M);

        aim = M.transform(aim);

        aim.setz(oldz);
        viewAltitude = oldViewAltitude;

        computeEye();

        redraw();
    }

    private void translateStick(float x, float y) {
        aim.setz(aim.z() - (y / 3));
        computeEye();
        redraw();
    }

    /**
     * Redraws the shapes
     */
    public void redraw() {
        if (glDrawable != null) {
            glDrawable.display();
        }
    }

    /**
     * Check if we are ready to draw
     */
    public boolean isReadyToDraw() {
        if (glDrawable != null) {
            return true;
        }
        return false;
    }

    /**
     * Creates a screenshot of the current display
     * @return	the screenshot
     */
    public BufferedImage createScreenShot() {
        // screenshot must be taken within display loop
        // set a boolean tag and rerun display loop
        makeScreenShot = true;
        glDrawable.display();
        return screenShot;
    }

    /**
     * This method is called by JOGL, do not use.
     * @see net.java.games.jogl.GLEventListener#reshape(net.java.games.jogl.GLAutoDrawable, int, int, int, int)
     */
    public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {
        this.width = width;
        this.height = height;

        //camera_aperture = 45.0f;
        camera_near = 1.0f;
        camera_far = 500.0f;

        gl.glMatrixMode(GL.GL_PROJECTION);
        gl.glLoadIdentity();
        glu.gluPerspective(camera_aperture, (float) width / (float) height, camera_near, camera_far);
        gl.glMatrixMode(GL.GL_MODELVIEW);
        gl.glLoadIdentity();
    }

    /**
     * This method is called by JOGL, do not use.
     * To force a redraw, use VisRenderer.redraw()
     *
     * @see net.java.games.jogl.GLEventListener#display(net.java.games.jogl.GLAutoDrawable)
     */
    public void display(GLAutoDrawable drawable) {

        reshape(drawable, 0, 0, this.getWidth(), this.getHeight());

        int n = 1;
        if (stereo) {
            n = 2;
        }

        for (int frame = 0; frame < n; frame++) {

            if (stereo) {
                if (frame == 0) {
                    gl.glDrawBuffer(GL.GL_BACK_LEFT);
                } else {
                    gl.glDrawBuffer(GL.GL_BACK_RIGHT);
                }
            } else {
                gl.glDrawBuffer(GL.GL_BACK);
            }
            //        gl = new DebugGL(drawable.getGL());
            //        gl.glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

            if (reverseVideo) {
                //gl.glClearColor(0.75f, 0.75f, 0.75f, 1.0f);
                gl.glClearColor(238 / 255.0f, 238 / 255.0f, 238 / 255.0f, 1.0f);
            } else {
                gl.glClearColor(0, 0, 0, 0);
            }
            gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);

            gl.glPushMatrix();
            gl.glTranslated(0, 0, -viewDistance);

            setLighting();

            if (aim == null) {
                aim = new Vec(0, 0, 0);
            }

            if (stereo) {

                float eyesep, wd2, ratio, radians;

                //camera_focallength = 50.0f;
                //camera_aperture = 45.0f;
                camera_near = 1.0f;
                camera_far = 500.0f;

                eyesep = camera_focallength / 20;

                ratio = (float) this.width / (float) this.height;
                radians = DTOR * camera_aperture / 2;
                wd2 = (float) (camera_near * Math.tan(radians));

                // direction vector
                Vec direction = eye.subtract(aim);

                // cross product of v-up and the view direction will give us the normal, which is the axis the eyes are on
                // we will add and subtract this vector to move to the left and right eye
                Vec crossRight = direction.cross(vup);
                crossRight.normalize();
                crossRight.scale((float) (eyesep / 2.0));

                float leftRightFrustumShift;

                if (frame == 0) {
                    leftRightFrustumShift = 0.5f;
                } else {
                    leftRightFrustumShift = -0.5f;
                }

                // create a parallel axis asymmetric frustum perspective projection
                float left, right, top, bottom;
                gl.glMatrixMode(GL.GL_PROJECTION);
                gl.glLoadIdentity();
                float ndfl = camera_near / camera_focallength;
                left = (float) (-ratio * wd2 + leftRightFrustumShift * eyesep * ndfl);
                right = (float) (ratio * wd2 + leftRightFrustumShift * eyesep * ndfl);
                top = wd2;
                bottom = -wd2;
                gl.glFrustum(left, right, bottom, top, camera_near, camera_far);
                gl.glMatrixMode(GL.GL_MODELVIEW);

                if (frame == 0) {
                    glu.gluLookAt(aim.x() - crossRight.x(), aim.y() - crossRight.y(), aim.z() - crossRight.z(), eye.x()
                            - crossRight.x(), eye.y() - crossRight.y(), eye.z() - crossRight.z(), vup.x(), vup.y(), vup.z());
                } else {
                    glu.gluLookAt(aim.x() + crossRight.x(), aim.y() + crossRight.y(), aim.z() + crossRight.z(), eye.x()
                            + crossRight.x(), eye.y() + crossRight.y(), eye.z() + crossRight.z(), vup.x(), vup.y(), vup.z());
                }
            } else {
                glu.gluLookAt(aim.x(), aim.y(), aim.z(), eye.x(), eye.y(), eye.z(), vup.x(), vup.y(), vup.z());
            }

            for (int i = 0; i < shapes.size(); i++) {
                Shape shape = (Shape) shapes.get(i);
                shape.render(this);
            }

            //        int err = gl.glGetError();
            //        if (err != GL.GL_NO_ERROR)
            //            System.out.println("err = " + glu.gluErrorString(err));

            gl.glPopMatrix();
            framesRendered++;

        }

        // if screenshot was requested since last draw
        if (makeScreenShot) {
            makeScreenShot = false;
            makeScreenShot(drawable);
        }

    }

    private void makeScreenShot(GLAutoDrawable drawable) {

        int width = drawable.getWidth();
        int height = drawable.getHeight();

        ByteBuffer pixelsRGB = BufferUtil.newByteBuffer(width * height * 3);

        GL gl = drawable.getGL();

        gl.glReadBuffer(GL.GL_BACK);
        gl.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1);

        gl.glReadPixels(0, // GLint x
                0, // GLint y
                width, // GLsizei width
                height, // GLsizei height
                GL.GL_RGB, // GLenum format
                GL.GL_UNSIGNED_BYTE, // GLenum type
                pixelsRGB); // GLvoid *pixels

        int[] pixelInts = new int[width * height];

        // Convert RGB bytes to ARGB ints with no transparency. Flip image vertically by reading the
        // rows of pixels in the byte buffer in reverse - (0,0) is at bottom left in OpenGL.

        int p = width * height * 3; // Points to first byte (red) in each row.
        int q; // Index into ByteBuffer
        int i = 0; // Index into target int[]
        int w3 = width * 3; // Number of bytes in each row

        for (int row = 0; row < height; row++) {
            p -= w3;
            q = p;
            for (int col = 0; col < width; col++) {
                int iR = pixelsRGB.get(q++);
                int iG = pixelsRGB.get(q++);
                int iB = pixelsRGB.get(q++);
                pixelInts[i++] = 0xFF000000 | ((iR & 0x000000FF) << 16) | ((iG & 0x000000FF) << 8) | (iB & 0x000000FF);
            }
        }

        screenShot = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        screenShot.setRGB(0, 0, width, height, pixelInts, 0, width);
    }

    public void displayChanged(GLAutoDrawable drawable, boolean modeChanged, boolean deviceChanged) {}

    // MouseListener implementation
    public void mouseEntered(MouseEvent e) {}

    public void mouseExited(MouseEvent e) {}

    public void mouseClicked(MouseEvent e) {}

    public void mouseMoved(MouseEvent e) {}

    public void mousePressed(MouseEvent e) {
        prevMouseX = e.getX();
        prevMouseY = e.getY();
        if ((e.getModifiers() & InputEvent.BUTTON3_MASK) != 0) {
            mouseRButtonDown = true;
        }
    }

    public void mouseReleased(MouseEvent e) {
        if ((e.getModifiers() & InputEvent.BUTTON3_MASK) != 0) {
            mouseRButtonDown = false;
        }
    }

    public void mouseDragged(MouseEvent e) {
        int x = e.getX();
        int y = e.getY();
        Dimension size = e.getComponent().getSize();

        float thetaY = 360.0f * ((float) (x - prevMouseX) / (float) size.width);
        float thetaX = 360.0f * ((float) (prevMouseY - y) / (float) size.height);

        float dy, dx;
        dx = x - prevMouseX;
        dy = y - prevMouseY;

        if (mouseRButtonDown) {
            if (cameraMode == CAMERA_PLOT) {
                translate(-dx, -dy);
            } else {
                translateStick(-dx, -dy);
            }
        } else {
            rotate(-dx, dy);
        }

        prevMouseX = x;
        prevMouseY = y;
    }

    // Zoom in and out with the mouse wheel
    public void mouseWheelMoved(MouseWheelEvent e) {
        int scrollAmount = e.getWheelRotation();
        if (scrollAmount > 0) {
            zoomOut();
        } else {
            zoomIn();
        }
    }

    /**
     * Zooms the camera in by dividingthe distance between the <tt>aim</tt> and <tt>eye</tt> by <tt>1.1</tt>.
     */
    public void zoomIn() {
        viewDistance /= 1.1;
        computeEye();
        glDrawable.display();
    }

    /**
     * Zooms the camera out by multiplying the distance between the <tt>aim</tt> and <tt>eye</tt> by <tt>1.1</tt>.
     */
    public void zoomOut() {
        viewDistance *= 1.1;
        computeEye();
        glDrawable.display();
    }

    /**
     * Returns the fps value.
     * @return the fps value.
     */
    public float getFps() {
        return fps;
    }

    /**
     * Sets the fps value, stored by this <tt>VisRenderer</tt>, but not currently used by it.
     * @param fps The fps to set.
     */
    public void setFps(float fps) {
        this.fps = fps;
    }

    /**
     * Returns the framesRendered.
     * @return the framesRendered.
     */
    public int getFramesRendered() {
        return framesRendered;
    }

    /**
     * Sets the number of frames rendered (used to reset it).
     * @param framesRendered the number of frames rendered (used to reset it).
     */
    public void setFramesRendered(int framesRendered) {
        this.framesRendered = framesRendered;
    }

    /**
     * Returns the height.
     * @return the height.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Returns the width.
     * @return the width.
     */
    public int getWidth() {
        return width;
    }

    /**
     * Returns the point (as a <tt>Vec</tt>) that the camera is located.
     * @return the point (as a <tt>Vec</tt>) that the camera is located.
     */
    public Vec getEye() {
        return eye;
    }

    /**
     * Returns the point (as a <tt>Vec</tt>) that the camera is currently aimed.
     * @return the point (as a <tt>Vec</tt>) that the camera is currently aimed.
     */
    public Vec getAim() {
        return aim;
    }

    /**
     * Set the point (as a <tt>Vec</tt>) that the camera is currently aimed at.
     * @param aim the <tt>Vec</tt> to aim at.
     */
    public void setAim(Vec aim) {
        this.aim = aim;
        computeEye();
    }

    /**
     * Creates a Swing JPanel with controls for this object.
     * 
     * When getControlPanel() is called, the controls will represent the current
     * values for the object, but currently, they will not stay in sync if the values
     * are changed using the public methods.  For example, if you call "setEnabled(false)"
     * The JCheckBox will not be set to unchecked.  This functionality could be added if
     * requested.
     * 
     * @return the control panel for this component
     */
    public JPanel getControlPanel() {
        JPanel panel = new JPanel();
        panel.setBorder(BorderFactory.createLoweredBevelBorder());
        panel.setLayout(new GridBagLayout());

        final JCheckBox rotateCheckBox = new JCheckBox("Auto-Rotate", visAnimator != null);
        rotateCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    if (rotateCheckBox.isSelected()) {
                        visAnimator = new VisAnimator();
                        visAnimator.start();
                    } else {
                        visAnimator.end();
                        visAnimator = null;
                    }

                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        final JCheckBox reverseCheckBox = new JCheckBox("Reverse Video", reverseVideo);
        reverseCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    setReverseVideo(reverseCheckBox.isSelected());
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        final JCheckBox antialiasCheckBox = new JCheckBox("AA Lines", antiAliasedLines);
        antialiasCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    setAntiAliasedLines(antialiasCheckBox.isSelected());
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        final JCheckBox fsaaCheckBox = new JCheckBox("Full Screen AA", fsaa);
        fsaaCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    setFSAA(fsaaCheckBox.isSelected());
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        stereoCheckBox = new JCheckBox("Stereo", stereo);
        stereoCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    setStereo(stereoCheckBox.isSelected());
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });
        stereoCheckBox.setEnabled(stereo_available);

        final JSlider focalLengthSlider = new JSlider(0, 200, (int) camera_focallength);
        focalLengthSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                setCamera_focallength(focalLengthSlider.getValue());
            }
        });

        final JSlider apertureSlider = new JSlider(0, 90, (int) camera_aperture);
        apertureSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                setCamera_aperture(apertureSlider.getValue());
            }
        });

        final JSlider speedSlider = new JSlider(0, 200, (int) (Math.sqrt(rotateSpeed) * 100));
        speedSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    rotateSpeed = speedSlider.getValue() / 100.0f;
                    rotateSpeed *= rotateSpeed;
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        final JButton glInfoButton = new JButton("GL Info");
        glInfoButton.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent e) {
                String message = "JOGL Class: " + gl.getClass().getName() + "\n" + "GL_VENDOR: " + glInfo_Vendor + "\n"
                        + "GL_RENDERER: " + glInfo_Renderer + "\n" + "GL_VERSION: " + glInfo_Version;

                JOptionPane.showMessageDialog(glInfoButton, message);
            }

        });

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.weighty = 0.2;
        gbc.weightx = 0.5;

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.WEST;
        VisTools.addCompItem(panel, rotateCheckBox, gbc, 0, 0, 2, 2);

        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0.5;
        gbc.fill = GridBagConstraints.NONE;
        VisTools.addCompItem(panel, new JLabel("Speed"), gbc, 2, 0, 1, 1);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        VisTools.addCompItem(panel, speedSlider, gbc, 2, 1, 1, 1);

        VisTools.addCompItem(panel, reverseCheckBox, gbc, 0, 2, 2, 1);
        VisTools.addCompItem(panel, stereoCheckBox, gbc, 1, 2, 2, 1);
        VisTools.addCompItem(panel, antialiasCheckBox, gbc, 0, 3, 2, 1);
        VisTools.addCompItem(panel, fsaaCheckBox, gbc, 1, 3, 3, 1);
        gbc.weightx = 0.1;
        VisTools.addCompItem(panel, new JLabel("Separation"), gbc, 0, 4, 1, 1);
        VisTools.addCompItem(panel, new JLabel("Aperture"), gbc, 0, 5, 1, 1);
        gbc.fill = GridBagConstraints.BOTH;
        gbc.weightx = 0.9;

        gbc.anchor = GridBagConstraints.CENTER;
        VisTools.addCompItem(panel, focalLengthSlider, gbc, 1, 4, 2, 1);
        VisTools.addCompItem(panel, apertureSlider, gbc, 1, 5, 2, 1);

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.CENTER;
        VisTools.addCompItem(panel, glInfoButton, gbc, 0, 6, 3, 1);

        return panel;
    }

    /**
     * Returns whether or not reverse video is on (black on white).
     * @return whether or not reverse video is on (black on white).
     */
    public boolean getReverseVideo() {
        return reverseVideo;
    }

    /**
     * Makes video reversed or not (black on white or white on black).
     * @param reverseVideo <tt>true</tt> for black background; <tt>false</tt> for white.
     */
    public void setReverseVideo(boolean reverseVideo) {
        this.reverseVideo = reverseVideo;
        this.redraw();
    }

    /**
     * Returns the <tt>GLAutoDrawable</tt> for this <tt>VisRenderer</tt>.
     * @return the <tt>GLAutoDrawable</tt> for this <tt>VisRenderer</tt>.
     */
    public GLAutoDrawable getGLAutoDrawable() {
        return glDrawable;
    }

    /**
     * Returns the direction of the camera.  Used by the plots to determine how 
     * to draw the data (for proper translucency, you must draw back to front).
     * @return the direction of the camera.
     */
    public Vec getViewDirection() {
        return eye.subtract(aim);
    }

    public boolean getAntiAliasedLines() {
        return antiAliasedLines;
    }

    public void setAntiAliasedLines(boolean antiAliasedLines) {
        this.antiAliasedLines = antiAliasedLines;
        this.redraw();
    }

    public boolean getStereo() {
        return stereo;
    }

    public void setStereo(boolean stereo) {
        this.stereo = stereo;
        this.redraw();
    }

    public float getCamera_focallength() {
        return camera_focallength;
    }

    public void setCamera_focallength(float camera_focallength) {
        this.camera_focallength = Math.max(1.0f, camera_focallength);
        this.redraw();
    }

    public float getCamera_aperture() {
        return camera_aperture;
    }

    public void setCamera_aperture(float camera_aperture) {
        this.camera_aperture = camera_aperture;
        this.redraw();
    }

    public int getCameraMode() {
        return cameraMode;
    }

    public void setCameraMode(int cameraMode) {
        this.cameraMode = cameraMode;
        computeEye();
    }

    public boolean getFSAA() {
        return fsaa;
    }

    public void setFSAA(boolean fsaa) {
        if (fsaa != this.fsaa) {
            this.fsaa = fsaa;
            
            
            if (visCanvasListener != null) {
                visCanvasListener.createNewCanvas();
            }
            
            for (int i = 0; i < shapes.size(); i++) {
                Shape shape = (Shape) shapes.get(i);
                shape.resetCanvas();
            }
            redraw();
        }
    }

}