package edu.uoregon.tau.paraprof.vis;

import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.nio.ByteBuffer;
import java.util.Vector;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.java.games.jogl.*;
import net.java.games.jogl.util.BufferUtils;
import edu.uoregon.tau.paraprof.ParaProfUtils;

/**
 * This object manages the JOGL interface
 *    
 * TODO: Add FPS indication
 * 
 * <P>CVS $Id: VisRenderer.java,v 1.3 2005/04/15 01:29:03 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.3 $
 */
public class VisRenderer implements GLEventListener, MouseListener, MouseMotionListener, MouseWheelListener {

    private class VisAnimator extends Thread {

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

        private volatile boolean stop = false;

        public void end() {
            stop = true;
        }

    }

    private int prevMouseX, prevMouseY;
    private boolean mouseRButtonDown = false;

    private GL gl;
    private GLU glu;
    private GLDrawable glDrawable;

    private Vec eye; // The location of the eye
    private Vec aim = new Vec(0, 0, 0); // Where the eye is focused at
    private Vec vup; // The canonical V-up vector

    private Vec viewDirection;
    
    final static private float rad = (float) (3.14 / 180);
    final static private float lateralSense = 1 * rad;
    final static private float verticalSense = 1 * rad;

    private double viewAltitude = -30 * rad; // The angle from the x-y plane that the eye is placed 
    private double viewAzimuth = -135 * rad; // The angle on the x-y plane that the eye is placed
    private double viewDistance = 50.0; // The distance from the eye to the aim
    private float fovy = 45.0f; // field of view (y direction)

    private boolean reverseVideo = false;
    
    private Color backColor = Color.WHITE;
    private Color foreColor = Color.BLACK;
    
    // the list of shapes to draw
    private Vector shapes = new Vector();

    private float fps;
    private int framesRendered;

    private int width, height;

    // screenshot capability
    private boolean makeScreenShot;
    private BufferedImage screenShot;

    // auto-rotation capability
    private VisAnimator visAnimator;
    private volatile float rotateSpeed = 0.5f;

    public VisRenderer() {
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

        //float lightPosition[] = { 0.7f, 1.0f, 0.6f, 0.0f };
        float lightPosition[] = { 0.0f, 0.0f, 1.0f, 0.0f };
        //float lightPosition[] = { 5.7f, 5.0f, 5.6f, 1.0f };

        //        float lightPosition[] = { 0f, 10.0f, 10.0f, 1.0f };

        //      float lightPosition2[] = { 0.7f, -1.0f, 0.6f, 0.0f };
        float whiteLight[] = { 0.75f, 0.75f, 0.75f, 1.0f };
        //float ambientLight[] = { 0.15f, 0.15f, 0.15f, 1.0f };
        float ambientLight[] = { 0.15f, 0.15f, 0.15f, 1.0f };

        //      float mat_shininess[] = { 50.0f };
        //    float mat_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };

        gl.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, lightPosition);
        gl.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, whiteLight);
        //        gl.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, whiteLight);
        gl.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, ambientLight);

        //        gl.glMaterialfv(GL.GL_FRONT, GL.GL_SHININESS, mat_shininess);
        //        gl.glMaterialfv(GL.GL_FRONT, GL.GL_SPECULAR, mat_specular);

        //    gl.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, lightPosition2);
        //    gl.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, whiteLight);
        //gl.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, whiteLight);
        //   gl.glLightfv(GL.GL_LIGHT1, GL.GL_AMBIENT, ambientLight);

        gl.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT, ambientLight);
        gl.glEnable(GL.GL_COLOR_MATERIAL);
        gl.glShadeModel(GL.GL_FLAT);
        gl.glEnable(GL.GL_LIGHTING);
        gl.glEnable(GL.GL_LIGHT0);
        //        gl.glEnable(GL.GL_LIGHT1);
        //gl.glEnable(GL.GL_BLEND);

    }

    /**
     * This method is called by JOGL, do not use.
     *
     * @see net.java.games.jogl.GLEventListener#init(net.java.games.jogl.GLDrawable)
     * 
     * @param drawable	The drawable provided by JOGL
     */
    public void init(GLDrawable drawable) {
        gl = drawable.getGL();
        glu = drawable.getGLU();
        this.glDrawable = drawable;

        ParaProfUtils.verr(this, "Initializing OpenGL (JOGL)");
        ParaProfUtils.verr(this, "INIT GL IS: " + gl.getClass().getName());
        ParaProfUtils.verr(this, "GL_VENDOR: " + gl.glGetString(GL.GL_VENDOR));
        ParaProfUtils.verr(this, "GL_RENDERER: " + gl.glGetString(GL.GL_RENDERER));
        ParaProfUtils.verr(this, "GL_VERSION: " + gl.glGetString(GL.GL_VERSION));

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

        //System.out.println ("viewAngle now " + viewAngle);
        //System.out.println ("viewDirection now " + viewDirection);

        eye = new Vec((float) Math.cos(viewAltitude), 0.0f, (float) Math.sin(viewAltitude));

        eye.sety(eye.x() * Math.sin(viewAzimuth));
        eye.setx(eye.x() * Math.cos(viewAzimuth));

        eye.normalize();

        vup = new Vec(0.0, 0.0, 1.0);
        vup = vup.subtract(eye);
        eye.setx(eye.x() * viewDistance);
        eye.sety(eye.y() * viewDistance);
        eye.setz(eye.z() * viewDistance);

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

    private void rotate(float x, float y) {
        final float limit = 90.0f;

        viewAltitude += verticalSense * y;
        if (viewAltitude < -rad * limit)
            viewAltitude = -rad * limit;
        if (viewAltitude > rad * limit)
            viewAltitude = rad * limit;

        viewAzimuth += lateralSense * x;
        if (viewAzimuth >= 2 * 3.14f)
            viewAzimuth -= 2 * 3.14f;
        if (viewAzimuth >= 2 * 3.14f)
            viewAzimuth -= 2 * 3.14f;

        computeEye();
        redraw();
    }

    private void translate(float x, float y) {

        double oldViewAngle = viewAltitude;
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

        Vec tmp2 = M.transform(aim);

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
        viewAltitude = oldViewAngle;

        computeEye();

        redraw();
    }

    /**
     * Redraws the shapes
     */
    public void redraw() {
        glDrawable.display();
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
     * @see net.java.games.jogl.GLEventListener#reshape(net.java.games.jogl.GLDrawable, int, int, int, int)
     */
    public void reshape(GLDrawable drawable, int x, int y, int width, int height) {
        this.width = width;
        this.height = height;
        gl.glMatrixMode(GL.GL_PROJECTION);
        gl.glLoadIdentity();
        glu.gluPerspective(45, (float) width / (float) height, 1.0f, 500.0f);
        gl.glMatrixMode(GL.GL_MODELVIEW);
        gl.glLoadIdentity();
    }

    /**
     * This method is called by JOGL, do not use.
     * To force a redraw, use VisRenderer.redraw()
     *
     * @see net.java.games.jogl.GLEventListener#display(net.java.games.jogl.GLDrawable)
     */
    public void display(GLDrawable drawable) {

        reshape(drawable, 0, 0, this.getWidth(), this.getHeight());

        //        gl = new DebugGL(drawable.getGL());

//        gl.glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        if (reverseVideo) {
            //gl.glClearColor(0.75f, 0.75f, 0.75f, 1.0f);
            gl.glClearColor(238/255.0f, 238/255.0f, 238/255.0f, 1.0f);
        } else {
            gl.glClearColor(0,0,0,0);
        }
        gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);

        gl.glPushMatrix();
        gl.glTranslated(0, 0, -viewDistance);

        setLighting();

        if (aim == null) {
            aim = new Vec(0, 0, 0);
        }
        glu.gluLookAt(aim.x(), aim.y(), aim.z(), eye.x(), eye.y(), eye.z(), vup.x(), vup.y(), vup.z());

        viewDirection = eye.subtract(aim);
        
        for (int i = 0; i < shapes.size(); i++) {
            Shape shape = (Shape) shapes.get(i);
            shape.render(this);
        }

        //        int err = gl.glGetError();
        //        if (err != GL.GL_NO_ERROR)
        //            System.out.println("err = " + glu.gluErrorString(err));

        gl.glPopMatrix();
        framesRendered++;

        // if screenshot was requested since last draw
        if (makeScreenShot) {
            makeScreenShot = false;
            makeScreenShot(drawable);
        }

    }

    private void makeScreenShot(GLDrawable drawable) {
        int width = drawable.getSize().width;
        int height = drawable.getSize().height;

        ByteBuffer pixelsRGB = BufferUtils.newByteBuffer(width * height * 3);

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

                pixelInts[i++] = 0xFF000000 | ((iR & 0x000000FF) << 16) | ((iG & 0x000000FF) << 8)
                        | (iB & 0x000000FF);
            }

        }

        screenShot = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        screenShot.setRGB(0, 0, width, height, pixelInts, 0, width);
    }

    public void displayChanged(GLDrawable drawable, boolean modeChanged, boolean deviceChanged) {
    }

    // Methods required for the implementation of MouseListener
    public void mouseEntered(MouseEvent e) {
    }

    public void mouseExited(MouseEvent e) {
    }

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

    public void mouseClicked(MouseEvent e) {
    }

    // Methods required for the implementation of MouseMotionListener
    public void mouseDragged(MouseEvent e) {
        int x = e.getX();
        int y = e.getY();
        Dimension size = e.getComponent().getSize();

        float thetaY = 360.0f * ((float) (x - prevMouseX) / (float) size.width);
        float thetaX = 360.0f * ((float) (prevMouseY - y) / (float) size.height);

        float dy, dx;
        dx = x - prevMouseX;
        dy = y - prevMouseY;

        // System.out.println ("nx = " + nx + ", ny = " + ny);

        if (mouseRButtonDown) {
            translate(-dx, -dy);
        } else {
            rotate(-dx, dy);
        }

        prevMouseX = x;
        prevMouseY = y;
    }

    public void mouseMoved(MouseEvent e) {
    }

    public void mouseWheelMoved(MouseWheelEvent e) {
        int scrollAmount = e.getWheelRotation();

        //viewDistance += scrollAmount * 2;
        if (scrollAmount > 0) {
            zoomOut();
        } else {
            zoomIn();
        }
    }

    public void zoomIn() {
        viewDistance /= 1.1;
        computeEye();
        glDrawable.display();
    }

    public void zoomOut() {
        viewDistance *= 1.1;
        computeEye();
        glDrawable.display();
    }

    /**
     * @return Returns the fps.
     */
    public float getFps() {
        return fps;
    }

    /**
     * @param fps The fps to set.
     */
    public void setFps(float fps) {
        this.fps = fps;
    }

    /**
     * @return Returns the framesRendered.
     */
    public int getFramesRendered() {
        return framesRendered;
    }

    /**
     * @param framesRendered The framesRendered to set.
     */
    public void setFramesRendered(int framesRendered) {
        this.framesRendered = framesRendered;
    }

    public int getHeight() {
        return height;
    }

    public int getWidth() {
        return width;
    }

    public Vec getEye() {
        return eye;
    }

    public void setEye(Vec eye) {
        //this.eye = eye;
    }

    public Vec getAim() {
        return aim;
    }

    public void setAim(Vec aim) {
        this.aim = aim;
        computeEye();
    }

    public JPanel getControlPanel() {
        JPanel panel = new JPanel();
        panel.setBorder(BorderFactory.createLoweredBevelBorder());
        panel.setLayout(new GridBagLayout());

        final JCheckBox rotateCheckBox = new JCheckBox("Rotate", visAnimator != null);
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
                    ParaProfUtils.handleException(e);
                }
            }
        });

        final JCheckBox reverseCheckBox = new JCheckBox("Reverse Video", reverseVideo);
        reverseCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                        setReverseVideo(reverseCheckBox.isSelected());
                }
                 catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }
        });
        
        final JSlider speedSlider = new JSlider(0, 200, (int) (Math.sqrt(rotateSpeed) * 100));

        speedSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    rotateSpeed = speedSlider.getValue() / 100.0f;
                    rotateSpeed *= rotateSpeed;
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }
        });

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.weighty = 0.2;
        gbc.weightx = 0.1;

        gbc.fill = GridBagConstraints.NONE;
        gbc.anchor = GridBagConstraints.WEST;
        ParaProfUtils.addCompItem(panel, rotateCheckBox, gbc, 0, 0, 1, 2);

        gbc.anchor = GridBagConstraints.CENTER;
        gbc.weightx = 0.9;
        gbc.fill = GridBagConstraints.NONE;
        ParaProfUtils.addCompItem(panel, new JLabel("Speed"), gbc, 1, 0, 1, 1);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        ParaProfUtils.addCompItem(panel, speedSlider, gbc, 1, 1, 1, 1);

        ParaProfUtils.addCompItem(panel, reverseCheckBox, gbc, 0, 2, 2, 1);

        return panel;
    }

    
  
    
    public boolean getReverseVideo() {
        return reverseVideo;
    }
    
    public void setReverseVideo(boolean reverseVideo) {
        this.reverseVideo = reverseVideo;
        this.redraw();
    }
    
    
    public GLDrawable getGLDrawable() {
        return glDrawable;
    }
    
    public Vec getViewDirection() {
        return viewDirection;
    }
    
}