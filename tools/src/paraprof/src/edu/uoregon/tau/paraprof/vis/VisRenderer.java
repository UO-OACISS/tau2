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

public class VisRenderer implements GLEventListener, MouseListener, MouseMotionListener, MouseWheelListener {

    private class Animator extends Thread {

        public void run() {
            stop = false;
            while (!stop) {
                try {
                    if (rotateSpeed == 0) {
                        Thread.sleep(250);

                    } else {
                        //System.out.println("bob");
                        //                    Thread.sleep(1000);
                        VisRenderer.this.rotate(rotateSpeed, 0);
                        //                    VisRenderer.this.redraw();
                    }
                } catch (Exception e) {

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
    private GLDrawable gldrawable;

    private Vec eye; // The location of the eye
    private Vec aim = new Vec(0, 0, 0); // Where the eye is focused at
    private Vec vup; // The canonical V-up

    final static private float rad = (float) (3.14 / 180);
    final static private float lateralSense = 1 * rad;
    final static private float verticalSense = 1 * rad;

    private double viewAngle = -30 * rad; // The angle from the x-y plane that the eye is placed 
    private double viewDirection = -135 * rad; // The angle on the x-y plane that the eye is placed
    private double viewDistance = 50.0; // The distance from the eye to the aim

    private Vector shapes = new Vector();

    private float fps;
    private int framesRendered;

    private int width, height;

    private BufferedImage screenShot;
    
    //    private float perspective = 4.0f;

    private float fovy = 45.0f;

    private boolean makeScreenShot;

    public VisRenderer() {
    }

    public void addShape(Shape shape) {
        shapes.add(shape);
    }

    public void removeShape(Shape shape) {

        shapes.remove(shape);
        System.gc();
    }

    private void setLighting() {

        //float lightPosition[] = { 0.7f, 1.0f, 0.6f, 0.0f };
        float lightPosition[] = { 1.0f, 1.0f, 1.0f, 0.0f };
        //float lightPosition[] = { 5.7f, 5.0f, 5.6f, 1.0f };

        //        float lightPosition[] = { 0f, 10.0f, 10.0f, 1.0f };

        //      float lightPosition2[] = { 0.7f, -1.0f, 0.6f, 0.0f };
        float whiteLight[] = { 0.75f, 0.75f, 0.75f, 1.0f };
        //float ambientLight[] = { 0.15f, 0.15f, 0.15f, 1.0f };
        float ambientLight[] = { 0.25f, 0.25f, 0.25f, 1.0f };

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

    public void init(GLDrawable drawable) {
        gl = drawable.getGL();
        glu = drawable.getGLU();
        this.gldrawable = drawable;
        System.err.println("INIT GL IS: " + gl.getClass().getName());
        System.err.println("GL_VENDOR: " + gl.glGetString(GL.GL_VENDOR));
        System.err.println("GL_RENDERER: " + gl.glGetString(GL.GL_RENDERER));
        System.err.println("GL_VERSION: " + gl.glGetString(GL.GL_VERSION));
        System.err.println();

        gl.glEnable(GL.GL_CULL_FACE);
        gl.glEnable(GL.GL_DEPTH_TEST);

        setLighting();

        // gl.glEnable(GL.GL_NORMALIZE);

        computeEye();

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

        eye = new Vec((float) Math.cos(viewAngle), 0.0f, (float) Math.sin(viewAngle));

        eye.sety(eye.x() * Math.sin(viewDirection));
        eye.setx(eye.x() * Math.cos(viewDirection));

        eye.normalize();

        vup = new Vec(0.0, 0.0, 1.0);
        vup = vup.subtract(eye);
        eye.setx(eye.x() * viewDistance);
        eye.sety(eye.y() * viewDistance);
        eye.setz(eye.z() * viewDistance);

        eye = eye.add(aim);

        //          System.out.println ("eye now " + eye);

    }

    public void cleanUp() {
        if (animator != null) {
            animator.end();
        }
    }

    private void rotate(float x, float y) {
        final float limit = 90.0f;

        viewAngle += verticalSense * y;
        if (viewAngle < -rad * limit)
            viewAngle = -rad * limit;
        if (viewAngle > rad * limit)
            viewAngle = rad * limit;

        viewDirection += lateralSense * x;
        if (viewDirection >= 2 * 3.14f)
            viewDirection -= 2 * 3.14f;
        if (viewDirection >= 2 * 3.14f)
            viewDirection -= 2 * 3.14f;

        computeEye();
        redraw();
    }

    private void translate(float x, float y) {

        double oldViewAngle = viewAngle;
        float oldz = aim.z();

        viewAngle = 45.0;

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
        viewAngle = oldViewAngle;

        computeEye();

        redraw();
    }

    public void redraw() {
        gldrawable.display();
    }

    public BufferedImage createScreenShot() {
        // screenshot must be taken within display loop
        // set a boolean tag and rerun display loop
        makeScreenShot = true;
        gldrawable.display();
        return screenShot;
    }

    public void reshape(GLDrawable drawable, int x, int y, int width, int height) {
        this.width = width;
        this.height = height;
        gl.glMatrixMode(GL.GL_PROJECTION);

        gl.glLoadIdentity();
        //gl.glFrustum(-1.0f, 1.0f, -ratio, ratio, 1.0f, 500.0f);

        //        perspective = 1.0f;
        //        
        //        if (width > height) {
        //            float ratio = (float) width / (float) height;
        //            //gl.glFrustum(-ratio*p, ratio*p, -p, p, 1.0f, 500.0f);
        //            gl.glFrustum(-ratio, ratio, -1.0f, 1.0f, perspective, 500.0f);
        //            //gl.glOrtho(-ratio, ratio, -1.0f, 1.0f, 1.0f, 500.0f);
        //        } else {
        //            float ratio = (float) height / (float) width;
        //            //gl.glFrustum(-p, p, -ratio*p, ratio*p, 1.0f, 500.0f);
        //            gl.glFrustum(-1.0f, 1.0f, -ratio, ratio, perspective, 500.0f);
        //            //gl.glOrtho(-1.0f, 1.0f, -ratio, ratio, 1.0f, 500.0f);
        //        }

        glu.gluPerspective(45, (float) width / (float) height, 1.0f, 500.0f);

        gl.glMatrixMode(GL.GL_MODELVIEW);
        gl.glLoadIdentity();
    }

    //    public void regenerateLists() {
    //
    //        
    //        gldrawable.setRenderingThread(Thread.currentThread());
    //        
    //        for (int i = 0; i < shapes.size(); i++) {
    //            Shape shape = (Shape) shapes.get(i);
    //            Integer displayList = (Integer) displayLists.get(shape);
    //
    //            //if (shape.isDirty() || displayList == null || true) {
    //            if (shape.isDirty() || displayList == null) {
    //
    //                if (displayList == null) {
    //                    displayList = new Integer(gl.glGenLists(1));
    //                    displayLists.put(shape, displayList);
    //                }
    //
    //                //System.out.println ("creating new display list");
    //                gl.glNewList(displayList.intValue(), GL.GL_COMPILE);
    //                shape.render(gl);
    //                shape.clean();
    //                gl.glEndList();
    //            }
    //        }
    //
    //        gldrawable.setRenderingThread(null);
    //
    //    }

    public void display(GLDrawable drawable) {

        reshape(drawable, 0, 0, this.getWidth(), this.getHeight());

        //System.out.println("display called");
        gl = new DebugGL(drawable.getGL());
        //rotate(2,0);
        gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);

        gl.glFrontFace(GL.GL_CCW);
        gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL);
        gl.glDisable(GL.GL_CULL_FACE);

        //float green[] = { 0.0f, 0.8f, 0.2f, 1.0f };
        //gl.glMaterialfv(GL.GL_FRONT, GL.GL_AMBIENT_AND_DIFFUSE, green);

        gl.glPushMatrix();

        gl.glTranslated(0, 0, -viewDistance);

        setLighting();

        glu.gluLookAt(aim.x(), aim.y(), aim.z(), eye.x(), eye.y(), eye.z(), vup.x(), vup.y(), vup.z());

        gl.glEnable(GL.GL_LINE_SMOOTH);
        gl.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST);

        //                gl.glDisable(GL.GL_LIGHTING);
        //                gl.glColor4f(1.0f, 0.0f, 1.0f, 1.0f);
        //        
        //                gl.glBegin(GL.GL_QUADS);
        //                gl.glNormal3f(0,0,1);
        //                gl.glVertex3f(-1, -1, 0);
        //                gl.glVertex3f(1, -1, 0);
        //                gl.glVertex3f(1, 1, 0);
        //                gl.glVertex3f(-1, 1, 0);
        //                
        //                gl.glNormal3f(1,0,0);
        //                gl.glVertex3f(0, -1, -1);
        //                gl.glVertex3f(0, 1, -1);
        //                gl.glVertex3f(0, 1, 1);
        //                gl.glVertex3f(0, -1, 1);
        //                gl.glEnd();

        //gl.glRotatef(view_rotx, 1.0f, 0.0f, 0.0f);
        //gl.glRotatef(view_roty, 0.0f, 1.0f, 0.0f);
        //gl.glRotatef(view_rotz, 0.0f, 0.0f, 1.0f);

        //        for (int i = 0; i < shapes.size(); i++) {
        //            Shape shape = (Shape) shapes.get(i);
        //            Integer displayList = (Integer) displayLists.get(shape);
        //
        //            //if (shape.isDirty() || displayList == null || true) {
        //            if (true || shape.isDirty() || displayList == null) {
        //
        //                if (displayList == null) {
        //                    displayList = new Integer(gl.glGenLists(1));
        //                    displayLists.put(shape, displayList);
        //                }
        //
        //                
        //                gl.glDeleteLists(displayList.intValue(), 1);
        //
        //                //System.out.println ("creating new display list");
        //                //gl.glNewList(displayList.intValue(), GL.GL_COMPILE_AND_EXECUTE);
        //                //gl.glNewList(displayList.intValue(), GL.GL_COMPILE);
        //                shape.render(drawable.getGL());
        //                shape.clean();
        //                //System.out.println ("about to call glEndList()");
        //                System.out.flush();
        //                Thread.yield();
        //                //gl.glEndList();
        //                //System.out.println ("DONE creating new display list");
        //                System.out.flush();
        //                Thread.yield();
        // 
        //            } else {
        //                gl.glCallList(displayList.intValue());
        //            }
        //        }

        for (int i = 0; i < shapes.size(); i++) {
            Shape shape = (Shape) shapes.get(i);
            shape.render(drawable);
        }

        //new ColorScale().render(gl);

        int err = gl.glGetError();

        if (err != GL.GL_NO_ERROR)
            System.out.println("err = " + glu.gluErrorString(err));
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
        gldrawable.display();
    }
    
    public void zoomOut() {
        viewDistance *= 1.1;
        computeEye();
        gldrawable.display();
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

    private Animator animator;
    private volatile float rotateSpeed = 0.5f;

    public JPanel getControlPanel() {
        JPanel panel = new JPanel();
        panel.setBorder(BorderFactory.createLoweredBevelBorder());
        panel.setLayout(new GridBagLayout());

      

        final JCheckBox rotateCheckBox = new JCheckBox("Rotate", animator != null);

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

        rotateCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    if (rotateCheckBox.isSelected()) {
                        animator = new Animator();
                        animator.start();
                    } else {
                        animator.end();
                        animator = null;
                    }

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

        return panel;
    }

}