/*
 * ScatterPlot.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.util.Observable;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.glu.GLU;
import javax.media.opengl.glu.GLUquadric;
import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

/**
 * Draws a scatterplot along 4 axes
 *
 * @author Alan Morris
 *
 * <P>CVS $Id: ScatterPlot.java,v 1.7 2009/08/20 22:09:34 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.7 $
 */
public class ScatterPlot implements Plot {


    // settings
    private float xSize = 15, ySize = 15, zSize = 15;
    private float sphereSize = 0.4f;
    private int sphereDetail = 8;
    private boolean visible = true;
    private boolean normalized = true;
    
    private float[][] values;
    private ColorScale colorScale;
    private Axes axes;

    // rendering details
    private int displayList;
    private boolean dirty = true;

    
    // grr... get rid of these
    private int selectedRow = 5;
    private int selectedCol = 5;

    
    public ScatterPlot() {
    }

    public void setSize(float xSize, float ySize, float zSize) {
        this.xSize = xSize;
        this.ySize = ySize;
        this.zSize = zSize;
        if (axes != null)
            axes.setSize(xSize, ySize, zSize);
        
        if (values != null)
            processValues();
        
        this.dirty = true;
    }

    /**
     * Sets the values.  The 2nd dimension must be of size 4 (one value for each axis).
     * @param values
     */
    public void setValues(float values[][]) {
        this.values = values;
        processValues();
        this.dirty = true;
    }

    /**
     * Returns the current sphere size.
     * @return the current sphere size.
     */
    public float getSphereSize() {
        return sphereSize;
    }

    /**
     * Sets the sphere size.
     * @param sphereSize the desired sphere size.
     */
    public void setSphereSize(float sphereSize) {
        this.sphereSize = sphereSize;
        this.dirty = true;
    }

    public Axes getAxes() {
        return axes;
    }

    public void setAxes(Axes axes) {
        this.axes = axes;
        axes.setSize(xSize, ySize, zSize);
    }

    /**
     * Returns the current sphere detail level.
     * @return the current sphere detail level.
     */
    public int getSphereDetail() {
        return sphereDetail;
    }

    /**
     * Sets the sphere detail level.
     * @param sphereDetail level of detail.
     */
    public void setSphereDetail(int sphereDetail) {
        this.sphereDetail = sphereDetail;
        this.dirty = true;
    }

    /**
     * Get the current associated <tt>ColorScale</tt>.
     * @return the currently associated <tt>ColorScale</tt>.
     */
    public ColorScale getColorScale() {
        return colorScale;
    }

    /**
     * Sets the associated <tt>ColorScale</tt>.  
     * This plot will use this <tt>ColorScale</tt> to resolve colors.
     * @param colorScale The <tt>ColorScale</tt>
     */
    public void setColorScale(ColorScale colorScale) {
        // first, remove ourselves from the previous (if any) colorScale's observer list
        if (this.colorScale != null) {
            this.colorScale.deleteObserver(this);
        }
        this.colorScale = colorScale;
        // add ourselves to the new colorScale
        if (colorScale != null) {
            colorScale.addObserver(this);
        }
    }
    
    public float getWidth() {
        return xSize;
    }

    public float getDepth() {
        return ySize;
    }

    public float getHeight() {
        return zSize;
    }

    public String getName() {
        return "ScatterPlot";
    }

    public void cleanUp() {

    }

    public JPanel getControlPanel(final VisRenderer visRenderer) {
        JPanel panel = new JPanel();

        panel.setLayout(new GridBagLayout());
        panel.setBorder(BorderFactory.createLoweredBevelBorder());

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 0.2;
        gbc.weighty = 0.2;

        final JSlider sphereSizeSlider = new JSlider(0, 20, (int) (sphereSize * 10));
        final JSlider sphereDetailSlider = new JSlider(3, 30, sphereDetail);

        sphereSizeSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    ScatterPlot.this.setSphereSize(sphereSizeSlider.getValue() / 10.0f);
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        sphereDetailSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    ScatterPlot.this.setSphereDetail(sphereDetailSlider.getValue());
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });
        
        VisTools.addCompItem(panel, new JLabel("Point size"), gbc, 0, 0, 1, 1);
        VisTools.addCompItem(panel, sphereSizeSlider, gbc, 1, 0, 1, 1);
        VisTools.addCompItem(panel, new JLabel("Point detail"), gbc, 0, 1, 1, 1);
        VisTools.addCompItem(panel, sphereDetailSlider, gbc, 1, 1, 1, 1);

        return panel;
    }

    public void render(VisRenderer visRenderer) {
        GLAutoDrawable glDrawable = visRenderer.getGLAutoDrawable();
        Vec direction = visRenderer.getViewDirection();

//    public void render(GLAutoDrawable glDrawable, Vec direction) {
        if (axes != null) {
            axes.render(visRenderer);
        }

        if (!visible)
            return;

        GL gl = glDrawable.getGL();

        if (dirty || displayList == 0) {
            displayList = gl.glGenLists(1);
            gl.glNewList(displayList, GL.GL_COMPILE);
            privateRender(glDrawable);
            gl.glEndList();
            dirty = false;
        }
        gl.glCallList(displayList);
    }

    private void processValues() {

        float[] norms = new float[4];
        norms[0] = xSize;
        norms[1] = ySize;
        norms[2] = zSize;
        norms[3] = 1.0f;

        for (int f = 0; f < 4; f++) {
            float maxValue = Float.MIN_VALUE;
            float minValue = Float.MAX_VALUE;

            for (int i = 0; i < values.length; i++) {
                maxValue = Math.max(maxValue, values[i][f]);
                minValue = Math.min(minValue, values[i][f]);
            }

            for (int i = 0; i < values.length; i++) {
                if (maxValue - minValue == 0) {
                    values[i][f] = 0;
                } else {
                    if (normalized) {
                        values[i][f] = (values[i][f] - minValue) / (maxValue - minValue) * norms[f];
                    } else {
                        values[i][f] = values[i][f] / maxValue * norms[f];
                    }
                }
            }

        }
    }

    private void privateRender(GLAutoDrawable glDrawable) {
        if (values == null)
            return;
        
        GL gl = glDrawable.getGL();
        GLU glu = new GLU();

        gl.glShadeModel(GL.GL_SMOOTH);

        // Set to red, in case there is no colorScale
        gl.glColor3f(1.0f, 0, 0);

        if (sphereSize < 0.1f) {
            gl.glDisable(GL.GL_LIGHTING);
            gl.glPointSize(2.5f);
            gl.glBegin(GL.GL_POINTS);
            for (int i = 0; i < values.length; i++) {
                if (colorScale != null) {
                    Color color = colorScale.getColor(values[i][3]);
                    gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);
                }
                gl.glVertex3f(values[i][0], values[i][1], values[i][2]);
            }
            gl.glEnd();
        } else {
            gl.glEnable(GL.GL_LIGHTING);
            gl.glEnable(GL.GL_DEPTH_TEST);
            gl.glFrontFace(GL.GL_CCW);
            GLUquadric qobj = glu.gluNewQuadric();
            gl.glEnable(GL.GL_CULL_FACE);
            glu.gluQuadricDrawStyle(qobj, GLU.GLU_FILL);
            glu.gluQuadricOrientation(qobj, GLU.GLU_OUTSIDE);
            glu.gluQuadricNormals(qobj, GLU.GLU_SMOOTH);

            for (int i = 0; i < values.length; i++) {
                gl.glPushMatrix();
                gl.glTranslatef(values[i][0], values[i][1], values[i][2]);
                if (colorScale != null) {
                    Color color = colorScale.getColor(values[i][3]);
                    gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);
                }
                glu.gluSphere(qobj, sphereSize, sphereDetail, sphereDetail);
                gl.glPopMatrix();
            }
        }

    }

    public void update(Observable o, Object arg) {
        if (o instanceof ColorScale) {
            this.dirty = true;
        }
    }

    public int getSelectedRow() {
        return selectedRow;
    }

    public void setSelectedRow(int selectedRow) {
        this.selectedRow = selectedRow;
    }

    public int getSelectedCol() {
        return selectedCol;
    }

    public void setSelectedCol(int selectedCol) {
        this.selectedCol = selectedCol;
    }

    
    /**
     * Returns whether or not this <tt>ScatterPlot</tt> is normalizing values.
     * @return whether or not this <tt>ScatterPlot</tt> is normalizing values.
     */
    public boolean getNormalized() {
        return normalized;
    }
    /**
     * Sets whether or not this <tt>ScatterPlot</tt> should normalize the values along its axes.
     * @param normalized <tt>true</tt> to normalize; <tt>false</tt> to leave values alone.
     */
    public void setNormalized(boolean normalized) {
        this.normalized = normalized;
    }

    /**
     * Returns whether or not this <tt>ScatterPlot</tt> instance is visible.
     * @return whether or not this <tt>ScatterPlot</tt> instance is visible.
     */
    public boolean getVisible() {
        return visible;
    }

    /**
     * Makes this <tt>ScatterPlot</tt> instance visible or invisible.
     * @param visible <tt>true</tt> to make the <tt>ScatterPlot</tt> visible; <tt>false</tt> to make it invisible.
     */
    public void setVisible(boolean visible) {
        this.visible = visible;
    }

    public void resetCanvas() {
        dirty = true;
        displayList = 0;
        if (axes != null ) {
            axes.resetCanvas();
        }
    }
}
