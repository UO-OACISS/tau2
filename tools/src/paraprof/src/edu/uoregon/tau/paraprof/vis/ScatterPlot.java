package edu.uoregon.tau.paraprof.vis;

import java.awt.*;
import java.util.Observable;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.java.games.jogl.*;
import edu.uoregon.tau.paraprof.ParaProfUtils;

/**
 * Draws a scatterplot along 4 axes
 *
 * @author Alan Morris
 *
 * TODO ...
 */
public class ScatterPlot implements Plot {


    // settings
    private float xSize = 15, ySize = 15, zSize = 15;
    private float sphereSize = 0.4f;
    private int sphereDetail = 8;
    private boolean enabled = true;
    private boolean normalized = true;
    
    private float[][] values;
    private ColorScale colorScale;
    private Axes axes;

    // rendering details
    private int displayList;
    private boolean dirty = true;
    private GL gl;

    
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

    public void setValues(float values[][]) {
        this.values = values;
        processValues();
        this.dirty = true;
    }

    public float getSphereSize() {
        return sphereSize;
    }

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

    public int getSphereDetail() {
        return sphereDetail;
    }

    public void setSphereDetail(int sphereDetail) {
        this.sphereDetail = sphereDetail;
        this.dirty = true;
    }

    public ColorScale getColorScale() {
        return colorScale;
    }

    public void setColorScale(ColorScale colorScale) {
        // remove ourselves from the previous (if any) colorScale's observer list
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
                    ParaProfUtils.handleException(e);
                }
            }
        });

        sphereDetailSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    ScatterPlot.this.setSphereDetail(sphereDetailSlider.getValue());
                    visRenderer.redraw();
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }
        });

        ParaProfUtils.addCompItem(panel, new JLabel("Point size"), gbc, 0, 0, 1, 1);
        ParaProfUtils.addCompItem(panel, sphereSizeSlider, gbc, 1, 0, 1, 1);
        ParaProfUtils.addCompItem(panel, new JLabel("Point detail"), gbc, 0, 1, 1, 1);
        ParaProfUtils.addCompItem(panel, sphereDetailSlider, gbc, 1, 1, 1, 1);

        return panel;
    }

    public void render(VisRenderer visRenderer) {
        GLDrawable glDrawable = visRenderer.getGLDrawable();
        Vec direction = visRenderer.getViewDirection();

//    public void render(GLDrawable glDrawable, Vec direction) {
        if (axes != null)
            axes.render(visRenderer);

        if (!enabled)
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

    private void privateRender(GLDrawable glDrawable) {
        if (values == null)
            return;
        
        GL gl = glDrawable.getGL();
        GLU glu = glDrawable.getGLU();

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

    public boolean getEnabled() {
        return enabled;
    }
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }
    public boolean getNormalized() {
        return normalized;
    }
    public void setNormalized(boolean normalized) {
        this.normalized = normalized;
    }
}
