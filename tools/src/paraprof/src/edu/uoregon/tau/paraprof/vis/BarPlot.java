package edu.uoregon.tau.paraprof.vis;

import java.awt.*;
import java.util.Iterator;
import java.util.Observable;
import java.util.Vector;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.java.games.jogl.GL;
import net.java.games.jogl.GLDrawable;
import net.java.games.jogl.util.GLUT;
import edu.uoregon.tau.paraprof.ParaProfUtils;

/**
 * Draws 3d bars
 * 
 * @author amorris
 */
public class BarPlot implements Plot {


    private final GLUT glut = new GLUT();

    // settings that the user can tweak
    private float xSize, ySize, zSize;
    private float barSize = 0.9f; // 0..1
    private ColorScale colorScale;
    private Axes axes;
    private int selectedRow = -1;
    private int selectedCol = -1;

    // implementation details
    private int nrows;
    private int ncols;
    private float[][] heightValues, colorValues;
    private boolean dirty = true;
    private int displayList;


    public BarPlot() {

    }

    public void initialize(Axes axes, float xSize, float ySize, float zSize, float heightValues[][],
            float colorValues[][], ColorScale colorScale) {

        setColorScale(colorScale);
        this.nrows = heightValues.length;
        this.ncols = heightValues[0].length;
        this.heightValues = heightValues;
        this.colorValues = colorValues;

        this.xSize = xSize;
        this.ySize = ySize;
        this.zSize = zSize;

        this.axes = axes;
        axes.setSize(xSize, ySize, zSize);

        processValues();
    }

    public String getName() {
        return "Bar Plot";
    }

    public void setValues(float xSize, float ySize, float zSize, float heightValues[][], float colorValues[][]) {
        this.nrows = heightValues.length;
        this.ncols = heightValues[0].length;
        this.heightValues = heightValues;
        this.colorValues = colorValues;
        this.xSize = xSize;
        this.ySize = ySize;
        this.zSize = zSize;

        axes.setSize(xSize, ySize, zSize);
        processValues();
        this.dirty = true;
    }

    /**
     * @return Returns the barSize.
     */
    public float getBarSize() {
        return barSize;
    }

    /**
     * @param barSize The barSize to set.  Value should be between 0 and 1 (1 being full bar size)
     */
    public void setBarSize(float barSize) {
        this.barSize = barSize;
        this.dirty = true;
    }

    private void processValues() {
        float maxHeightValue = Float.MIN_VALUE;
        float maxColorValue = Float.MIN_VALUE;
        for (int y = 0; y < nrows; y++) {
            for (int x = 0; x < ncols; x++) {
                float heightValue = heightValues[y][x];
                float colorValue = colorValues[y][x];
                maxHeightValue = Math.max(maxHeightValue, heightValue);
                maxColorValue = Math.max(maxColorValue, colorValue);
            }
        }

        for (int y = 0; y < nrows; y++) {
            for (int x = 0; x < ncols; x++) {
                float heightValue = heightValues[y][x];
                float colorValue = colorValues[y][x];
                heightValues[y][x] = heightValue / maxHeightValue * zSize;
                colorValues[y][x] = colorValue / maxColorValue;
            }
        }
    }

    public void setSize(float xSize, float ySize, float zSize) {
        this.xSize = xSize;
        this.ySize = ySize;
        this.zSize = zSize;
        axes.setSize(xSize, ySize, zSize);
        processValues();
        this.dirty = true;
    }

    private void optRender(GL gl) {

        float xIncrement = xSize / (ncols + 1);
        float yIncrement = ySize / (nrows + 1);

        for (int y = 0; y < nrows; y++) {
            for (int x = 0; x < ncols; x++) {

                float xPosition = x * xIncrement;
                float yPosition = y * yIncrement;

                Color color = colorScale.getColor(colorValues[y][x]);

                gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);

                Vec min = new Vec(xPosition, yPosition, 0);
                Vec max = new Vec(xPosition + xIncrement * barSize, yPosition + yIncrement * barSize,
                        heightValues[y][x]);

                // front
                gl.glNormal3f(0, 0, 1);
                gl.glVertex3f(min.x(), min.y(), max.z());
                gl.glVertex3f(max.x(), min.y(), max.z());
                gl.glVertex3f(max.x(), max.y(), max.z());
                gl.glVertex3f(min.x(), max.y(), max.z());

                // no need to draw the bottom
                //              // back
                //gl.glNormal3f(0,0,-1);
                //gl.glVertex3f(min.x(),min.y(),min.z());
                //gl.glVertex3f(min.x(),max.y(),min.z());
                //gl.glVertex3f(max.x(),max.y(),min.z());
                //gl.glVertex3f(max.x(),min.y(),min.z());

                // left
                gl.glNormal3f(-1, 0, 0);
                gl.glVertex3f(min.x(), min.y(), min.z());
                gl.glVertex3f(min.x(), min.y(), max.z());
                gl.glVertex3f(min.x(), max.y(), max.z());
                gl.glVertex3f(min.x(), max.y(), min.z());

                // right
                gl.glNormal3f(1, 0, 0);
                gl.glVertex3f(max.x(), min.y(), min.z());
                gl.glVertex3f(max.x(), max.y(), min.z());
                gl.glVertex3f(max.x(), max.y(), max.z());
                gl.glVertex3f(max.x(), min.y(), max.z());

                // top
                gl.glNormal3f(0, 1, 0);
                gl.glVertex3f(min.x(), max.y(), min.z());
                gl.glVertex3f(min.x(), max.y(), max.z());
                gl.glVertex3f(max.x(), max.y(), max.z());
                gl.glVertex3f(max.x(), max.y(), min.z());

                // bottom
                gl.glNormal3f(0, -1, 0);
                gl.glVertex3f(min.x(), min.y(), min.z());
                gl.glVertex3f(max.x(), min.y(), min.z());
                gl.glVertex3f(max.x(), min.y(), max.z());
                gl.glVertex3f(min.x(), min.y(), max.z());

            }
        }
    }

    public void cleanUp() {

    }

    public void render(GLDrawable glDrawable) {

        axes.render(glDrawable);
        GL gl = glDrawable.getGL();

        if (gl == null)
            return;

        if (dirty || displayList == 0) {
            displayList = gl.glGenLists(1);
            gl.glNewList(displayList, GL.GL_COMPILE_AND_EXECUTE);

            gl.glEnable(GL.GL_LIGHTING);
            gl.glFrontFace(GL.GL_CCW);
            gl.glEnable(GL.GL_CULL_FACE);
            gl.glShadeModel(GL.GL_FLAT);

            float xIncrement = xSize / (ncols + 1);
            float yIncrement = ySize / (nrows + 1);

            gl.glPushMatrix();
            gl.glTranslatef(xIncrement - (xIncrement * barSize / 2), yIncrement - (yIncrement * barSize / 2),
                    0.05f);

            gl.glBegin(GL.GL_QUADS);
            optRender(gl);

            gl.glEnd();
            gl.glPopMatrix();

            gl.glEndList();
            dirty = false;

        }

        gl.glCallList(displayList);

        renderSelection(gl);

    }

    private void renderSelection(GL gl) {

        if (selectedRow < 0 || selectedCol < 0)
            return;

        gl.glPushMatrix();

        float xIncrement = xSize / (ncols + 1);
        float yIncrement = ySize / (nrows + 1);

        gl.glDisable(GL.GL_LIGHTING);
        //        
        gl.glEnable(GL.GL_LINE_SMOOTH);
        gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        gl.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST);
        gl.glEnable(GL.GL_BLEND);
        //
        gl.glLineWidth(4.0f);

        gl.glDepthFunc(GL.GL_LEQUAL);

        gl.glDisable(GL.GL_CULL_FACE);
        float height = heightValues[selectedRow][selectedCol];

        gl.glTranslatef(0, 0, 0.055f);

        gl.glBegin(GL.GL_QUADS);

        float x = (selectedCol + 1) * xIncrement - xIncrement * barSize / 2;
        float y = (selectedRow + 1) * yIncrement - yIncrement * barSize / 2;

        gl.glColor4f(0, 1, 0, 0.75f);
        gl.glVertex3f(x, 0, height);
        gl.glVertex3f(x + xIncrement * barSize, 0, height);
        gl.glVertex3f(x + xIncrement * barSize, ySize, height);
        gl.glVertex3f(x, ySize, height);
        gl.glEnd();

        gl.glTranslatef(0, 0, 0.005f);
        gl.glBegin(GL.GL_QUADS);

        gl.glColor4f(1, 1, 0, 0.75f);
        gl.glVertex3f(0, y, height);
        gl.glVertex3f(0, y + yIncrement * barSize, height);
        gl.glVertex3f(xSize, y + yIncrement * barSize, height);
        gl.glVertex3f(xSize, y, height);

        gl.glEnd();

        gl.glDisable(GL.GL_LINE_SMOOTH);
        gl.glDisable(GL.GL_BLEND);
        gl.glLineWidth(1.0f);
        gl.glPopMatrix();

    }

   

    public JPanel getControlPanel(final VisRenderer visRenderer) {

        JPanel sizePanel = new JPanel();
        sizePanel.setBorder(BorderFactory.createLoweredBevelBorder());
        sizePanel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        final JSlider plotWidthSlider = new JSlider(5, 100, (int) xSize);
        final JSlider plotDepthSlider = new JSlider(5, 100, (int) ySize);
        final JSlider plotHeightSlider = new JSlider(2, 40, (int) zSize);

        ChangeListener chageListener = new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    BarPlot.this.setSize(plotWidthSlider.getValue(), plotDepthSlider.getValue(),
                            plotHeightSlider.getValue());
                    visRenderer.redraw();
                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }
        };

        plotWidthSlider.addChangeListener(chageListener);
        plotDepthSlider.addChangeListener(chageListener);
        plotHeightSlider.addChangeListener(chageListener);

        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0.0;
        gbc.weighty = 0.0;
        addCompItem(sizePanel, new JLabel("Plot Width"), gbc, 0, 0, 1, 1);
        addCompItem(sizePanel, new JLabel("Plot Depth"), gbc, 0, 1, 1, 1);
        addCompItem(sizePanel, new JLabel("Plot Height"), gbc, 0, 2, 1, 1);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;
        addCompItem(sizePanel, plotWidthSlider, gbc, 1, 0, 1, 1);
        addCompItem(sizePanel, plotDepthSlider, gbc, 1, 1, 1, 1);
        addCompItem(sizePanel, plotHeightSlider, gbc, 1, 2, 1, 1);
        return sizePanel;
    }

    private void addCompItem(JPanel jPanel, Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        jPanel.add(c, gbc);
    }

    /* (non-Javadoc)
     * @see java.util.Observer#update(java.util.Observable, java.lang.Object)
     */
    public void update(Observable o, Object arg) {
        if (o instanceof ColorScale) {
            this.dirty = true;
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
    
    public Axes getAxes() {
        return axes;
    }

    public void setAxes(Axes axes) {
        this.axes = axes;
    }

    public int getSelectedRow() {
        return selectedRow;
    }

    public void setSelectedRow(int selectedRow) {
        this.selectedRow = selectedRow;
        axes.setSelectedRow(selectedRow);
    }

    public int getSelectedCol() {
        return selectedCol;
    }

    public void setSelectedCol(int selectedCol) {
        this.selectedCol = selectedCol;
        axes.setSelectedCol(selectedCol);
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

}
