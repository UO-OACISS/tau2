/*
 * Created on Mar 22, 2005
 *
 * TODO To change the template for this generated file go to
 * Window - Preferences - Java - Code Style - Code Templates
 */
package edu.uoregon.tau.paraprof.vis;

import java.awt.*;
import java.nio.FloatBuffer;
import java.util.Observable;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import net.java.games.jogl.GL;
import net.java.games.jogl.GLDrawable;
import net.java.games.jogl.util.BufferUtils;
import net.java.games.jogl.util.GLUT;
import edu.uoregon.tau.paraprof.ParaProfUtils;

/**
 * @author amorris
 *
 * TODO ...
 */
public class TriangleMeshPlot implements Plot {

    protected Vec normals[][];
    protected float[][] heightValues;
    protected float[][] colorValues;

    protected GLUT glut = new GLUT();

    protected int nrows;
    protected int ncols;
    protected float xSize, ySize, zSize;
    protected boolean dirty = true;

    protected int displayList;

    protected ColorScale colorScale;
    protected Axes axes;
    protected GL gl;

    private VisRenderer visRenderer;

    private int selectedRow = -1;
    private int selectedCol = -1;

    public TriangleMeshPlot() {

    }

    public void initialize(Axes axes, float xSize, float ySize, float zSize, float heightValues[][],
            float colorValues[][], ColorScale colorScale, VisRenderer visRenderer) {
        this.nrows = heightValues.length;
        this.ncols = heightValues[0].length;
        this.heightValues = heightValues;
        this.colorValues = colorValues;
        this.colorScale = colorScale;

        this.xSize = xSize;
        this.ySize = ySize;
        this.zSize = zSize;

        this.axes = axes;
        axes.setSize(xSize, ySize, zSize);
        processValues();

        generateNormals();

        this.visRenderer = visRenderer;
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
        generateNormals();
        this.dirty = true;
    }

    
//    public void setValues(float heightValues[][], float colorValues[][]) {
//        this.nrows = heightValues.length;
//        this.ncols = heightValues[0].length;
//        this.heightValues = heightValues;
//        this.colorValues = colorValues;
//
//        processValues();
//        generateNormals();
//        this.dirty = true;
//    }
    
    public String getName() {
        return "Mesh Plot";
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

    public JPanel getControlPanel() {

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
                    TriangleMeshPlot.this.setSize(plotWidthSlider.getValue(), plotDepthSlider.getValue(),
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

    private void generateNormals() {

        this.normals = new Vec[nrows][ncols];

        // this can only be done on regularly spaced grids

        for (int y = 0; y < nrows; y++) {
            for (int x = 0; x < ncols; x++) {

                float left, right, up, down;

                left = right = up = down = heightValues[y][x];

                if (x > 0)
                    left = heightValues[y][x - 1];

                if (y > 0)
                    down = heightValues[y - 1][x];

                if (x < ncols - 1)
                    right = heightValues[y][x + 1];

                if (y < nrows - 1)
                    up = heightValues[y + 1][x];

                float slopex = (left - right) / 2;
                float slopey = (down - up) / 2;

                Vec n = new Vec(-slopex, -slopey, 1);
                n.normalize();
                normals[y][x] = n;
            }
        }
    }

    public void setSize(float xSize, float ySize, float zSize) {

        this.xSize = xSize;
        this.ySize = ySize;
        this.zSize = zSize;

        axes.setSize(xSize, ySize, zSize);
        processValues();
        generateNormals();

        this.dirty = true;
    }

    public void cleanUp() {
        normals = null;
        heightValues = null;
        colorValues = null;

        if (displayList != 0) {
            gl.glDeleteLists(displayList, 1);
        }
    }

    public void render(GLDrawable glDrawable) {

        axes.render(glDrawable);

        GL gl = glDrawable.getGL();
        renderDL(gl);
    }

    private void renderDL(GL gl) {

        if (gl == null)
            return;

        this.gl = gl;

        if (dirty || displayList == 0) {
            System.out.println("creating new display lists");
            displayList = gl.glGenLists(1);
            gl.glNewList(displayList, GL.GL_COMPILE);

            //          gl.glColor3f(1.0f, 1.0f, 0.0f);
            gl.glFrontFace(GL.GL_CW);
            gl.glEnable(GL.GL_LIGHTING);

            gl.glEnable(GL.GL_CULL_FACE);

            //gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL);
            //gl.glEnable(GL.GL_NORMALIZE);
            gl.glPushMatrix();

            //gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
            // gl.glBlendFunc(GL.GL_ONE, GL.GL_ZERO);

            gl.glShadeModel(GL.GL_SMOOTH);
            //gl.glEnable(GL.GL_BLEND);

            float xIncrement = xSize / (ncols - 1);
            float yIncrement = ySize / (nrows - 1);

            for (int y = 0; y < nrows - 1; y++) {
                gl.glBegin(GL.GL_TRIANGLE_STRIP);
                //gl.glBegin(GL.GL_LINES);

                for (int x = 0; x < ncols; x++) {

                    float xPosition = x * xIncrement;
                    float yPosition = y * yIncrement;

                    float v1 = heightValues[y][x];
                    float v2 = heightValues[y + 1][x];

                    float c1 = colorValues[y][x];
                    float c2 = colorValues[y + 1][x];

                    Vec n1 = (Vec) normals[y][x];
                    Vec n2 = (Vec) normals[y + 1][x];

                    Color color = colorScale.getColor(c1);
                    gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);

                    gl.glNormal3f(n1.x(), n1.y(), n1.z());
                    gl.glVertex3f(xPosition, yPosition, v1);

                    color = colorScale.getColor(c2);
                    gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);

                    gl.glNormal3f(n2.x(), n2.y(), n2.z());
                    gl.glVertex3f(xPosition, yPosition + yIncrement, v2);

                }
                gl.glEnd();
            }

            gl.glPopMatrix();

            gl.glShadeModel(GL.GL_FLAT);
            //gl.glDisable(GL.GL_BLEND);
            gl.glDisable(GL.GL_LIGHTING);

            gl.glEndList();
            dirty = false;
        }

        gl.glCallList(displayList);

        renderSelection(gl);
    }

    private void renderSelection(GL gl) {

        if (selectedRow < 0 || selectedCol < 0)
            return;

        float xIncrement = xSize / (ncols - 1);
        float yIncrement = ySize / (nrows - 1);

        gl.glDisable(GL.GL_DEPTH_TEST);
        //        
        gl.glEnable(GL.GL_LINE_SMOOTH);
        gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        gl.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST);
        gl.glEnable(GL.GL_BLEND);
        //
        gl.glLineWidth(4.0f);

        float height = heightValues[selectedRow][selectedCol];

        gl.glBegin(GL.GL_LINES);

        gl.glColor4f(0, 1, 0, 0.75f);
        gl.glVertex3f(selectedCol * xIncrement, 0, height);
        gl.glVertex3f(selectedCol * xIncrement, ySize, height);

        gl.glColor4f(0, 1, 0, 0.75f);
        gl.glVertex3f(0, selectedRow * yIncrement, height);
        gl.glVertex3f(xSize, selectedRow * yIncrement, height);

        gl.glColor4f(1, 1, 0, 0.75f);
        gl.glVertex3f(selectedCol * xIncrement, selectedRow * yIncrement, 0);
        gl.glVertex3f(selectedCol * xIncrement, selectedRow * yIncrement, zSize);

        gl.glEnd();

        //        gl.glBegin(GL.GL_QUADS);
        //
        //        gl.glColor4f(0,1,0,0.75f);
        //        gl.glVertex3f(selectedCol*xIncrement, 0, 0);
        //        gl.glVertex3f(selectedCol*xIncrement, 0, zSize);
        //        gl.glVertex3f(selectedCol*xIncrement, ySize, zSize);
        //        gl.glVertex3f(selectedCol*xIncrement, ySize, 0);
        //
        //
        //        gl.glColor4f(1,1,0, 0.75f);
        //        gl.glVertex3f(0, selectedRow*yIncrement, 0);
        //        gl.glVertex3f(xSize, selectedRow*yIncrement, 0);
        //        gl.glVertex3f(xSize, selectedRow*yIncrement, zSize);
        //        gl.glVertex3f(0, selectedRow*yIncrement, zSize);
        //        
        //        gl.glEnd();

        //        
        //        
        gl.glEnable(GL.GL_DEPTH_TEST);
        //
        gl.glDisable(GL.GL_LINE_SMOOTH);
        gl.glDisable(GL.GL_BLEND);
        //
        gl.glLineWidth(1.0f);

    }

    private void renderVA(GL gl) {

        if (gl == null)
            return;

        this.gl = gl;

        gl.glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        gl.glFrontFace(GL.GL_CW);
        gl.glEnable(GL.GL_LIGHTING);

        gl.glEnable(GL.GL_CULL_FACE);

        //gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL);
        //gl.glEnable(GL.GL_NORMALIZE);
        gl.glPushMatrix();

        //gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        // gl.glBlendFunc(GL.GL_ONE, GL.GL_ZERO);

        gl.glShadeModel(GL.GL_SMOOTH);
        //gl.glEnable(GL.GL_BLEND);

        float xIncrement = xSize / (ncols - 1);
        float yIncrement = ySize / (nrows - 1);

        //        float maxValue = Float.MIN_VALUE;
        //        for (int y = 0; y < nrows; y++) {
        //            for (int x = 0; x < ncols; x++) {
        //                float value = heightValues[y][x];
        //                maxValue = Math.max(maxValue, value);
        //            }
        //        }

        int err = gl.glGetError();
        if (err != GL.GL_NO_ERROR)
            System.out.println("0err = " + err);

        //dirty = true;

        if (dirty || displayList == 0) {
//            System.out.println("creating new display lists");
            displayList = gl.glGenLists(1);
            gl.glNewList(displayList, GL.GL_COMPILE);

            //            createArrays();

            gl.glEnableClientState(GL.GL_COLOR_ARRAY);
            gl.glEnableClientState(GL.GL_VERTEX_ARRAY);
            gl.glEnableClientState(GL.GL_NORMAL_ARRAY);

            float[] vertexArray = new float[ncols * 2 * 3];
            float[] colorArray = new float[ncols * 2 * 3];
            float[] normalArray = new float[ncols * 2 * 3];

            for (int y = 0; y < nrows - 1; y++) {
                FloatBuffer vertexBuffer, colorBuffer, normalBuffer;

                int colorIndex = 0;
                int vertexIndex = 0;
                int normalIndex = 0;

                for (int x = 0; x < ncols; x++) {

                    float xPosition = x * xIncrement;
                    float yPosition = y * yIncrement;

                    float v1 = heightValues[y][x];
                    float v2 = heightValues[y + 1][x];

                    float c1 = colorValues[y][x];
                    float c2 = colorValues[y + 1][x];

                    Vec n1 = (Vec) normals[y][x];
                    Vec n2 = (Vec) normals[y + 1][x];

                    Color color = colorScale.getColor(c1);

                    //gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);
                    colorArray[colorIndex++] = color.getRed() / 255.0f;
                    colorArray[colorIndex++] = color.getGreen() / 255.0f;
                    colorArray[colorIndex++] = color.getBlue() / 255.0f;
                    //gl.glNormal3f(n1.x(), n1.y(), n1.z());
                    normalArray[normalIndex++] = n1.x();
                    normalArray[normalIndex++] = n1.y();
                    normalArray[normalIndex++] = n1.z();
                    //gl.glVertex3f(xPosition, yPosition, v1);
                    vertexArray[vertexIndex++] = xPosition;
                    vertexArray[vertexIndex++] = yPosition;
                    vertexArray[vertexIndex++] = v1;

                    color = colorScale.getColor(c2);

                    //gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);
                    colorArray[colorIndex++] = color.getRed() / 255.0f;
                    colorArray[colorIndex++] = color.getGreen() / 255.0f;
                    colorArray[colorIndex++] = color.getBlue() / 255.0f;
                    //gl.glNormal3f(n2.x(), n2.y(), n2.z());
                    normalArray[normalIndex++] = n2.x();
                    normalArray[normalIndex++] = n2.y();
                    normalArray[normalIndex++] = n2.z();
                    //gl.glVertex3f(xPosition, yPosition, v2);
                    vertexArray[vertexIndex++] = xPosition;
                    vertexArray[vertexIndex++] = yPosition + yIncrement;
                    vertexArray[vertexIndex++] = v2;

                }
                //                colorBuffer.position(y*ncols);

                colorBuffer = BufferUtils.newFloatBuffer(nrows * ncols * 3);
                vertexBuffer = BufferUtils.newFloatBuffer(nrows * ncols * 3);
                normalBuffer = BufferUtils.newFloatBuffer(nrows * ncols * 3);

                colorBuffer.put(colorArray);
                vertexBuffer.put(vertexArray);
                normalBuffer.put(normalArray);

                gl.glColorPointer(3, GL.GL_FLOAT, 0, colorBuffer);
                gl.glVertexPointer(3, GL.GL_FLOAT, 0, vertexBuffer);
                gl.glNormalPointer(GL.GL_FLOAT, 0, normalBuffer);

                gl.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, ncols * 2);

            }
            //            gl.glDrawArrays(GL.GL_POINTS, 0, 5);
            //            gl.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, vertexArray.length/3);

            gl.glEndList();
            dirty = false;
        }

        gl.glCallList(displayList);

        err = gl.glGetError();
        if (err != GL.GL_NO_ERROR)
            System.out.println("1err = " + err);

        gl.glPopMatrix();

        gl.glShadeModel(GL.GL_FLAT);
        //gl.glDisable(GL.GL_BLEND);
        gl.glDisable(GL.GL_LIGHTING);

    }

    public void update(Observable o, Object arg) {
        if (o instanceof ColorScale) {
            this.dirty = true;
        }
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
        //this.dirty = true;
    }

    public int getSelectedCol() {
        return selectedCol;
    }

    public void setSelectedCol(int selectedCol) {
        this.selectedCol = selectedCol;
        axes.setSelectedCol(selectedCol);
        //this.dirty = true;
    }
}
