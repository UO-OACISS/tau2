/*
 * TriangleMeshPlot.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

import java.awt.Color;
import java.awt.Component;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Observable;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.swing.BorderFactory;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import com.sun.opengl.util.GLUT;

/**
 * Draws a 3d triangle mesh.
 * 
 * TODO: Back to front drawing (utilize 'direction') for correct blending.
 *
 * <P>CVS $Id: TriangleMeshPlot.java,v 1.7 2009/08/20 22:09:35 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.7 $
 */
public class TriangleMeshPlot implements Plot {

    private Vec normals[][];
    private float[][] heightValues;
    private float[][] colorValues;

    private GLUT glut = new GLUT();

    private int nrows;
    private int ncols;
    private float xSize, ySize, zSize;
    private boolean dirty = true;

    private List displayLists;

    private ColorScale colorScale;
    private Axes axes;
    private GL gl;

    private int selectedRow = -1;
    private int selectedCol = -1;

    private boolean translucent = false;
    private float translucency = 0.50f;

    public TriangleMeshPlot() {

    }

    /**
     * Initializes this <tt>TriangleMeshPlot</tt> with the given values.
     * @param axes Axes to use for this plot.
     * @param xSize size in x direction.
     * @param ySize size in y direction.
     * @param zSize size in z direction.
     * @param heightValues the height values to use.
     * @param colorValues the color values to use.
     * @param colorScale ColorScale to use for this plot.
     */
    public void initialize(Axes axes, float xSize, float ySize, float zSize, float heightValues[][], float colorValues[][],
            ColorScale colorScale) {
        this.nrows = heightValues.length;
        this.ncols = heightValues[0].length;
        this.heightValues = heightValues;
        this.colorValues = colorValues;
        setColorScale(colorScale);

        this.xSize = xSize;
        this.ySize = ySize;
        this.zSize = zSize;

        this.axes = axes;
        axes.setSize(xSize, ySize, zSize);
        processValues();

        generateNormals();
    }

    /**
     * Resets the data values for this <tt>TriangleMeshPlot</tt>.
     * @param xSize size in x direction.
     * @param ySize size in y direction.
     * @param zSize size in z direction.
     * @param heightValues the height values to use.
     * @param colorValues the color values to use.
     */

    public void setValues(float xSize, float ySize, float zSize, float heightValues[][], float colorValues[][]) {
        this.nrows = heightValues.length;
        if (heightValues.length > 0) {
            this.ncols = heightValues[0].length;
        } else {
            this.ncols = 0;
        }
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

        if (displayLists != null) {
            // delete old displaylists
            for (int i = 0; i < displayLists.size(); i++) {
                gl.glDeleteLists(((Integer) displayLists.get(i)).intValue(), 1);
            }
            displayLists = null;
        }
    }

    public void render(VisRenderer visRenderer) {
        GLAutoDrawable glDrawable = visRenderer.getGLAutoDrawable();
        Vec direction = visRenderer.getViewDirection();

        //    public void render(GLAutoDrawable glDrawable, Vec direction) {

        GL gl = glDrawable.getGL();
        if (gl == null)
            return;

        this.gl = gl;

        if (translucent) {
            renderSelectionTranslucent(gl);
        }

        if (nrows == 1 || ncols == 1) {
            if (axes.getOnEdge() == false) {
                axes.setOnEdge(true);
            }
            axes.render(visRenderer);
            renderLine(gl);

        } else {
            axes.render(visRenderer);
            renderDL(gl);
            //renderDLX(gl);
            //renderVA(gl);
        }

        if (!translucent) {
            renderSelectionRegular(gl);
        }

    }

    // either nrows == 0 or ncols == 0
    private void renderLine(GL gl) {

        gl.glDisable(GL.GL_LIGHTING);
        gl.glEnable(GL.GL_LINE_SMOOTH);
        gl.glShadeModel(GL.GL_SMOOTH);

        if (nrows == 1) {
            float xIncrement = xSize / (ncols - 1);

            gl.glBegin(GL.GL_LINE_STRIP);
            for (int x = 0; x < nrows; x++) {

                float c1 = colorValues[0][x];
                Color color1 = colorScale.getColor(c1);
                gl.glColor3f(color1.getRed() / 255.0f, color1.getGreen() / 255.0f, color1.getBlue() / 255.0f);
                gl.glVertex3f(x * xIncrement, ySize / 2.0f, heightValues[0][x]);
            }
            gl.glEnd();

        } else {
            float yIncrement = ySize / (nrows - 1);

            gl.glBegin(GL.GL_LINE_STRIP);
            for (int y = 0; y < nrows; y++) {

                float c1 = colorValues[y][0];
                Color color1 = colorScale.getColor(c1);
                gl.glColor3f(color1.getRed() / 255.0f, color1.getGreen() / 255.0f, color1.getBlue() / 255.0f);
                gl.glVertex3f(xSize / 2.0f, y * yIncrement, heightValues[y][0]);
            }
            gl.glEnd();
        }

    }

    private void renderDL(GL gl) {

        if (dirty || displayLists == null) {

            if (displayLists != null) {
                // delete old displaylists
                for (int i = 0; i < displayLists.size(); i++) {
                    gl.glDeleteLists(((Integer) displayLists.get(i)).intValue(), 1);
                }
                displayLists = new ArrayList();
            } else {
                displayLists = new ArrayList();
            }

            Integer displayList = new Integer(gl.glGenLists(1));
            displayLists.add(displayList);

            gl.glNewList(displayList.intValue(), GL.GL_COMPILE);

            gl.glFrontFace(GL.GL_CW);
            gl.glEnable(GL.GL_LIGHTING);

            gl.glEnable(GL.GL_CULL_FACE);
            gl.glDisable(GL.GL_CULL_FACE);

            gl.glPushMatrix();

            gl.glShadeModel(GL.GL_SMOOTH);

            if (translucent) {
                gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
                gl.glEnable(GL.GL_BLEND);
            } else {
                gl.glBlendFunc(GL.GL_ONE, GL.GL_ZERO);
                gl.glDisable(GL.GL_BLEND);
            }

            float xIncrement = xSize / (ncols - 1);
            float yIncrement = ySize / (nrows - 1);

            int vertexCount = 0;

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
                    gl.glColor4f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f, translucency);

                    gl.glNormal3f(n1.x(), n1.y(), n1.z());
                    gl.glVertex3f(xPosition, yPosition, v1);

                    color = colorScale.getColor(c2);
                    gl.glColor4f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f, translucency);

                    gl.glNormal3f(n2.x(), n2.y(), n2.z());
                    gl.glVertex3f(xPosition, yPosition + yIncrement, v2);

                    vertexCount += 2;
                }
                gl.glEnd();

                if (vertexCount > 5000) {
                    vertexCount = 0;
                    gl.glEndList();
                    displayList = new Integer(gl.glGenLists(1));
                    displayLists.add(displayList);
                    gl.glNewList(displayList.intValue(), GL.GL_COMPILE);
                }
            }

            gl.glPopMatrix();

            gl.glShadeModel(GL.GL_FLAT);
            gl.glDisable(GL.GL_BLEND);
            gl.glDisable(GL.GL_LIGHTING);

            gl.glEndList();

            VisTools.vout(this, "Created " + displayLists.size() + " display lists");
            dirty = false;
        }

        for (int i = 0; i < displayLists.size(); i++) {
            gl.glCallList(((Integer) displayLists.get(i)).intValue());
        }
    }

    private void renderDLX(GL gl) {

        if (dirty || displayLists == null) {

            if (displayLists != null) {
                // delete old displaylists
                for (int i = 0; i < displayLists.size(); i++) {
                    gl.glDeleteLists(((Integer) displayLists.get(i)).intValue(), 1);
                }
                displayLists = new ArrayList();
            } else {
                displayLists = new ArrayList();
            }

            Integer displayList = new Integer(gl.glGenLists(1));
            displayLists.add(displayList);

            gl.glNewList(displayList.intValue(), GL.GL_COMPILE);

            gl.glFrontFace(GL.GL_CW);
            gl.glEnable(GL.GL_LIGHTING);

            gl.glEnable(GL.GL_CULL_FACE);
            gl.glDisable(GL.GL_CULL_FACE);

            //gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL);
            //gl.glEnable(GL.GL_NORMALIZE);
            gl.glPushMatrix();

            gl.glShadeModel(GL.GL_SMOOTH);

            if (translucent) {
                gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
                gl.glEnable(GL.GL_BLEND);
            } else {
                gl.glBlendFunc(GL.GL_ONE, GL.GL_ZERO);
                gl.glDisable(GL.GL_BLEND);
            }

            float xIncrement = xSize / (ncols - 1);
            float yIncrement = ySize / (nrows - 1);

            int vertexCount = 0;

            for (int x = 0; x < ncols - 1; x++) {
                gl.glBegin(GL.GL_TRIANGLE_STRIP);
                //gl.glBegin(GL.GL_LINES);

                for (int y = 0; y < nrows; y++) {

                    float xPosition = x * xIncrement;
                    float yPosition = y * yIncrement;

                    float v1 = heightValues[y][x];
                    float v2 = heightValues[y][x + 1];

                    float c1 = colorValues[y][x];
                    float c2 = colorValues[y][x + 1];

                    Vec n1 = (Vec) normals[y][x];
                    Vec n2 = (Vec) normals[y][x + 1];

                    Color color = colorScale.getColor(c1);
                    gl.glColor4f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f, translucency);

                    gl.glNormal3f(n1.x(), n1.y(), n1.z());
                    gl.glVertex3f(xPosition, yPosition, v1);

                    color = colorScale.getColor(c2);
                    gl.glColor4f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f, translucency);

                    gl.glNormal3f(n2.x(), n2.y(), n2.z());
                    gl.glVertex3f(xPosition + xIncrement, yPosition, v2);

                    vertexCount += 2;
                }
                gl.glEnd();

                if (vertexCount > 5000) {
                    vertexCount = 0;
                    gl.glEndList();
                    displayList = new Integer(gl.glGenLists(1));
                    displayLists.add(displayList);
                    gl.glNewList(displayList.intValue(), GL.GL_COMPILE);
                }
            }

            gl.glPopMatrix();

            gl.glShadeModel(GL.GL_FLAT);
            gl.glDisable(GL.GL_BLEND);
            gl.glDisable(GL.GL_LIGHTING);

            gl.glEndList();

            VisTools.vout(this, "Created " + displayLists.size() + " display lists");
            dirty = false;
        }

        for (int i = 0; i < displayLists.size(); i++) {
            gl.glCallList(((Integer) displayLists.get(i)).intValue());
        }
    }

    private void renderSelectionTranslucent(GL gl) {
        if (selectedRow < 0 || selectedCol < 0)
            return;

        gl.glFrontFace(GL.GL_CW);
        gl.glDisable(GL.GL_LIGHTING);

        gl.glDisable(GL.GL_CULL_FACE);

        //gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL);
        //gl.glEnable(GL.GL_NORMALIZE);
        //gl.glPushMatrix();

        //gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        //gl.glBlendFunc(GL.GL_ONE, GL.GL_ZERO);

        gl.glShadeModel(GL.GL_SMOOTH);
        gl.glDisable(GL.GL_BLEND);

        float xIncrement = xSize / (ncols - 1);
        float yIncrement = ySize / (nrows - 1);

        gl.glBegin(GL.GL_TRIANGLE_STRIP);

        for (int y = 0; y < nrows; y++) {
            //gl.glBegin(GL.GL_LINES);

            int x = selectedCol;

            //for (int x = 0; x < ncols; x++) {

            float xPosition = x * xIncrement;
            float yPosition = y * yIncrement;

            float v1 = heightValues[y][x];

            float c1 = colorValues[y][x];

            //Vec n1 = (Vec) normals[y][x];

            Color color = colorScale.getColor(c1);
            gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);

            //gl.glColor3f(1,0,0);

            //gl.glNormal3f(n1.x(), n1.y(), n1.z());
            gl.glVertex3f(xPosition, yPosition, v1);
            gl.glVertex3f(xPosition, yPosition, 0);

            //            }
        }
        gl.glEnd();

        gl.glBegin(GL.GL_TRIANGLE_STRIP);
        for (int x = 0; x < ncols; x++) {
            //gl.glBegin(GL.GL_LINES);

            int y = selectedRow;

            //for (int x = 0; x < ncols; x++) {

            float xPosition = x * xIncrement;
            float yPosition = y * yIncrement;

            float v1 = heightValues[y][x];

            float c1 = colorValues[y][x];

            //Vec n1 = (Vec) normals[y][x];

            Color color = colorScale.getColor(c1);
            gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);
            //gl.glColor3f(1,0,0);

            //gl.glNormal3f(n1.x(), n1.y(), n1.z());
            gl.glVertex3f(xPosition, yPosition, v1);
            gl.glVertex3f(xPosition, yPosition, 0);

            //            }
        }
        gl.glEnd();

        gl.glDisable(GL.GL_LIGHTING);

        gl.glLineWidth(4.0f);
        gl.glColor3f(1.0f, 0, 0);
        gl.glBegin(GL.GL_LINE_STRIP);

        for (int y = 0; y < nrows; y++) {
            //gl.glBegin(GL.GL_LINES);

            int x = selectedCol;

            //for (int x = 0; x < ncols; x++) {

            float xPosition = x * xIncrement;
            float yPosition = y * yIncrement;

            float v1 = heightValues[y][x];

            float c1 = colorValues[y][x];

            //Vec n1 = (Vec) normals[y][x];

            //Color color = colorScale.getColor(c1);
            //gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);

            //gl.glColor3f(1,0,0);

            //gl.glNormal3f(n1.x(), n1.y(), n1.z());
            gl.glVertex3f(xPosition, yPosition, v1);

            //            }
        }
        gl.glEnd();

        gl.glBegin(GL.GL_LINE_STRIP);
        for (int x = 0; x < ncols; x++) {
            //gl.glBegin(GL.GL_LINES);

            int y = selectedRow;

            //for (int x = 0; x < ncols; x++) {

            float xPosition = x * xIncrement;
            float yPosition = y * yIncrement;

            float v1 = heightValues[y][x];

            float c1 = colorValues[y][x];

            //Vec n1 = (Vec) normals[y][x];

            //Color color = colorScale.getColor(c1);
            //gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);
            //gl.glColor3f(1,0,0);

            //gl.glNormal3f(n1.x(), n1.y(), n1.z());
            gl.glVertex3f(xPosition, yPosition, v1);
            //gl.glVertex3f(xPosition, yPosition, 0);

            //            }
        }
        gl.glEnd();

    }

    private void renderSelectionRegular(GL gl) {

        if (selectedRow < 0 || selectedCol < 0)
            return;

        if (selectedRow > heightValues.length - 1) {
            selectedRow = heightValues.length - 1;
        }

        if (selectedCol > heightValues[selectedRow].length - 1) {
            selectedCol = heightValues[selectedRow].length - 1;
        }

        float xIncrement = xSize / 2.0f, yIncrement = ySize / 2.0f;

        if (ncols != 1) {
            xIncrement = xSize / (ncols - 1);
        }

        if (nrows != 1) {
            yIncrement = ySize / (nrows - 1);
        }

        gl.glDisable(GL.GL_DEPTH_TEST);
        gl.glEnable(GL.GL_LINE_SMOOTH);
        gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        gl.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST);
        gl.glEnable(GL.GL_BLEND);
        gl.glLineWidth(4.0f);

        float targetX = selectedCol * xIncrement;
        float targetY = selectedRow * yIncrement;
        float targetZ = heightValues[selectedRow][selectedCol];

        if (nrows == 1) {
            targetY += yIncrement;
        }

        if (ncols == 1) {
            targetX += xIncrement;
        }
        gl.glBegin(GL.GL_LINES);

        gl.glColor4f(0, 1, 0, 0.75f);
        gl.glVertex3f(targetX, 0, targetZ);
        gl.glVertex3f(targetX, ySize, targetZ);

        gl.glColor4f(0, 1, 0, 0.75f);
        gl.glVertex3f(0, targetY, targetZ);
        gl.glVertex3f(xSize, targetY, targetZ);

        gl.glColor4f(1, 1, 0, 0.75f);
        gl.glVertex3f(targetX, targetY, 0);
        gl.glVertex3f(targetX, targetY, zSize);

        gl.glEnd();

        gl.glEnable(GL.GL_DEPTH_TEST);
        //gl.glDisable(GL.GL_LINE_SMOOTH);
        //gl.glDisable(GL.GL_BLEND);
        gl.glLineWidth(1.0f);
    }

    //    private void renderVA(GL gl) {
    //
    //        if (gl == null)
    //            return;
    //
    //        this.gl = gl;
    //
    //        gl.glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
    //        gl.glFrontFace(GL.GL_CW);
    //        gl.glEnable(GL.GL_LIGHTING);
    //
    //        gl.glEnable(GL.GL_CULL_FACE);
    //
    //        //gl.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL);
    //        //gl.glEnable(GL.GL_NORMALIZE);
    //        gl.glPushMatrix();
    //
    //        //gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
    //        // gl.glBlendFunc(GL.GL_ONE, GL.GL_ZERO);
    //
    //        gl.glShadeModel(GL.GL_SMOOTH);
    //        //gl.glEnable(GL.GL_BLEND);
    //
    //        float xIncrement = xSize / (ncols - 1);
    //        float yIncrement = ySize / (nrows - 1);
    //
    //        //        float maxValue = Float.MIN_VALUE;
    //        //        for (int y = 0; y < nrows; y++) {
    //        //            for (int x = 0; x < ncols; x++) {
    //        //                float value = heightValues[y][x];
    //        //                maxValue = Math.max(maxValue, value);
    //        //            }
    //        //        }
    //
    //        int err = gl.glGetError();
    //        if (err != GL.GL_NO_ERROR)
    //            System.out.println("0err = " + err);
    //
    //        //dirty = true;
    //
    //        if (dirty || displayList == 0) {
    ////            System.out.println("creating new display lists");
    //            displayList = gl.glGenLists(1);
    //            gl.glNewList(displayList, GL.GL_COMPILE);
    //
    //            //            createArrays();
    //
    //            gl.glEnableClientState(GL.GL_COLOR_ARRAY);
    //            gl.glEnableClientState(GL.GL_VERTEX_ARRAY);
    //            gl.glEnableClientState(GL.GL_NORMAL_ARRAY);
    //
    //            float[] vertexArray = new float[ncols * 2 * 3];
    //            float[] colorArray = new float[ncols * 2 * 3];
    //            float[] normalArray = new float[ncols * 2 * 3];
    //
    //            for (int y = 0; y < nrows - 1; y++) {
    //                FloatBuffer vertexBuffer, colorBuffer, normalBuffer;
    //
    //                int colorIndex = 0;
    //                int vertexIndex = 0;
    //                int normalIndex = 0;
    //
    //                for (int x = 0; x < ncols; x++) {
    //
    //                    float xPosition = x * xIncrement;
    //                    float yPosition = y * yIncrement;
    //
    //                    float v1 = heightValues[y][x];
    //                    float v2 = heightValues[y + 1][x];
    //
    //                    float c1 = colorValues[y][x];
    //                    float c2 = colorValues[y + 1][x];
    //
    //                    Vec n1 = (Vec) normals[y][x];
    //                    Vec n2 = (Vec) normals[y + 1][x];
    //
    //                    Color color = colorScale.getColor(c1);
    //
    //                    //gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);
    //                    colorArray[colorIndex++] = color.getRed() / 255.0f;
    //                    colorArray[colorIndex++] = color.getGreen() / 255.0f;
    //                    colorArray[colorIndex++] = color.getBlue() / 255.0f;
    //                    //gl.glNormal3f(n1.x(), n1.y(), n1.z());
    //                    normalArray[normalIndex++] = n1.x();
    //                    normalArray[normalIndex++] = n1.y();
    //                    normalArray[normalIndex++] = n1.z();
    //                    //gl.glVertex3f(xPosition, yPosition, v1);
    //                    vertexArray[vertexIndex++] = xPosition;
    //                    vertexArray[vertexIndex++] = yPosition;
    //                    vertexArray[vertexIndex++] = v1;
    //
    //                    color = colorScale.getColor(c2);
    //
    //                    //gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);
    //                    colorArray[colorIndex++] = color.getRed() / 255.0f;
    //                    colorArray[colorIndex++] = color.getGreen() / 255.0f;
    //                    colorArray[colorIndex++] = color.getBlue() / 255.0f;
    //                    //gl.glNormal3f(n2.x(), n2.y(), n2.z());
    //                    normalArray[normalIndex++] = n2.x();
    //                    normalArray[normalIndex++] = n2.y();
    //                    normalArray[normalIndex++] = n2.z();
    //                    //gl.glVertex3f(xPosition, yPosition, v2);
    //                    vertexArray[vertexIndex++] = xPosition;
    //                    vertexArray[vertexIndex++] = yPosition + yIncrement;
    //                    vertexArray[vertexIndex++] = v2;
    //
    //                }
    //                //                colorBuffer.position(y*ncols);
    //
    //                colorBuffer = BufferUtils.newFloatBuffer(nrows * ncols * 3);
    //                vertexBuffer = BufferUtils.newFloatBuffer(nrows * ncols * 3);
    //                normalBuffer = BufferUtils.newFloatBuffer(nrows * ncols * 3);
    //
    //                colorBuffer.put(colorArray);
    //                vertexBuffer.put(vertexArray);
    //                normalBuffer.put(normalArray);
    //
    //                gl.glColorPointer(3, GL.GL_FLOAT, 0, colorBuffer);
    //                gl.glVertexPointer(3, GL.GL_FLOAT, 0, vertexBuffer);
    //                gl.glNormalPointer(GL.GL_FLOAT, 0, normalBuffer);
    //
    //                gl.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, ncols * 2);
    //
    //            }
    //            //            gl.glDrawArrays(GL.GL_POINTS, 0, 5);
    //            //            gl.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, vertexArray.length/3);
    //
    //            gl.glEndList();
    //            dirty = false;
    //        }
    //
    //        gl.glCallList(displayList);
    //
    //        err = gl.glGetError();
    //        if (err != GL.GL_NO_ERROR)
    //            System.out.println("1err = " + err);
    //
    //        gl.glPopMatrix();
    //
    //        gl.glShadeModel(GL.GL_FLAT);
    //        //gl.glDisable(GL.GL_BLEND);
    //        gl.glDisable(GL.GL_LIGHTING);
    //
    //    }

    public JPanel getControlPanel(final VisRenderer visRenderer) {

        JPanel sizePanel = new JPanel();
        sizePanel.setBorder(BorderFactory.createLoweredBevelBorder());
        sizePanel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        final JSlider translucencySlider = new JSlider(0, 100, (int) (translucency * 100.f));
        translucencySlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    setTranslucencyRatio(translucencySlider.getValue() / 100.0f);
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        final JSlider plotWidthSlider = new JSlider(5, 400, (int) xSize);
        final JSlider plotDepthSlider = new JSlider(5, 400, (int) ySize);
        final JSlider plotHeightSlider = new JSlider(2, 50, (int) zSize);

        ChangeListener chageListener = new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {

                    //double width = plotWidthSlider.getValue() / 7.5f;
                    // width *= width;

                    //System.out.println(width);

                    TriangleMeshPlot.this.setSize(plotWidthSlider.getValue(), plotDepthSlider.getValue(),
                            plotHeightSlider.getValue());
                    //TriangleMeshPlot.this.setSize((int)width, plotDepthSlider.getValue(),
                    //        plotHeightSlider.getValue());
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        };

        plotWidthSlider.addChangeListener(chageListener);
        plotDepthSlider.addChangeListener(chageListener);
        plotHeightSlider.addChangeListener(chageListener);

        final JCheckBox translucentBox = new JCheckBox("Transparency", translucent);
        translucentBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    setTranslucent(translucentBox.isSelected());
                    visRenderer.redraw();

                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0.0;
        gbc.weighty = 0.0;
        addCompItem(sizePanel, new JLabel("Plot Width"), gbc, 0, 0, 1, 1);
        addCompItem(sizePanel, new JLabel("Plot Depth"), gbc, 0, 1, 1, 1);
        addCompItem(sizePanel, new JLabel("Plot Height"), gbc, 0, 2, 1, 1);
        addCompItem(sizePanel, translucentBox, gbc, 0, 3, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;
        addCompItem(sizePanel, plotWidthSlider, gbc, 1, 0, 1, 1);
        addCompItem(sizePanel, plotDepthSlider, gbc, 1, 1, 1, 1);
        addCompItem(sizePanel, plotHeightSlider, gbc, 1, 2, 1, 1);
        addCompItem(sizePanel, translucencySlider, gbc, 1, 3, 1, 1);

        return sizePanel;
    }

    private void addCompItem(JPanel jPanel, Component c, GridBagConstraints gbc, int x, int y, int w, int h) {
        gbc.gridx = x;
        gbc.gridy = y;
        gbc.gridwidth = w;
        gbc.gridheight = h;
        jPanel.add(c, gbc);
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

    /**
     * Returns whether or not translucency is on.
     * @return whether or not translucency is on.
     */
    public boolean getTranslucent() {
        return translucent;
    }

    /**
     * Turns translucency on/off.
     * @param translucent new translucency setting.
     */
    public void setTranslucent(boolean translucent) {
        this.translucent = translucent;
        this.dirty = true;
    }

    /**
     * Returns current translucency ratio.
     * @return the current translucency value (0..1).
     */
    public float getTranslucencyRatio() {
        return translucency;
    }

    /**
     * Sets the translucency ratio (0..1).
     * @param translucency the new translucency ratio.
     */
    public void setTranslucencyRatio(float translucency) {
        this.translucency = translucency;
        this.dirty = true;
    }

    public void resetCanvas() {
        dirty = true;
        displayLists.clear();
        displayLists = null;
        if (axes != null ) {
            axes.resetCanvas();
        }
    }

}
