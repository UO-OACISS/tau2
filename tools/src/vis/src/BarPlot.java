/*
 * BarPlot.java
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
 * Draws a 3d bar plot.
 *
 * <P>CVS $Id: BarPlot.java,v 1.9 2009/08/21 23:58:00 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.9 $
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
    private boolean translucent = false;
    private float translucency = 0.50f;

    // if a value is less than this threshold, no box will be drawn 
    // (this is essential to drawing huge datasets (64k+)
    private static float threshold = 0.05f; // 5 percent

    // implementation details
    private int nrows;
    private int ncols;
    private float[][] heightValues, colorValues;
    private boolean dirty = true;
    private List displayLists = new ArrayList();

    private int translucentDisplayListsXplus;
    private int translucentDisplayListsXminus;
    private int translucentDisplayListsYplus;
    private int translucentDisplayListsYminus;
    private int translucentDisplayListsXsize;
    private int translucentDisplayListsYsize;

    private float scaleZ = 1.0f;

    private GL gl;
    
    float maxHeightValue;
    float maxColorValue;

    /**
     * Create a BarPlot with a given Axes and ColorScale
     */
    public BarPlot(Axes axes, ColorScale colorScale) {
        this.axes = axes;
        setColorScale(colorScale);
    }

    public String getName() {
        return "Bar Plot";
    }

    /**
     * Initializes this BarPlot with the given values.
     * @param axes Axes to use for this plot.
     * @param xSize size in x direction.
     * @param ySize size in y direction.
     * @param zSize size in z direction.
     * @param heightValues the height values to use.
     * @param colorValues the color values to use.
     * @param colorScale ColorScale to use for this plot.
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
        this.dirty = true;
    }

    /**
     * Initializes this BarPlot with the given values.
     * @param axes Axes to use for this plot.
     * @param xSize size in x direction.
     * @param ySize size in y direction.
     * @param zSize size in z direction.
     * @param heightValues the height values to use.
     * @param colorValues the color values to use.
     * @param colorScale ColorScale to use for this plot.
     */
    public void setValues(float heightValues[][], float colorValues[][]) {
        this.nrows = heightValues.length;
        if (heightValues.length > 0) {
            this.ncols = heightValues[0].length;
        } else {
            this.ncols = 0;
        }
        this.heightValues = heightValues;
        this.colorValues = colorValues;
        processValues();
        this.dirty = true;
    }

    /**
     * Returns the current barSize.
     * @return the current barSize (0..1).
     */
    public float getBarSize() {
        return barSize;
    }

    /**
     * Sets the barSize.  Value should be between 0 and 1 (1 being full bar size)
     * @param barSize The barSize to set.  
     */
    public void setBarSize(float barSize) {
        this.barSize = barSize;

        // clamp the values
        this.barSize = Math.min(this.barSize, 1.0f);
        this.barSize = Math.max(this.barSize, 0.01f);

        this.dirty = true;
    }

    private void processValues() {
        maxHeightValue = -Float.MAX_VALUE;
        maxColorValue = -Float.MAX_VALUE;
        for (int y = 0; y < nrows; y++) {
            for (int x = 0; x < ncols; x++) {
                float heightValue = heightValues[y][x];
                float colorValue = colorValues[y][x];
                maxHeightValue = Math.max(maxHeightValue, heightValue);
                maxColorValue = Math.max(maxColorValue, colorValue);
            }
        }

//        for (int y = 0; y < nrows; y++) {
//            for (int x = 0; x < ncols; x++) {
//                float heightValue = heightValues[y][x];
//                float colorValue = colorValues[y][x];
//                heightValues[y][x] = heightValue / maxHeightValue * zSize;
//                if (maxHeightValue == 0 ) {
//                    heightValues[y][x] = 0.0f;
//                }
//                colorValues[y][x] = colorValue / maxColorValue;
//                if (maxColorValue == 0) {
//                    colorValues[y][x] = 1.0f;
//                }
//            }
//        }
    }
    
    
    private float getHeight(int x, int y) {
        if (maxHeightValue == 0) {
            return 0;
        }
        return heightValues[y][x] / maxHeightValue * zSize; 
    }

    private float getColor(int x, int y) {
        if (maxColorValue == 0) {
            return 0;
        }
        return colorValues[y][x] / maxColorValue; 
    }

    

    public void setSize(float xSize, float ySize, float zSize) {
        this.xSize = xSize;
        this.ySize = ySize;
        this.zSize = zSize;
        axes.setSize(xSize, ySize, zSize);
        processValues();
        this.dirty = true;
    }

    private void doBox(Vec min, Vec max, float value) {

        if (value >= threshold) {
            // top
            gl.glNormal3f(0, 0, 1);
            gl.glVertex3f(min.x(), min.y(), max.z());
            gl.glVertex3f(max.x(), min.y(), max.z());
            gl.glVertex3f(max.x(), max.y(), max.z());
            gl.glVertex3f(min.x(), max.y(), max.z());

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

            // front
            gl.glNormal3f(0, 1, 0);
            gl.glVertex3f(min.x(), max.y(), min.z());
            gl.glVertex3f(min.x(), max.y(), max.z());
            gl.glVertex3f(max.x(), max.y(), max.z());
            gl.glVertex3f(max.x(), max.y(), min.z());

            // back
            gl.glNormal3f(0, -1, 0);
            gl.glVertex3f(min.x(), min.y(), min.z());
            gl.glVertex3f(max.x(), min.y(), min.z());
            gl.glVertex3f(max.x(), min.y(), max.z());
            gl.glVertex3f(min.x(), min.y(), max.z());
        }

    }

    private void generateTranslucentDisplayLists(GL gl) {

        translucentDisplayListsXplus = gl.glGenLists(ncols);
        translucentDisplayListsYplus = gl.glGenLists(nrows);
        translucentDisplayListsXminus = gl.glGenLists(ncols);
        translucentDisplayListsYminus = gl.glGenLists(nrows);
        translucentDisplayListsXsize = ncols;
        translucentDisplayListsYsize = nrows;

        float xIncrement = xSize / (ncols + 1);
        float yIncrement = ySize / (nrows + 1);

        for (int x = 0; x < ncols; x++) {
            gl.glNewList(translucentDisplayListsXplus + x, GL.GL_COMPILE);
            gl.glBegin(GL.GL_QUADS);
            for (int y = 0; y < nrows; y++) {

                float xPosition = x * xIncrement;
                float yPosition = y * yIncrement;

                Color color = colorScale.getColor(getColor(x,y));

                gl.glColor4f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f, translucency);

                Vec min = new Vec(xPosition, yPosition, 0);
                Vec max = new Vec(xPosition + xIncrement * barSize, yPosition + yIncrement * barSize, getHeight(x,y));

                doBox(min, max, getHeight(x,y));
            }
            gl.glEnd();
            gl.glEndList();

            gl.glNewList(translucentDisplayListsXminus + x, GL.GL_COMPILE);
            gl.glBegin(GL.GL_QUADS);
            for (int y = nrows - 1; y >= 0; y--) {

                float xPosition = x * xIncrement;
                float yPosition = y * yIncrement;

                Color color = colorScale.getColor(getColor(x,y));

                gl.glColor4f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f, translucency);

                Vec min = new Vec(xPosition, yPosition, 0);
                Vec max = new Vec(xPosition + xIncrement * barSize, yPosition + yIncrement * barSize, getHeight(x,y));

                doBox(min, max, getHeight(x,y));
            }
            gl.glEnd();
            gl.glEndList();

        }

        for (int y = 0; y < nrows; y++) {
            gl.glNewList(translucentDisplayListsYplus + y, GL.GL_COMPILE);
            gl.glBegin(GL.GL_QUADS);
            for (int x = 0; x < ncols; x++) {

                float xPosition = x * xIncrement;
                float yPosition = y * yIncrement;

                Color color = colorScale.getColor(getColor(x,y));

                gl.glColor4f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f, translucency);

                Vec min = new Vec(xPosition, yPosition, 0);
                Vec max = new Vec(xPosition + xIncrement * barSize, yPosition + yIncrement * barSize, getHeight(x,y));

                doBox(min, max, getHeight(x,y));
            }
            gl.glEnd();
            gl.glEndList();

            gl.glNewList(translucentDisplayListsYminus + y, GL.GL_COMPILE);
            gl.glBegin(GL.GL_QUADS);
            for (int x = ncols - 1; x >= 0; x--) {

                float xPosition = x * xIncrement;
                float yPosition = y * yIncrement;

                Color color = colorScale.getColor(getColor(x,y));

                gl.glColor4f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f, translucency);

                Vec min = new Vec(xPosition, yPosition, 0);
                Vec max = new Vec(xPosition + xIncrement * barSize, yPosition + yIncrement * barSize, getHeight(x,y));

                doBox(min, max, getHeight(x,y));
            }
            gl.glEnd();
            gl.glEndList();
        }

    }

    private void drawBoxesTranslucent(GL gl, Vec direction) {

        if (dirty) {
            gl.glDeleteLists(translucentDisplayListsXminus, translucentDisplayListsXsize);
            gl.glDeleteLists(translucentDisplayListsYminus, translucentDisplayListsYsize);
            gl.glDeleteLists(translucentDisplayListsXplus, translucentDisplayListsXsize);
            gl.glDeleteLists(translucentDisplayListsYplus, translucentDisplayListsYsize);
            generateTranslucentDisplayLists(gl);
        }

        direction.normalize();
        //float xIncrement = xSize / (ncols + 1);
        //float yIncrement = ySize / (nrows + 1);

        if (Math.abs(direction.x()) > Math.abs(direction.y())) {

            for (int x = 0; x < ncols; x++) {
                if (direction.x() < 0) {
                    if (direction.y() < 0) {
                        gl.glCallList(translucentDisplayListsXplus + x);
                    } else {
                        gl.glCallList(translucentDisplayListsXminus + x);
                    }
                } else {
                    if (direction.y() < 0) {
                        gl.glCallList(translucentDisplayListsXplus + (ncols - 1) - x);
                    } else {
                        gl.glCallList(translucentDisplayListsXminus + (ncols - 1) - x);
                    }

                }
            }
        } else {
            for (int y = 0; y < nrows; y++) {
                if (direction.y() < 0) {

                    if (direction.x() < 0) {
                        gl.glCallList(translucentDisplayListsYplus + y);
                    } else {
                        gl.glCallList(translucentDisplayListsYminus + y);
                    }
                } else {
                    if (direction.x() < 0) {
                        gl.glCallList(translucentDisplayListsYplus + (nrows - 1) - y);
                    } else {
                        gl.glCallList(translucentDisplayListsYminus + (nrows - 1) - y);
                    }
                }
            }
        }

    }

    public void cleanUp() {
        VisTools.vout(this, "Cleaning up!");
        // delete displaylists
        if (displayLists != null) {
            for (int i = 0; i < displayLists.size(); i++) {
                gl.glDeleteLists(((Integer) displayLists.get(i)).intValue(), 1);
            }
            displayLists.clear();
            displayLists = null;
        }
    }

    private void renderOpaque(GLAutoDrawable glDrawable) {

        GL gl = glDrawable.getGL();

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

            gl.glEnable(GL.GL_LIGHTING);
            gl.glFrontFace(GL.GL_CCW);
            gl.glEnable(GL.GL_CULL_FACE);
            //gl.glShadeModel(GL.GL_FLAT);
            gl.glShadeModel(GL.GL_SMOOTH);
            gl.glEnable(GL.GL_DEPTH_TEST);
            gl.glDisable(GL.GL_BLEND);
            gl.glBlendFunc(GL.GL_ONE, GL.GL_ZERO);

            // Experimental
            //gl.glDisable(GL.GL_FOG);

            //gl.glEnable(GL.GL_FOG);

            float xIncrement = xSize / (ncols + 1);
            float yIncrement = ySize / (nrows + 1);

            gl.glPushMatrix();
            gl.glTranslatef(xIncrement - (xIncrement * barSize / 2), yIncrement - (yIncrement * barSize / 2), 0.05f);
            gl.glDepthFunc(GL.GL_LESS);

            gl.glBegin(GL.GL_QUADS);

            float lastRed = 0;
            float lastBlue = 0;
            float lastGreen = 0;

            int colorsSaved = 0, colorsNotSaved = 0, numBoxesSaved = 0, numBoxesUsed = 0;
            int vertexCount = 0;

            for (int y = 0; y < nrows; y++) {
                for (int x = 0; x < ncols; x++) {

                    int yValue = y;
                    int xValue = x;

                    float xPosition = xValue * xIncrement;
                    float yPosition = yValue * yIncrement;

                    Color color = colorScale.getColor(getColor(xValue,yValue));

                    float r = color.getRed() / 255.0f;
                    float g = color.getGreen() / 255.0f;
                    float b = color.getBlue() / 255.0f;

                    if (VisTools.isSufficientlyEqual(r, lastRed) && VisTools.isSufficientlyEqual(g, lastGreen)
                            && VisTools.isSufficientlyEqual(b, lastBlue)) {
                        colorsSaved++;
                    } else {
                        gl.glColor3f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f);
                        lastRed = r;
                        lastBlue = b;
                        lastGreen = g;
                        colorsNotSaved++;
                    }
                    Vec min = new Vec(xPosition, yPosition, 0);
                    Vec max = new Vec(xPosition + xIncrement * barSize, yPosition + yIncrement * barSize,
                            getHeight(xValue,yValue));

                    if (getHeight(xValue,yValue) >= threshold) {

                        // top
                        gl.glNormal3f(0, 0, 1);
                        gl.glVertex3f(min.x(), min.y(), max.z());
                        gl.glVertex3f(max.x(), min.y(), max.z());
                        gl.glVertex3f(max.x(), max.y(), max.z());
                        gl.glVertex3f(min.x(), max.y(), max.z());

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

                        // front
                        gl.glNormal3f(0, 1, 0);
                        gl.glVertex3f(min.x(), max.y(), min.z());
                        gl.glVertex3f(min.x(), max.y(), max.z());
                        gl.glVertex3f(max.x(), max.y(), max.z());
                        gl.glVertex3f(max.x(), max.y(), min.z());

                        // back
                        gl.glNormal3f(0, -1, 0);
                        gl.glVertex3f(min.x(), min.y(), min.z());
                        gl.glVertex3f(max.x(), min.y(), min.z());
                        gl.glVertex3f(max.x(), min.y(), max.z());
                        gl.glVertex3f(min.x(), min.y(), max.z());
                        numBoxesUsed++;
                        vertexCount += 20;
                    } else {
                        numBoxesSaved++;
                        vertexCount += 4;
                    }

                    if (vertexCount > 10000) {
                        vertexCount = 0;
                        gl.glEnd();
                        gl.glEndList();
                        displayList = new Integer(gl.glGenLists(1));
                        displayLists.add(displayList);
                        gl.glNewList(displayList.intValue(), GL.GL_COMPILE);
                        gl.glBegin(GL.GL_QUADS);
                    }

                }

            }

            VisTools.vout(this, "Saved " + colorsSaved + " colors");
            VisTools.vout(this, "Used " + colorsNotSaved + " colors");
            VisTools.vout(this, "Saved " + numBoxesSaved + " boxes");
            VisTools.vout(this, "Used " + numBoxesUsed + " boxes");
            gl.glEnd();
            gl.glPopMatrix();
            gl.glEndList();
            VisTools.vout(this, "Created " + displayLists.size() + " display lists");
        }

        for (int i = 0; i < displayLists.size(); i++) {
            gl.glCallList(((Integer) displayLists.get(i)).intValue());
        }

        renderSelection(gl);

    }

    private void renderTranslucent(GLAutoDrawable glDrawable, Vec direction) {
        GL gl = glDrawable.getGL();
        gl.glEnable(GL.GL_DEPTH_TEST);

        renderSelectionForTranslucency(gl);
        //gl.glDepthMask(false);

        // dirty = true;

        gl.glEnable(GL.GL_LIGHTING);
        gl.glFrontFace(GL.GL_CCW);
        gl.glEnable(GL.GL_CULL_FACE);
        gl.glShadeModel(GL.GL_FLAT);
        gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        gl.glEnable(GL.GL_BLEND);

        float xIncrement = xSize / (ncols + 1);
        float yIncrement = ySize / (nrows + 1);

        gl.glPushMatrix();
        gl.glTranslatef(xIncrement - (xIncrement * barSize / 2), yIncrement - (yIncrement * barSize / 2), 0.05f);

        drawBoxesTranslucent(gl, direction);

        gl.glPopMatrix();

        //gl.glDepthMask(true);

    }

    public void render(VisRenderer visRenderer) {
        GLAutoDrawable glDrawable = visRenderer.getGLAutoDrawable();
        Vec direction = visRenderer.getViewDirection();

        axes.render(visRenderer);
        GL gl = glDrawable.getGL();

        if (gl == null)
            return;

        this.gl = gl;

        gl.glPushMatrix();

        gl.glScalef(1.0f, 1.0f, scaleZ);

        if (translucent) {
            renderTranslucent(glDrawable, direction);
        } else {
            renderOpaque(glDrawable);
        }

        gl.glPopMatrix();

        dirty = false;

    }

    private void renderSelectionForTranslucency(GL gl) {

        if (selectedRow < 0 || selectedCol < 0)
            return;

        float xIncrement = xSize / (ncols + 1);
        float yIncrement = ySize / (nrows + 1);
        gl.glPushMatrix();
        gl.glTranslatef(xIncrement - (xIncrement * barSize / 2), yIncrement - (yIncrement * barSize / 2), 0.05f);

        gl.glFrontFace(GL.GL_CCW);
        gl.glEnable(GL.GL_CULL_FACE);

        //gl.glDisable(GL.GL_LIGHTING);

        //gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        //gl.glEnable(GL.GL_BLEND);
        gl.glBlendFunc(GL.GL_ONE, GL.GL_ZERO);
        gl.glDisable(GL.GL_BLEND);

        gl.glBegin(GL.GL_QUADS);

        int x = selectedCol;
        for (int y = 0; y < nrows; y++) {

            float xPosition = x * xIncrement;
            float yPosition = y * yIncrement;

            Color color = colorScale.getColor(getColor(x,y));

            gl.glColor4f(color.getRed() / 255.0f + 0.25f, color.getGreen() / 255.0f + 0.25f, color.getBlue() / 255.0f + 0.25f,
                    1.0f);

            Vec min = new Vec(xPosition, yPosition, 0);
            Vec max = new Vec(xPosition + xIncrement * barSize, yPosition + yIncrement * barSize, getHeight(x,y));

            // top
            gl.glNormal3f(0, 0, 1);
            gl.glVertex3f(min.x(), min.y(), max.z());
            gl.glVertex3f(max.x(), min.y(), max.z());
            gl.glVertex3f(max.x(), max.y(), max.z());
            gl.glVertex3f(min.x(), max.y(), max.z());

            // no need to draw the bottom
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

        int y = selectedRow;
        for (x = 0; x < ncols; x++) {

            float xPosition = x * xIncrement;
            float yPosition = y * yIncrement;

            Color color = colorScale.getColor(getColor(x,y));

            //            gl.glColor4f(color.getRed() / 255.0f, color.getGreen() / 255.0f, color.getBlue() / 255.0f, 1.0f);
            gl.glColor4f(color.getRed() / 255.0f + 0.25f, color.getGreen() / 255.0f + 0.25f, color.getBlue() / 255.0f + 0.25f,
                    1.0f);

            Vec min = new Vec(xPosition, yPosition, 0);
            Vec max = new Vec(xPosition + xIncrement * barSize, yPosition + yIncrement * barSize, getHeight(x,y));

            // front
            gl.glNormal3f(0, 0, 1);
            gl.glVertex3f(min.x(), min.y(), max.z());
            gl.glVertex3f(max.x(), min.y(), max.z());
            gl.glVertex3f(max.x(), max.y(), max.z());
            gl.glVertex3f(min.x(), max.y(), max.z());

            // no need to draw the bottom
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

        gl.glEnd();
        gl.glPopMatrix();
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
        float height = getHeight(selectedCol,selectedRow);

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

        final JSlider barSizeSlider = new JSlider(1, 100, (int) (barSize * 100.f));
        barSizeSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    BarPlot.this.setBarSize(barSizeSlider.getValue() / 100.0f);
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        final JSlider translucencySlider = new JSlider(0, 100, (int) (translucency * 100.f));
        translucencySlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    BarPlot.this.setTranslucencyRatio(translucencySlider.getValue() / 100.0f);
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
                    BarPlot.this.setSize(plotWidthSlider.getValue(), plotDepthSlider.getValue(), plotHeightSlider.getValue());
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
        addCompItem(sizePanel, new JLabel("Bar Size"), gbc, 0, 3, 1, 1);
        addCompItem(sizePanel, translucentBox, gbc, 0, 4, 1, 1);

        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1.0;
        gbc.weighty = 1.0;
        addCompItem(sizePanel, plotWidthSlider, gbc, 1, 0, 1, 1);
        addCompItem(sizePanel, plotDepthSlider, gbc, 1, 1, 1, 1);
        addCompItem(sizePanel, plotHeightSlider, gbc, 1, 2, 1, 1);
        addCompItem(sizePanel, barSizeSlider, gbc, 1, 3, 1, 1);
        addCompItem(sizePanel, translucencySlider, gbc, 1, 4, 1, 1);
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
        // first, remove ourselves from the previous (if any) colorScale's observer list
        if (this.colorScale != null) {
            this.colorScale.deleteObserver(this);
        }
        this.colorScale = colorScale;
        // add ourselves to the new colorScale's observer list
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

    public float getScaleZ() {
        return scaleZ;
    }

    public void setScaleZ(float scaleZ) {
        this.scaleZ = scaleZ;
    }

    public void resetCanvas() {
        dirty = true;
        displayLists.clear();
        displayLists = null;
        if (axes != null) {
            axes.resetCanvas();
        }
    }
}
