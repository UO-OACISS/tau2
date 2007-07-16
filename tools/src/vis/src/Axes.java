/*
 * Axes.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import com.sun.opengl.util.GLUT;

/**
 * Draws axes with labels.
 *
 * <P>CVS $Id: Axes.java,v 1.8 2007/07/16 17:12:50 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.8 $
 */
public class Axes implements Shape {

    private GLUT glut = new GLUT();

    private float xSize, ySize, zSize;

    private String xlabel, ylabel, zlabel;

    private int xTickSkip, yTickSkip, zTickSkip;
    private int xLabelSkip, yLabelSkip, zLabelSkip;
    private List xStrings;
    private List yStrings;
    private List zStrings;

    private boolean autoSkip = true;
    private boolean onEdge;
    private boolean dirty = true;
    private Orientation orientation = Orientation.NW;

    private int font = GLUT.STROKE_MONO_ROMAN;

    private float stringSize = 3;
    private float labelSize = 8;
    private float fontScale = 1200;

    private boolean enabled = true;
    private Color highlightColor = new Color(1, 0, 0);

    private int displayList;

    private int selectedRow = -1;
    private int selectedCol = -1;

    private Color textColor = Color.white;
    private Color majorColor = Color.white;
    private Color minorColor = new Color(0.5f, 0.5f, 0.5f);

    // until I add a proper event interface, each component will just keep track of the old values
    // and check on each render if it needs to recreate it's display lists

    // this is to keep track of the old reverseVideo value
    // I need to come up with a better way of tracking the settings
    // we have to know whether to recreate the display list or not
    private boolean oldReverseVideo;
    private boolean oldAntiAlias;

    /**
     * Typesafe enum for axes orientation
     */
    public static class Orientation {

        private final String name;

        private Orientation(String name) {
            this.name = name;
        }

        public String toString() {
            return name;
        }

        /**
         * North West
         */
        public static final Orientation NW = new Orientation("NW");
        /**
         * North East
         */
        public static final Orientation NE = new Orientation("NE");
        /**
         * South West
         */
        public static final Orientation SW = new Orientation("SW");
        /**
         * South East
         */
        public static final Orientation SE = new Orientation("SE");

    }

    /**
     * Returns whether or not this <tt>Axes</tt> instance is visible.
     * @return whether or not this <tt>Axes</tt> instance is visible.
     */
    public boolean getVisible() {
        return enabled;
    }

    /**
     * Makes this <tt>Axes</tt> instance visible or invisible.
     * @param visible <tt>true</tt> to make the <tt>Axes</tt> visible; <tt>false</tt> to make it invisible.
     */
    public void setVisible(boolean visible) {
        this.enabled = visible;
        this.dirty = true;
    }

    /**
     * Sets the string labels
     * @param xlabel label for the x axis
     * @param ylabel label for the y axis
     * @param zlabel label for the z axis
     * @param xStrings List of strings for the x axis
     * @param yStrings List of strings for the y axis
     * @param zStrings List of strings for the z axis
     */
    public void setStrings(String xlabel, String ylabel, String zlabel, List xStrings, List yStrings, List zStrings) {
        this.xlabel = xlabel;
        this.ylabel = ylabel;
        this.zlabel = zlabel;
        this.xStrings = xStrings;
        this.yStrings = yStrings;
        this.zStrings = zStrings;

        if (this.xStrings == null)
            this.xStrings = new ArrayList();

        if (this.yStrings == null)
            this.yStrings = new ArrayList();

        if (this.zStrings == null)
            this.zStrings = new ArrayList();

        setAutoTickSkip();
        this.dirty = true;

    }

    /**
     * Sets whether or not the data points land on the intersection of two lines of the axes or inbetween.
     * @param onEdge whether or not the axes are offset
     */
    public void setOnEdge(boolean onEdge) {
        this.onEdge = onEdge;
        this.dirty = true;
    }

    /**
     * Returns the onEdge property
     * @return whether the data is on edge or not
     * @see #setOnEdge(boolean)
     */
    public boolean getOnEdge() {
        return this.onEdge;
    }

    /**
     * Sets the size of the axes.  Note that this is usually called by the plot.
     * @param xSize size in the x direction
     * @param ySize size in the y direction
     * @param zSize size in the z direction
     */
    public void setSize(float xSize, float ySize, float zSize) {
        this.xSize = xSize;
        this.ySize = ySize;
        this.zSize = zSize;
        if (this.autoSkip)
            setAutoTickSkip();
        this.dirty = true;
    }

    /**
     * Sets the highlight color (for the selectedRow, selectedCol)
     * @param color the color to use
     */
    public void setHighlightColor(Color color) {
        highlightColor = color;
    }

    /**
     * Gets the current orientation.
     * @return Returns the orientation.
     */
    public Orientation getOrientation() {
        return orientation;
    }

    /**
     * Sets the orientation.
     * @param orientation The orientation to set.
     */
    public void setOrientation(Orientation orientation) {
        this.orientation = orientation;
        this.dirty = true;
    }

    /**
     * Creates a Swing JPanel with controls for this object.  These controls will 
     * change the state of the axes and automatically call visRenderer.redraw().<p>
     * 
     * When getControlPanel() is called, the controls will represent the current
     * values for the object, but currently, they will not stay in sync if the values
     * are changed using the public methods.  For example, if you call "setEnabled(false)"
     * The JCheckBox will not be set to unchecked.  This functionality could be added if
     * requested.
     * 
     * @param visRenderer The associated VisRenderer
     * @return the control panel for this component
     */
    public JPanel getControlPanel(final VisRenderer visRenderer) {
        JPanel axesPanel = new JPanel();

        axesPanel.setBorder(BorderFactory.createLoweredBevelBorder());
        axesPanel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);

        final JCheckBox enabledCheckBox = new JCheckBox("Show Axes", enabled);

        enabledCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    Axes.this.setVisible(enabledCheckBox.isSelected());
                    visRenderer.redraw();

                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        ActionListener actionListener = new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {

                    String arg = evt.getActionCommand();

                    if (arg.equals("NW")) {
                        Axes.this.setOrientation(Orientation.NW);
                    } else if (arg.equals("SW")) {
                        Axes.this.setOrientation(Orientation.SW);
                    } else if (arg.equals("SE")) {
                        Axes.this.setOrientation(Orientation.SE);
                    } else if (arg.equals("NE")) {
                        Axes.this.setOrientation(Orientation.NE);
                    }

                    visRenderer.redraw();

                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        };

        
        final JSlider fontScaleSlider = new JSlider(0,4000,4600-(int)getFontScale());
        fontScaleSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    Axes.this.setFontScale(4600-(float)(fontScaleSlider.getValue()));
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });    

        

        
        ButtonGroup group = new ButtonGroup();

        JRadioButton nw = new JRadioButton("NW", orientation == Orientation.NW);
        JRadioButton ne = new JRadioButton("NE", orientation == Orientation.NW);
        JRadioButton sw = new JRadioButton("SE", orientation == Orientation.SE);
        JRadioButton se = new JRadioButton("SW", orientation == Orientation.SW);

        group.add(nw);
        group.add(ne);
        group.add(sw);
        group.add(se);

        nw.addActionListener(actionListener);
        ne.addActionListener(actionListener);
        se.addActionListener(actionListener);
        sw.addActionListener(actionListener);

        nw.setHorizontalTextPosition(JButton.LEADING);
        sw.setHorizontalTextPosition(JButton.LEADING);

        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0.2;
        gbc.weighty = 0.2;
        VisTools.addCompItem(axesPanel, enabledCheckBox, gbc, 0, 0, 2, 2);
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.fill = GridBagConstraints.NONE;
        VisTools.addCompItem(axesPanel, new JLabel("Orientation"), gbc, 2, 0, 2, 1);

        gbc.anchor = GridBagConstraints.EAST;
        gbc.fill = GridBagConstraints.NONE;

        VisTools.addCompItem(axesPanel, nw, gbc, 2, 1, 1, 1);
        VisTools.addCompItem(axesPanel, sw, gbc, 2, 2, 1, 1);

        gbc.anchor = GridBagConstraints.WEST;
        VisTools.addCompItem(axesPanel, ne, gbc, 3, 1, 1, 1);
        VisTools.addCompItem(axesPanel, se, gbc, 3, 2, 1, 1);

        
        gbc.anchor = GridBagConstraints.CENTER;
        VisTools.addCompItem(axesPanel, new JLabel("Font Size"),gbc,0,2,1,1);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        VisTools.addCompItem(axesPanel, fontScaleSlider,gbc,0,3,1,1);
        
        return axesPanel;
    }

    /**
     * Returns the currently selected column.
     * @return the currently selected column.
     */
    public int getSelectedCol() {
        return selectedCol;
    }

    /**
     * Sets the selected column.
     * @param selectedCol the selected column.
     */
    public void setSelectedCol(int selectedCol) {
        this.selectedCol = selectedCol;
        dirty = true;
    }

    /**
     * Returns the currently selected row.
     * @return the currently selected row.
     */
    public int getSelectedRow() {
        return selectedRow;
    }

    /**
     * Sets the selected row.
     * @param selectedRow the selected row.
     */
    public void setSelectedRow(int selectedRow) {
        this.selectedRow = selectedRow;
        dirty = true;
    }

    public void render(VisRenderer visRenderer) {
        GLAutoDrawable glDrawable = visRenderer.getGLAutoDrawable();

        // If the reverse video setting has changed, we must redraw
        if (oldReverseVideo != visRenderer.getReverseVideo()) {
            dirty = true;
        }
        oldReverseVideo = visRenderer.getReverseVideo();

        if (oldAntiAlias != visRenderer.getAntiAliasedLines()) {
            dirty = true;
        }
        oldAntiAlias = visRenderer.getAntiAliasedLines();

        if (!enabled)
            return;

        GL gl = glDrawable.getGL();

        if (dirty || displayList == 0) {
            if (displayList == 0) {
                displayList = gl.glGenLists(1);
            }
            gl.glNewList(displayList, GL.GL_COMPILE);
            privateRender(visRenderer);
            gl.glEndList();
            dirty = false;
        }
        gl.glCallList(displayList);
    }

    private void setTickSkipping(int xTicSkip, int yTicSkip, int zTicSkip, int xLabelSkip, int yLabelSkip, int zLabelSkip) {

        this.xTickSkip = xTicSkip;
        this.yTickSkip = yTicSkip;
        this.zTickSkip = zTicSkip;

        this.xLabelSkip = xLabelSkip;
        this.yLabelSkip = yLabelSkip;
        this.zLabelSkip = zLabelSkip;
    }

    private void setAutoTickSkip() {
        this.xLabelSkip = (int) (xStrings.size() / (xSize * 2));
        this.yLabelSkip = (int) (yStrings.size() / (ySize * 2));
        this.zLabelSkip = (int) (zStrings.size() / (zSize * 2));

        if (xLabelSkip > 0) {
            this.xTickSkip = xLabelSkip / 3;
        } else {
            this.xTickSkip = 0;
        }

        if (yLabelSkip > 0) {
            this.yTickSkip = yLabelSkip / 3;
        } else {
            this.yTickSkip = 0;
        }

        if (zLabelSkip > 0) {
            this.zTickSkip = zLabelSkip / 3;
        } else {
            this.zTickSkip = 0;
        }

    }

    private void applyMajorColor(VisRenderer visRenderer) {
        //gl.glColor3f(0.9f, 0.9f, 0.9f);
        //gl.glColor3f(1.0f, 0, 0);
        VisTools.glApplyInvertableColor(visRenderer, majorColor);

    }

    private void applyTextColor(VisRenderer visRenderer) {
        // gl.glColor3f(0, 0, 1);
        VisTools.glApplyInvertableColor(visRenderer, textColor);
    }

    private void applyMinorColor(VisRenderer visRenderer) {
        //gl.glColor3f(0.5f, 0.5f, 0.5f);
        //gl.glColor3f(0,1,0);
        VisTools.glApplyInvertableColor(visRenderer, minorColor);
    }

    private void drawGrid(VisRenderer visRenderer, int numx, int numy, float xSize, float ySize, int xLabelSkip, int yLabelSkip) {
        GL gl = visRenderer.getGLAutoDrawable().getGL();
        applyMinorColor(visRenderer);

        gl.glBegin(GL.GL_LINES);
        int xOffset = 0;
        int yOffset = 0;
        if (onEdge) {
            xOffset = 1;
            yOffset = 1;
        }

        // grid for x-y plane
        for (int x = 0; x < numx; x++) {
            if ((x - xOffset) % (xLabelSkip + 1) == 0 || x == numx - 1) {
                if (x == 0 || x == numx - 1) {
                    applyMajorColor(visRenderer);
                }
                float position = (float) x / (numx - 1) * xSize;
                gl.glVertex3f(position, 0, 0);
                gl.glVertex3f(position, ySize, 0);
                if (x == 0 || x == numx - 1) {
                    applyMinorColor(visRenderer);
                }
            }
        }
        for (int y = 0; y < numy; y++) {
            if ((y - yOffset) % (yLabelSkip + 1) == 0 || y == numy - 1) {
                if (y == 0 || y == numy - 1) {
                    applyMajorColor(visRenderer);
                }
                float pos = (float) y / (numy - 1) * ySize;
                gl.glVertex3f(0, pos, 0);
                gl.glVertex3f(xSize, pos, 0);
                if (y == 0 || y == numy - 1) {
                    applyMinorColor(visRenderer);
                }
            }
        }
        gl.glEnd();

    }

    private void privateRender(VisRenderer visRenderer) {
        if (!enabled)
            return;

        GL gl = visRenderer.getGLAutoDrawable().getGL();

        // don't highlight on if they're not both on
        if (selectedRow < 0 || selectedCol < 0) {
            selectedRow = -1;
            selectedCol = -1;
        }

        int numx = this.xStrings.size();
        int numy = this.yStrings.size();
        int numz = this.zStrings.size();

        int xOffset = 0;
        int yOffset = 0;
        if (onEdge) {
            numx += 2;
            numy += 2;
            xOffset = 1;
            yOffset = 1;
        }

        int zOffset = 0;

        gl.glDisable(GL.GL_LIGHTING);

        if (visRenderer.getAntiAliasedLines()) {
            gl.glEnable(GL.GL_LINE_SMOOTH);
            gl.glEnable(GL.GL_BLEND);
            gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
            gl.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST);
        } else {
            gl.glDisable(GL.GL_LINE_SMOOTH);
            gl.glDisable(GL.GL_BLEND);
        }

        gl.glLineWidth(1.0f);

        // grid for x-y plane
        drawGrid(visRenderer, numx, numy, xSize, ySize, this.xLabelSkip, this.yLabelSkip);

        // grid for x-z plane
        gl.glPushMatrix();
        if (orientation == Orientation.NE || orientation == Orientation.SE)
            gl.glTranslatef(0, ySize, 0);
        gl.glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
        drawGrid(visRenderer, numx, numz, xSize, zSize, this.xLabelSkip, this.zLabelSkip);
        gl.glPopMatrix();

        // grid for y-z plane
        gl.glPushMatrix();
        if (orientation == Orientation.SE || orientation == Orientation.SW)
            gl.glTranslatef(xSize, 0, 0);
        gl.glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
        gl.glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
        drawGrid(visRenderer, numy, numz, ySize, zSize, this.yLabelSkip, this.zLabelSkip);
        gl.glPopMatrix();

        applyMajorColor(visRenderer);

        // Draw the Y axis strings
        float increment = ySize / (numy - 1);
        gl.glPushMatrix();

        if (orientation == Orientation.NW) {
            gl.glTranslatef(xSize, 0.0f, 0.0f);
            gl.glRotatef(180.0f, 0.0f, 0.0f, 1.0f);
            increment = -increment;
        } else if (orientation == Orientation.NE) {
            gl.glTranslatef(xSize, 0, 0);
        } else if (orientation == Orientation.SE) {
            // we're ok
        } else if (orientation == Orientation.SW) {
            gl.glRotatef(180.0f, 0.0f, 0.0f, 1.0f);
            increment = -increment;
        }

        if (onEdge)
            gl.glTranslatef(0.0f, increment, 0.0f);
        drawLabels(visRenderer, ylabel, yStrings, increment, yLabelSkip, yTickSkip, orientation == Orientation.NE
                || orientation == Orientation.SW, selectedRow);
        gl.glPopMatrix();

        // Draw the X axis strings
        gl.glPushMatrix();
        increment = xSize / (numx - 1);

        if (orientation == Orientation.NW) {
            gl.glTranslatef(0.0f, ySize, 0.0f);
            gl.glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
            increment = -increment;
        } else if (orientation == Orientation.NE) {
            gl.glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
            increment = -increment;
        } else if (orientation == Orientation.SE) {
            gl.glRotatef(270.0f, 0.0f, 0.0f, 1.0f);
        } else if (orientation == Orientation.SW) {
            gl.glTranslatef(0, ySize, 0);
            gl.glRotatef(270.0f, 0.0f, 0.0f, 1.0f);
        }

        if (onEdge)
            gl.glTranslatef(0.0f, increment, 0.0f);
        drawLabels(visRenderer, xlabel, xStrings, increment, xLabelSkip, xTickSkip, orientation == Orientation.NW
                || orientation == Orientation.SE, selectedCol);
        gl.glPopMatrix();

        // Draw the Z axis strings
        gl.glPushMatrix();
        increment = zSize / (numz - 1);

        if (orientation == Orientation.NW) {
            gl.glTranslatef(0.0f, ySize, 0.0f);
            gl.glRotatef(45.0f, 0.0f, 0.0f, 1.0f);
        } else if (orientation == Orientation.NE) {
            gl.glTranslatef(xSize, ySize, 0.0f);
            gl.glRotatef(-45.0f, 0.0f, 0.0f, 1.0f);
        } else if (orientation == Orientation.SW) {
            gl.glTranslatef(xSize, ySize, 0.0f);
            gl.glRotatef(135.0f, 0.0f, 0.0f, 1.0f);
        } else if (orientation == Orientation.SE) {
            gl.glTranslatef(0.0f, ySize, 0.0f);
            gl.glRotatef(-135.0f, 0.0f, 0.0f, 1.0f);
        }
        gl.glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
        gl.glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
        drawLabels(visRenderer, zlabel, zStrings, increment, zLabelSkip, zTickSkip, orientation == Orientation.NW
                || orientation == Orientation.NE, -1);
        gl.glPopMatrix();

        // Again, on the other size
        gl.glPushMatrix();
        increment = zSize / (numz - 1);

        if (orientation == Orientation.NW) {
            gl.glTranslatef(xSize, 0.0f, 0.0f);
            gl.glRotatef(45.0f, 0.0f, 0.0f, 1.0f);
        } else if (orientation == Orientation.NE) {
            gl.glRotatef(-45.0f, 0.0f, 0.0f, 1.0f);
        } else if (orientation == Orientation.SW) {
            gl.glRotatef(135.0f, 0.0f, 0.0f, 1.0f);
        } else if (orientation == Orientation.SE) {
            gl.glTranslatef(xSize, 0.0f, 0.0f);
            gl.glRotatef(-135.0f, 0.0f, 0.0f, 1.0f);
        }
        gl.glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
        gl.glRotatef(90.0f, 1.0f, 0.0f, 0.0f);

        drawLabels(visRenderer, zlabel, zStrings, increment, zLabelSkip, zTickSkip, orientation == Orientation.SE
                || orientation == Orientation.SW, -1);
        gl.glPopMatrix();

        gl.glEnable(GL.GL_LIGHTING);

    }

    private void drawLabels(VisRenderer visRenderer, String label, List strings, float increment, int labelSkip, int tickSkip,
            boolean leftJustified, int selected) {
        // Draw the strings for an axis

        GL gl = visRenderer.getGLAutoDrawable().getGL();
        double maxPoint = 0;

        applyTextColor(visRenderer);

        for (int i = 0; i < strings.size(); i++) {

            if (i % (labelSkip + 1) == 0) {

                if (i == selected) {
                    VisTools.glSetColor(gl, highlightColor);
                }

                String string = (String) strings.get(i);
                double width = glut.glutStrokeLengthf(font, string);

                gl.glPushMatrix();

                gl.glBegin(GL.GL_LINES);
                if (leftJustified) {
                    gl.glVertex3f(0.0f, 0.0f, 0.0f);
                    gl.glVertex3f(1.0f, 0.0f, 0.0f);
                } else {
                    gl.glVertex3f(0.0f, 0.0f, 0.0f);
                    gl.glVertex3f(-1.0f, 0.0f, 0.0f);

                }
                gl.glEnd();


                
                if (leftJustified) {
                    gl.glTranslatef(1.5f, 0.0f, 0.0f);
                    gl.glScalef(stringSize / fontScale, stringSize / fontScale, stringSize / fontScale);
                } else {
                    gl.glTranslatef(-1.5f, 0.0f, 0.0f);
                    gl.glScalef(stringSize / fontScale, stringSize / fontScale, stringSize / fontScale);
                    gl.glTranslated(-width, 0.0, 0.0);
                }

                // keep track of the widest width to determine where to draw the label for this axis
                maxPoint = Math.max(maxPoint, width);

                // the text seems to be about 100 in height, so move to the middle
                gl.glTranslatef(0.0f, -50.0f, 0.0f);

                // Render The Text
                for (int c = 0; c < string.length(); c++) {
                    char ch = string.charAt(c);
                    glut.glutStrokeCharacter(font, ch);
                }
                gl.glPopMatrix();
            } else {
                if (i % (tickSkip + 1) == 0) {
                    gl.glBegin(GL.GL_LINES);
                    if (leftJustified) {
                        gl.glVertex3f(0.0f, 0.0f, 0.0f);
                        gl.glVertex3f(0.5f, 0.0f, 0.0f);
                    } else {
                        gl.glVertex3f(0.0f, 0.0f, 0.0f);
                        gl.glVertex3f(-0.5f, 0.0f, 0.0f);
                    }
                    gl.glEnd();
                }
            }
            gl.glTranslatef(0.0f, increment, 0.0f); // move 'increment' in the y direction for the next string

            if (i == selected) { // set it back to the regular text color if this one was selected
                applyTextColor(visRenderer);
            }

        }

        // maxPoint is the extent of the labels
        maxPoint = maxPoint * 0.003f;

        // Now draw the axis labels
        gl.glPushMatrix();
        if (leftJustified) {
            gl.glTranslated(maxPoint + 3.0f, -increment * (strings.size() + 1) / 2, 0);
            gl.glRotatef(90, 0, 0, 1);
            gl.glScalef(labelSize / fontScale, labelSize / fontScale, labelSize / fontScale);


            StringTokenizer st = new StringTokenizer(label, "\n");
            while (st.hasMoreTokens()) {
                String line = st.nextToken();
                float width = glut.glutStrokeLength(font, line);

                gl.glTranslatef(-width / 2, 0, 0);

                // Render The Text
                for (int c = 0; c < line.length(); c++) {
                    char ch = line.charAt(c);
                    glut.glutStrokeCharacter(font, ch);
                }

                gl.glTranslatef(-width / 2, 0, 0);
                gl.glTranslated(0, -250.0f, 0);
            }
        } else {
            gl.glTranslated(-(maxPoint + 3.0f), -increment * (strings.size() + 1) / 2, 0);
            gl.glRotatef(-90, 0, 0, 1);
            gl.glScalef(labelSize / fontScale, labelSize / fontScale, labelSize / fontScale);
           

            StringTokenizer st = new StringTokenizer(label, "\n");
            while (st.hasMoreTokens()) {
                String line = st.nextToken();
                float width = glut.glutStrokeLength(font, line);

                gl.glTranslatef(-width / 2, 0, 0);

                // Render The Text
                for (int c = 0; c < line.length(); c++) {
                    char ch = line.charAt(c);
                    glut.glutStrokeCharacter(font, ch);
                }

                gl.glTranslatef(-width / 2, 0, 0);
                gl.glTranslated(0, -250.0f, 0);
            }

        }
        gl.glPopMatrix();

    }

    private void renderStrokeString(GL gl, int font, String string) {
        // Center Our Text On The Screen
        float width = glut.glutStrokeLength(font, string);
        gl.glTranslatef(-width / 2f, 0, 0);
        // Render The Text
        for (int i = 0; i < string.length(); i++) {
            char c = string.charAt(i);
            glut.glutStrokeCharacter(font, c);
        }
    }

    /**
     * Retrieves the autoSkip setting.
     * @see #setAutoSkip
     * @return Returns the autoSkip.
     */
    public boolean getAutoSkip() {
        return autoSkip;
    }

    /**
     * Sets whether or not axis labels are automatically skipped or not.  If set, the Axes will 
     * skip some labels such that the text won't overlap if the plot is not large enough. 
     * @param autoSkip The autoSkip to set.
     */
    public void setAutoSkip(boolean autoSkip) {
        this.autoSkip = autoSkip;
    }

    /**
     * Returns the major color of the Axes.
     * @return the major color of the Axes.
     */
    public Color getMajorColor() {
        return majorColor;
    }

    /**
     * Sets the major color of the Axes.  The major color is used for the axis 
     * outline, not for the inner grid.  The default is <tt>Color.white</tt>.
     * @param majorColor color to use.
     */
    public void setMajorColor(Color majorColor) {
        this.majorColor = majorColor;
        this.dirty = true;
    }

    /**
     * Returns the minor color of the Axes.
     * @return the minor color of the Axes.
     */
    public Color getMinorColor() {
        return minorColor;
    }

    /**
     * Sets the minor color of the Axes.  The minor color is used for the axis 
     * inner grid.  The default is <tt>Color(0.5, 0.5, 0.5)</tt>.
     * @param minorColor the color to use.
     */
    public void setMinorColor(Color minorColor) {
        this.minorColor = minorColor;
        this.dirty = true;
    }

    /**
     * Returns the text color.
     * @return the text color.
     */
    public Color getTextColor() {
        return textColor;
    }

    /**
     * Sets the text color.  The text color is used for the labels.  
     * The default is <tt>Color.white</tt> 
     * @param textColor the color to use.
     */
    public void setTextColor(Color textColor) {
        this.textColor = textColor;
        this.dirty = true;
    }

    public float getFontScale() {
        return fontScale;
    }

    public void setFontScale(float fontScale) {
        this.fontScale = fontScale;
        this.dirty = true;
    }
}
