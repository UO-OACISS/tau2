package edu.uoregon.tau.paraprof.vis;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.StringTokenizer;
import java.util.Vector;

import javax.swing.*;

import net.java.games.jogl.GL;
import net.java.games.jogl.GLDrawable;
import net.java.games.jogl.util.GLUT;
import edu.uoregon.tau.paraprof.ParaProfUtils;

/**
 * Draws axes with labels
 * 
 * @author amorris
 */
public class Axes implements Shape {

    private GLUT glut = new GLUT();

    private float xSize, ySize, zSize;

    private String xlabel, ylabel, zlabel;

    private int xTickSkip, yTickSkip, zTickSkip;
    private int xLabelSkip, yLabelSkip, zLabelSkip;
    private Vector xStrings;
    private Vector yStrings;
    private Vector zStrings;

    private boolean autoSkip = true;
    private boolean onEdge;
    private boolean dirty = true;
    private Orientation orientation = Orientation.NW;

    private int font = GLUT.STROKE_MONO_ROMAN;

    
    private float stringSize = 3;
    private float labelSize = 8;
    
    private boolean enabled = true;
    private Color highlightColor = new Color(1, 0, 0);

    private int displayList;

    private int selectedRow = -1;
    private int selectedCol = -1;


    /**
     * Type safe enum for axes orientation
     */
    public static class Orientation {

        private final String name;

        private Orientation(String name) {
            this.name = name;
        }

        public String toString() {
            return name;
        }

        public static final Orientation NW = new Orientation("NW");
        public static final Orientation NE = new Orientation("NE");
        public static final Orientation SW = new Orientation("SW");
        public static final Orientation SE = new Orientation("SE");

    }

    /**
     * Constructor for the Axes
     */
    public Axes() {
    }

    /**
     * Returns whether or not the axes are enabled (shown)
     * @return enabled
     */
    public boolean getEnabled() {
        return enabled;
    }

    /**
     * Enables or disables the axes
     * @param enabled - the boolean to be set
     */
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        this.dirty = true;
    }

    /**
     * Sets the string labels
     * @param xlabel - label for the x axis
     * @param ylabel - label for the y axis
     * @param zlabel - label for the z axis
     * @param xStrings - vector of strings for the x axis
     * @param yStrings - vector of strings for the y axis
     * @param zStrings - vector of strings for the z axis
     */
    public void setStrings(String xlabel, String ylabel, String zlabel, Vector xStrings, Vector yStrings,
            Vector zStrings) {
        this.xlabel = xlabel;
        this.ylabel = ylabel;
        this.zlabel = zlabel;
        this.xStrings = xStrings;
        this.yStrings = yStrings;
        this.zStrings = zStrings;

        if (this.xStrings == null)
            this.xStrings = new Vector();

        if (this.yStrings == null)
            this.yStrings = new Vector();

        if (this.zStrings == null)
            this.zStrings = new Vector();

        setAutoTickSkip();
        this.dirty = true;

    }

    /**
     * Sets whether or not the data points land on the intersection of two lines of the axes or inbetween
     * @param offset - whether or not the axes are offset
     */
    public void setOnEdge(boolean onEdge) {
        this.onEdge = onEdge;
        this.dirty = true;
    }

    public boolean getOnEdge() {
        return this.onEdge;
    }
    
    
    /**
     * Sets the size of the axes.  Note that this is usually called by the plot.
     * @param xSize - size in the x direction
     * @param ySize - size in the y direction
     * @param zSize - size in the z direction
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
     * @param color - the color to use
     */
    public void setHighlightColor(Color color) {
        highlightColor = color;
    }

    /**
     * @return Returns the orientation.
     */
    public Orientation getOrientation() {
        return orientation;
    }

    /**
     * @param orientation - The orientation to set.
     */
    public void setOrientation(Orientation orientation) {
        this.orientation = orientation;
        this.dirty = true;
    }

    /**
     * Creates a Swing JPanel with controls for the axes.  These controls will 
     * change the state of the axes and automatically call visRenderer.redraw()
     * 
     * When getControlPanel() is called, the controls will represent the current
     * values for the axes, but currently, they will not stay in sync if the values
     * are changed using the public methods.  For example, if you call "setEnabled(false)"
     * The JCheckBox will not be set to unchecked.  This functionality could be added if
     * requested.
     * 
     * @return JPanel - the control panel for this component
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
                    Axes.this.setEnabled(enabledCheckBox.isSelected());
                    visRenderer.redraw();

                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
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
                    ParaProfUtils.handleException(e);
                }
            }
        };

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
        ParaProfUtils.addCompItem(axesPanel, enabledCheckBox, gbc, 0, 0, 1, 3);
        gbc.anchor = GridBagConstraints.CENTER;
        gbc.fill = GridBagConstraints.NONE;
        ParaProfUtils.addCompItem(axesPanel, new JLabel("Orientation"), gbc, 1, 0, 2, 1);

        gbc.anchor = GridBagConstraints.EAST;
        gbc.fill = GridBagConstraints.NONE;

        ParaProfUtils.addCompItem(axesPanel, nw, gbc, 1, 1, 1, 1);
        ParaProfUtils.addCompItem(axesPanel, sw, gbc, 1, 2, 1, 1);

        gbc.anchor = GridBagConstraints.WEST;
        ParaProfUtils.addCompItem(axesPanel, ne, gbc, 2, 1, 1, 1);
        ParaProfUtils.addCompItem(axesPanel, se, gbc, 2, 2, 1, 1);

        return axesPanel;
    }

    public int getSelectedCol() {
        return selectedCol;
    }

    public void setSelectedCol(int selectedCol) {
        this.selectedCol = selectedCol;
        dirty = true;
    }

    public int getSelectedRow() {
        return selectedRow;
    }

    public void setSelectedRow(int selectedRow) {
        this.selectedRow = selectedRow;
        dirty = true;
    }

    public void render(GLDrawable glDrawable) {
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

    private void setTickSkipping(int xTicSkip, int yTicSkip, int zTicSkip, int xLabelSkip, int yLabelSkip,
            int zLabelSkip) {

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

    private void drawGrid(GL gl, int numx, int numy, float xSize, float ySize, int xLabelSkip, int yLabelSkip) {
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
                    gl.glColor3f(0.9f, 0.9f, 0.9f);
                }
                float position = (float) x / (numx - 1) * xSize;
                gl.glVertex3f(position, 0, 0);
                gl.glVertex3f(position, ySize, 0);
                if (x == 0 || x == numx - 1) {
                    gl.glColor3f(0.5f, 0.5f, 0.5f);
                }
            }
        }
        for (int y = 0; y < numy; y++) {
            if ((y - yOffset) % (yLabelSkip + 1) == 0 || y == numy - 1) {
                if (y == 0 || y == numy - 1) {
                    gl.glColor3f(0.9f, 0.9f, 0.9f);
                }
                float pos = (float) y / (numy - 1) * ySize;
                gl.glVertex3f(0, pos, 0);
                gl.glVertex3f(xSize, pos, 0);
                if (y == 0 || y == numy - 1) {
                    gl.glColor3f(0.5f, 0.5f, 0.5f);
                }
            }
        }
        gl.glEnd();

    }

    private void privateRender(GLDrawable glDrawable) {
        if (!enabled)
            return;

        GL gl = glDrawable.getGL();

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

        gl.glColor3f(0.5f, 0.5f, 0.5f);

        //        gl.glEnable(GL.GL_LINE_SMOOTH);
        //        gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
        //        gl.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST);
        //        gl.glEnable(GL.GL_BLEND);

        gl.glLineWidth(1.0f);
        gl.glDisable(GL.GL_LINE_SMOOTH);

        // grid for x-y plane
        drawGrid(gl, numx, numy, xSize, ySize, this.xLabelSkip, this.yLabelSkip);

        // grid for x-z plane
        gl.glPushMatrix();
        if (orientation == Orientation.NE || orientation == Orientation.SE)
            gl.glTranslatef(0, ySize, 0);
        gl.glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
        drawGrid(gl, numx, numz, xSize, zSize, this.xLabelSkip, this.zLabelSkip);
        gl.glPopMatrix();

        // grid for y-z plane
        gl.glPushMatrix();
        if (orientation == Orientation.SE || orientation == Orientation.SW)
            gl.glTranslatef(xSize, 0, 0);
        gl.glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
        gl.glRotatef(90.0f, 1.0f, 0.0f, 0.0f);
        drawGrid(gl, numy, numz, ySize, zSize, this.yLabelSkip, this.zLabelSkip);
        gl.glPopMatrix();

        gl.glColor3f(0.9f, 0.9f, 0.9f);

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
        drawLabels(gl, ylabel, yStrings, increment, yLabelSkip, yTickSkip, orientation == Orientation.NE
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
        drawLabels(gl, xlabel, xStrings, increment, xLabelSkip, xTickSkip, orientation == Orientation.NW
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
        drawLabels(gl, zlabel, zStrings, increment, zLabelSkip, zTickSkip, orientation == Orientation.NW
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

        drawLabels(gl, zlabel, zStrings, increment, zLabelSkip, zTickSkip, orientation == Orientation.SE
                || orientation == Orientation.SW, -1);
        gl.glPopMatrix();

        gl.glEnable(GL.GL_LIGHTING);

    }

    private void drawLabels(GL gl, String label, Vector strings, float increment, int labelSkip, int tickSkip,
            boolean leftJustified, int selected) {
        // Draw the strings for an axis

        float maxPoint = 0;

        for (int i = 0; i < strings.size(); i++) {

            if (i % (labelSkip + 1) == 0) {

                if (i == selected) {
                    gl.glColor3f(highlightColor.getRed() / 255.0f, highlightColor.getGreen() / 255.0f,
                            highlightColor.getBlue() / 255.0f);
                }

                String string = (String) strings.get(i);
                float width = glut.glutStrokeLength(font, string);

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
                    gl.glScalef(stringSize / 1000, stringSize / 1000, stringSize / 1000);
                    maxPoint = Math.max(maxPoint, width);
                } else {
                    gl.glTranslatef(-1.5f, 0.0f, 0.0f);
                    gl.glScalef(stringSize / 1000, stringSize / 1000, stringSize / 1000);
                    gl.glTranslatef(-width, 0.0f, 0.0f);
                    maxPoint = Math.max(maxPoint, width);
                }

                // the text seems to be about 100 in height, so move to the middle
                gl.glTranslatef(0.0f, -50.0f, 0.0f);

                // Render The Text
                for (int c = 0; c < string.length(); c++) {
                    char ch = string.charAt(c);
                    glut.glutStrokeCharacter(gl, font, ch);
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

            if (i == selected) {
                gl.glColor3f(1, 1, 1);
            }

        }

        // maxPoint is the extent of the labels
        maxPoint = maxPoint * 0.003f;

        // Now draw the axis labels
        gl.glPushMatrix();
        if (leftJustified) {
            gl.glTranslated(maxPoint + 3.0f, -increment * (strings.size() + 1) / 2, 0);
            gl.glRotatef(90, 0, 0, 1);
            gl.glScalef(labelSize / 1000, labelSize / 1000, labelSize / 1000);

            StringTokenizer st = new StringTokenizer(label, "\n");
            while (st.hasMoreTokens()) {
                String line = st.nextToken();
                float width = glut.glutStrokeLength(font, line);

                gl.glTranslatef(-width / 2, 0, 0);
                gl.glColor3f(1, 1, 1);

                // Render The Text
                for (int c = 0; c < line.length(); c++) {
                    char ch = line.charAt(c);
                    glut.glutStrokeCharacter(gl, font, ch);
                }

                gl.glTranslatef(-width / 2, 0, 0);
                gl.glTranslated(0, -250.0f, 0);
            }
        } else {
            gl.glTranslated(-(maxPoint + 3.0f), -increment * (strings.size() + 1) / 2, 0);
            gl.glRotatef(-90, 0, 0, 1);

            gl.glScalef(labelSize / 1000, labelSize / 1000, labelSize / 1000);

            StringTokenizer st = new StringTokenizer(label, "\n");
            while (st.hasMoreTokens()) {
                String line = st.nextToken();
                float width = glut.glutStrokeLength(font, line);

                gl.glTranslatef(-width / 2, 0, 0);
                gl.glColor3f(1, 1, 1);

                // Render The Text
                for (int c = 0; c < line.length(); c++) {
                    char ch = line.charAt(c);
                    glut.glutStrokeCharacter(gl, font, ch);
                }

                gl.glTranslatef(-width / 2, 0, 0);
                gl.glTranslated(0, -250.0f, 0);
            }

        }
        gl.glPopMatrix();

        gl.glColor3f(1, 1, 1);

    }

    private void renderStrokeString(GL gl, int font, String string) {
        // Center Our Text On The Screen
        float width = glut.glutStrokeLength(font, string);
        gl.glTranslatef(-width / 2f, 0, 0);
        // Render The Text
        for (int i = 0; i < string.length(); i++) {
            char c = string.charAt(i);
            glut.glutStrokeCharacter(gl, font, c);
        }
    }

    /**
     * @return Returns the autoSkip.
     */
    public boolean getAutoSkip() {
        return autoSkip;
    }

    /**
     * @param autoSkip The autoSkip to set.
     */
    public void setAutoSkip(boolean autoSkip) {
        this.autoSkip = autoSkip;
    }

}
