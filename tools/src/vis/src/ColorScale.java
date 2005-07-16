/*
 * ColorScale.java
 *
 * Copyright 2005                                                 
 * Department of Computer and Information Science, University of Oregon
 */
package edu.uoregon.tau.vis;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.*;

import javax.swing.*;

import net.java.games.jogl.GL;
import net.java.games.jogl.GLDrawable;
import net.java.games.jogl.util.GLUT;

/**
 * Draws a colorscale and also provides services to other vis components.
 * 
 * The colorscale is a drawable Shape, but is also used by other components
 * allowing them to query values (0..1) and get colors in the current
 * color set. 
 *    
 * <P>CVS $Id: ColorScale.java,v 1.2 2005/07/16 00:21:07 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.2 $
 */

/* TODO: Provide control over font size perhaps? */
public class ColorScale extends Observable implements Shape {

    /**
     * Represents a set of colors for a ColorScale
     * 
     * @author Alan Morris
     *
     */
    public static class ColorSet {

        private final String name;
        public final double colorsR[];
        public final double colorsG[];
        public final double colorsB[];
        
        private ColorSet(String name, double[] colorsR, double[] colorsG, double[] colorsB) {
            this.name = name;
            this.colorsR = colorsR;
            this.colorsG = colorsG;
            this.colorsB = colorsB;
        }

        public String toString() {
            return name;
        }

        public static final ColorSet RAINBOW = new ColorSet("Rainbow", new double[] { 0, 0, 0, 1, 1 },
                new double[] { 0, 1, 1, 1, 0 }, new double[] { 1, 1, 0, 0, 0 });

        public static final ColorSet GRAYSCALE = new ColorSet("Grayscale", new double[] { 0, 1, },
                new double[] { 0, 1, }, new double[] { 0, 1, });

        public static final ColorSet BLUE_RED = new ColorSet("Blue-Red", new double[] { 0, 1 }, new double[] {
                0, 0 }, new double[] { 1, 0 });

        public static final ColorSet BLUE_WHITE_RED = new ColorSet("Blue-White-Red", new double[] { 0, 1, 1, },
                new double[] { 0, 1, 0, }, new double[] { 1, 1, 0, });

        public static ColorSet[] VALUES = { RAINBOW, GRAYSCALE, BLUE_RED, BLUE_WHITE_RED };

    }

    private Color textColor = Color.white;
    private int font = GLUT.STROKE_MONO_ROMAN;
    private GLUT glut = new GLUT();
    private boolean dirty = true;
    private boolean enabled = true;
    private ColorSet colorSet = ColorSet.RAINBOW;
    private String lowString, highString, label;
    private double fontScale = 0.10;
    private int displayList;

    public ColorScale() {
    }

    /**
     * Sets the strings lables for this ColorScale
     * @param low the low end of the scale
     * @param high the high end of the scale
     * @param label the label for the scale
     */
    public void setStrings(String low, String high, String label) {
        this.lowString = low;
        this.highString = high;
        this.label = label;
    }

    /**
     * Retrieves a color in the colorscale based on a ratio (0..1) (blue..red)
     * 
     * @param	ratio a value between 0 and 1
     * @return	a color from the scale blue..red (with some other colors inbetween)
     */
    public Color getColor(float ratio) {

        float origRatio = ratio;
        double r, g, b;

        int section = 0;

        int numSections = colorSet.colorsR.length - 1;

        float limit = 1.0f / numSections;

        section = 0;
        ratio = origRatio * numSections;
        while (origRatio > limit) {
            section++;
            ratio = (origRatio - limit) * numSections;
            limit += 1.0f / numSections;
        }

        r = colorSet.colorsR[section] + ratio * (colorSet.colorsR[section + 1] - colorSet.colorsR[section]);
        g = colorSet.colorsG[section] + ratio * (colorSet.colorsG[section + 1] - colorSet.colorsG[section]);
        b = colorSet.colorsB[section] + ratio * (colorSet.colorsB[section + 1] - colorSet.colorsB[section]);

        // clamp colors (just in case)
        r = Math.min(r, 1);
        g = Math.min(g, 1);
        b = Math.min(b, 1);

        r = Math.max(r, 0);
        g = Math.max(g, 0);
        b = Math.max(b, 0);

        return new Color((float) r, (float) g, (float) b);
    }

    /**
     * Creates a Swing JPanel with controls for this object.  These controls will 
     * change the state of the axes and automatically call visRenderer.redraw()
     * 
     * When getControlPanel() is called, the controls will represent the current
     * values for the axes, but currently, they will not stay in sync if the values
     * are changed using the public methods.  For example, if you call "setEnabled(false)"
     * The JCheckBox will not be set to unchecked.  This functionality could be added if
     * requested.
     * 
     * @param visRenderer The associated VisRenderer
     * @return the control panel for this component
     */
    public JPanel getControlPanel(final VisRenderer visRenderer) {
        JPanel controlPanel = new JPanel();

        controlPanel.setLayout(new GridBagLayout());
        controlPanel.setBorder(BorderFactory.createLoweredBevelBorder());

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.NONE;
        gbc.weightx = 0.2;
        gbc.weighty = 0.2;

        JCheckBox enabledCheckBox = new JCheckBox("Show ColorScale", true);
        VisTools.addCompItem(controlPanel, enabledCheckBox, gbc, 0, 0, 1, 1);
        enabledCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                ColorScale.this.enabled = ((JCheckBox) evt.getSource()).isSelected();
                ColorScale.this.dirty = true;
                visRenderer.redraw();
            }
        });

        //        VisTools.addCompItem(controlPanel, new JLabel("ColorScale Selection"), gbc, 0, 1, 1, 1);
        ButtonGroup group = new ButtonGroup();

        final Map colorScaleMap = new HashMap();

        int x = 0;
        int y = 1;
        for (int i = 0; i < ColorSet.VALUES.length; i++) {
            ColorSet colorSet = ColorSet.VALUES[i];
            JRadioButton jrb = new JRadioButton(colorSet.toString(), this.colorSet == colorSet);
            group.add(jrb);
            colorScaleMap.put(jrb, colorSet);
            jrb.addActionListener(new ActionListener() {
                public void actionPerformed(ActionEvent evt) {
                    ColorScale.this.colorSet = (ColorSet) colorScaleMap.get(evt.getSource());
                    ColorScale.this.dirty = true;

                    // notify anyone watching the colorScale
                    ColorScale.this.setChanged();
                    ColorScale.this.notifyObservers();

                    visRenderer.redraw();
                }
            });

            VisTools.addCompItem(controlPanel, jrb, gbc, x, y, 1, 1);

            if (x == 0) {
                x++;
            } else {
                x = 0;
                y++;
            }
        }

        return controlPanel;
    }

    private void drawText(GL gl, double x, double y, String text) {
        if (text == null)
            return;

        gl.glPushMatrix();

        StringTokenizer st = new StringTokenizer(text, "\n");
        while (st.hasMoreTokens()) {
            double thisX = x;
            gl.glPushMatrix();

            String line = st.nextToken();
            float width = glut.glutStrokeLength(font, line);

            double fontScale = 0.10;

            thisX -= (width * fontScale / 2);
            thisX = Math.max(thisX, -20);

            gl.glTranslated(thisX, y, 0);
            gl.glScaled(fontScale, fontScale, fontScale);

            // Render The Text
            for (int c = 0; c < line.length(); c++) {
                char ch = line.charAt(c);
                glut.glutStrokeCharacter(gl, font, ch);
            }

            gl.glTranslatef(-width / 2, 0, 0);
            gl.glTranslated(0, -250.0f, 0);

            gl.glPopMatrix();

            gl.glTranslated(0, -25.0f, 0);

        }

        gl.glPopMatrix();
    }

    
    /**
     * Renders to the given VisRenderer
     * @param visRenderer the associated control panel
     */
    public void render(VisRenderer visRenderer) {
        GLDrawable glDrawable = visRenderer.getGLDrawable();

        if (!enabled)
            return;

        dirty = true;
        GL gl = glDrawable.getGL();

        if (dirty || displayList == 0) {
            displayList = gl.glGenLists(1);
            gl.glNewList(displayList, GL.GL_COMPILE);
            privateRender(visRenderer);
            gl.glEndList();
            dirty = false;
        }
        gl.glCallList(displayList);
    }

    private void privateRender(VisRenderer visRenderer) {
        GLDrawable glDrawable = visRenderer.getGLDrawable();

        GL gl = glDrawable.getGL();
        if (enabled == false)
            return;

        int width = (int) glDrawable.getSize().getWidth();
        int height = (int) glDrawable.getSize().getHeight();
        gl.glMatrixMode(GL.GL_PROJECTION);
        gl.glPushMatrix();
        gl.glLoadIdentity();
        gl.glOrtho(0, width, 0, height, -1.0f, 1.0f);

        gl.glMatrixMode(GL.GL_MODELVIEW);
        gl.glPushMatrix();

        gl.glLoadIdentity();

        gl.glDisable(GL.GL_LIGHTING);
        gl.glDisable(GL.GL_DEPTH_TEST);
        gl.glDisable(GL.GL_LINE_SMOOTH);
        gl.glDisable(GL.GL_BLEND);
        gl.glLineWidth(1.0f);

        int pixelHeight = Math.min(300, height - 60);
        int pixelWidth = 25;
        gl.glTranslated(25, (height / 2) - (pixelHeight / 2), 0);

        gl.glShadeModel(GL.GL_SMOOTH);

        gl.glFrontFace(GL.GL_CW);

        gl.glBegin(GL.GL_QUADS);

        int nBlocks = 10;
        float increment = pixelHeight / 10.0f;
        for (float i = 0; i < nBlocks; i++) {

            float ratio1 = (float) i / (float) nBlocks;
            Color c1 = getColor(ratio1);

            float ratio2 = ((float) (i + 1)) / nBlocks;
            Color c2 = getColor(ratio2);

            VisTools.glSetColor(gl, c1);
            gl.glVertex3f(pixelWidth, i * increment, 0);
            gl.glVertex3f(0, i * increment, 0);
            VisTools.glSetColor(gl, c2);
            gl.glVertex3f(0, (i + 1) * increment, 0);
            gl.glVertex3f(pixelWidth, (i + 1) * increment, 0);

        }
        gl.glEnd();

        if (visRenderer.getReverseVideo()) {
            VisTools.glSetColor(gl, VisTools.invert(textColor));
        } else {
            VisTools.glSetColor(gl, textColor);
        }

        drawText(gl, pixelWidth / 2, pixelHeight + 10, highString);
        drawText(gl, pixelWidth / 2, -15.0, lowString);

        gl.glRotatef(90, 0, 0, 1);

        drawText(gl, pixelHeight / 2, -pixelWidth - 15, label);

        gl.glPopMatrix();

        gl.glMatrixMode(GL.GL_PROJECTION);
        gl.glPopMatrix();
        gl.glMatrixMode(GL.GL_MODELVIEW);

        gl.glEnable(GL.GL_LIGHTING);
        gl.glEnable(GL.GL_DEPTH_TEST);

    }

}
