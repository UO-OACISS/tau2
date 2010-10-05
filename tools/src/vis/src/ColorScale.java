/*
 * ColorScale.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;
import java.util.Observable;
import java.util.StringTokenizer;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.ButtonGroup;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import com.sun.opengl.util.GLUT;

/**
 * Draws a colorscale and also provides services to other vis components.
 * 
 * The colorscale is a drawable Shape, but is also used by other components
 * allowing them to query values (0..1) and get colors in the current
 * color set. 
 *    
 * <P>CVS $Id: ColorScale.java,v 1.12 2009/08/21 00:21:23 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.12 $
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

        public static final ColorSet RAINBOW = new ColorSet("Rainbow", new double[] { 0, 0, 0, 1, 1 }, new double[] { 0, 1, 1, 1,
                0 }, new double[] { 1, 1, 0, 0, 0 });

        public static final ColorSet GRAYSCALE = new ColorSet("Grayscale", new double[] { 0, 1, }, new double[] { 0, 1, },
                new double[] { 0, 1, });

        public static final ColorSet INVERSE_GRAYSCALE = new ColorSet("Inverse Grayscale", new double[] { 1, 0, }, new double[] {
                1, 0, }, new double[] { 1, 0, });

        public static final ColorSet BLUE_RED = new ColorSet("Blue-Red", new double[] { 0, 1 }, new double[] { 0, 0 },
                new double[] { 1, 0 });

        public static final ColorSet BLUE_WHITE_RED = new ColorSet("Blue-White-Red", new double[] { 0, 1, 1, }, new double[] { 0,
                1, 0, }, new double[] { 1, 1, 0, });

        public static ColorSet[] VALUES = { RAINBOW, GRAYSCALE, INVERSE_GRAYSCALE, BLUE_RED, BLUE_WHITE_RED };

    }

    
    public class ColorPan extends JPanel {
    	  /**
		 * 
		 */
		private static final long serialVersionUID = -3355829427807565800L;
		BufferedImage image;

		
		ColorPan(){
			super();
			Dimension d = new Dimension();
			d.setSize(50, 50);
			this.setSize(d);
			this.setPreferredSize(d);
			this.setMinimumSize(d);
			
			
			this.addComponentListener(new ComponentListener() 
			{  

					public void componentResized(ComponentEvent e) {
						initialize();
						
					}

					public void componentMoved(ComponentEvent e) {
						
					}

					public void componentShown(ComponentEvent e) {
						
					}

					public void componentHidden(ComponentEvent e) {
						
					}
			});
			
		}
		
    	  public void initialize() {
    	    int width = getSize().width;
    	    int height = getSize().height;
    	    if(height<=0||width<=0)
    	    	return;
    	    int[] data = new int[width * height];
    	    int index = (width*height)-1;
    	    
    	    int nBlocks=height;
    	    
    	    for(float i = 0; i< nBlocks; i++){
    	    	Color c1 = getColor((float)(i)/(float)nBlocks);
    	    	Color c2 = getColor((float)(i+1)/nBlocks);
    	    	
    	    for (int h = 0; h < height/nBlocks; h++) {
    	    	
    	      for (int w = 0; w < width; w++) {
    	    	  Color use=c1;
                  if((h%2==0 && w%2==0)||(h%2!=0 && w%2!=0))
    	    	    use=c2;
    	        data[index--] = (use.getRed() << 16) | (use.getGreen() << 8) | use.getBlue();
    	      }
    	    }
    	    }
    	    image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    	    image.setRGB(0, 0, width, height, data, 0, width);
    	  }

    	  public void paint(Graphics g) {
    	    if (image == null)
    	      initialize();
    	    g.drawImage(image, 0, 0, this);
    	  }

    	}

    
    private Color textColor = Color.white;
    private int font = GLUT.STROKE_MONO_ROMAN;
    private GLUT glut = new GLUT();
    private boolean dirty = true;
    private boolean enabled = false;
    private ColorSet colorSet = ColorSet.RAINBOW;
    private String lowString, highString, label;
    private double fontScale = 0.12;
    private float leftMargin = 25; // distance between left side and colorscale
    private float leftTextMargin = 5; // minimum distance between the left side and text labels (high and low)

    private int displayList;

    private int width = 25;
    private int height = 300;
    private int topBottomMargin = 60;

    // this is to keep track of the old reverseVideo value
    // I need to come up with a better way of tracking the settings
    // we have to know whether to recreate the display list or not
    private boolean oldReverseVideo;
    // need to mark as dirty when size has changed
    private int oldWidth;
    private int oldHeight;

    private boolean oldAntiAlias;

    public ColorScale() {}

    public ColorScale(ColorSet colorSet) {
        super();
        this.colorSet = colorSet;
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
        this.dirty = true;
        if(highLabel!=null)
        highLabel.setText(highString);
        lowLabel.setText("<html>"+lowString+"<br>"+label+"</html>");
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

    
    JLabel highLabel=new JLabel();
    JLabel lowLabel=new JLabel();
    ColorPan cp;
    
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

        int defAnchor = gbc.anchor;
        
        JPanel ckPanel = new JPanel();
        ckPanel.setLayout(new BoxLayout(ckPanel, BoxLayout.Y_AXIS));
        
        gbc.anchor=GridBagConstraints.LAST_LINE_START;
        gbc.fill=GridBagConstraints.HORIZONTAL;
        gbc.weighty=0;
        VisTools.addCompItem(controlPanel, highLabel, gbc, 0, 0, 3, 1);
        gbc.weighty=1;
        cp = new ColorPan();
        ckPanel.add(cp);
        gbc.anchor=defAnchor;
        
        gbc.fill=GridBagConstraints.VERTICAL;
        VisTools.addCompItem(controlPanel, ckPanel, gbc, 0, 1, 1, 5);
        gbc.fill=GridBagConstraints.NONE;
        gbc.weighty=0;
        
        gbc.anchor=GridBagConstraints.FIRST_LINE_START;
        gbc.fill=GridBagConstraints.HORIZONTAL;
        VisTools.addCompItem(controlPanel, lowLabel, gbc, 0, 6, 3, 1);
        gbc.fill=GridBagConstraints.NONE;
        gbc.weighty=0.2;
        gbc.anchor=defAnchor;
        JCheckBox enabledCheckBox = new JCheckBox("Show ColorScale", false);
        VisTools.addCompItem(controlPanel, enabledCheckBox, gbc, 1, 1, 1, 1);
        enabledCheckBox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                ColorScale.this.enabled = ((JCheckBox) evt.getSource()).isSelected();
                ColorScale.this.dirty = true;
                visRenderer.redraw();
            }
        });

        final JSlider fontScaleSlider = new JSlider(0, 100, (int) (getFontScale() * 100));
        fontScaleSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    ColorScale.this.setFontScale(fontScaleSlider.getValue() / (double) 100);
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        VisTools.addCompItem(controlPanel, new JLabel("Font Size"), gbc, 1, 2, 1, 1);
        gbc.fill = GridBagConstraints.BOTH;
        VisTools.addCompItem(controlPanel, fontScaleSlider, gbc, 2, 2, 1, 1);
        gbc.fill = GridBagConstraints.NONE;
        

        //        VisTools.addCompItem(controlPanel, new JLabel("ColorScale Selection"), gbc, 0, 1, 1, 1);
        ButtonGroup group = new ButtonGroup();

        final Map<JRadioButton, ColorSet> colorScaleMap = new HashMap<JRadioButton, ColorSet>();

        int x = 1;
        int y = 3;
        for (int i = 0; i < ColorSet.VALUES.length; i++) {
            ColorSet colorSet = ColorSet.VALUES[i];
            JRadioButton jrb = new JRadioButton(colorSet.toString(), this.colorSet == colorSet);
            group.add(jrb);
            colorScaleMap.put(jrb, colorSet);
            jrb.addActionListener(new ActionListener() {
                public void actionPerformed(ActionEvent evt) {
                    ColorScale.this.colorSet = colorScaleMap.get(evt.getSource());
                    ColorScale.this.dirty = true;

                    // notify anyone watching the colorScale
                    ColorScale.this.setChanged();
                    ColorScale.this.notifyObservers();

                    visRenderer.redraw();
                    cp.initialize();
                    cp.repaint();
                }
            });

            VisTools.addCompItem(controlPanel, jrb, gbc, x, y, 1, 1);

            if (x == 1) {
                x++;
            } else {
                x = 1;
                y++;
            }
        }

        return controlPanel;
    }

    private float getTextWidth(GL gl, String text) {
        if (text == null) {
            return 0;
        }
        StringTokenizer st = new StringTokenizer(text, "\n");
        float maxWidth = 0;
        while (st.hasMoreTokens()) {
            String line = st.nextToken();
            float width = glut.glutStrokeLength(font, line);
            maxWidth = Math.max(maxWidth, width);
        }
        return (float) (maxWidth * fontScale);
    }

    private void drawText(GL gl, double x, double y, String text, boolean growDown) {
        if (text == null)
            return;

        gl.glPushMatrix();

        int numlines = 0;
        StringTokenizer st = new StringTokenizer(text, "\n");
        numlines = st.countTokens();

        double ascent = VisTools.fontAscent * fontScale;
        double descent = VisTools.fontDescent * fontScale;

        float rowHeight = (float) (VisTools.fontHeight * fontScale);

        // 10% extra spacing between rows
        rowHeight = (float) ((ascent + descent) * 1.10);

        if (growDown) {
            y = y - ascent - 5;
        } else {
            y = y + (rowHeight * (numlines - 1)) + descent + 3;
        }

        while (st.hasMoreTokens()) {
            String line = st.nextToken();
            double startX = x;
            gl.glPushMatrix();

            float width = glut.glutStrokeLength(font, line);

            startX -= (width * fontScale / 2);
            //            thisX = Math.max(thisX, -20);

            //            gl.glBegin(GL.GL_LINES);
            //            gl.glVertex2f(0,0);
            //            gl.glVertex2f((float)x,(float)y);
            //            gl.glVertex2f((float)x,(float)y);
            //            gl.glVertex2f((float)startX,(float)y);
            //            gl.glEnd();
            //            
            gl.glTranslated(startX, y, 0);

            //            // 112 seems to be the actual height
            //            float value = (float)(VisTools.fontHeight * fontScale);
            //            gl.glBegin(GL.GL_QUADS);
            //            gl.glVertex3f(0.0f,0.0f,0.0f);
            //            gl.glVertex3f(0.0f,value,0.0f);
            //            gl.glVertex3f(value,value,0.0f);
            //            gl.glVertex3f(value,0.0f,0.0f);
            //            gl.glEnd();

            gl.glScaled(fontScale, fontScale, fontScale);

            // Render The Text
            for (int c = 0; c < line.length(); c++) {
                char ch = line.charAt(c);
                glut.glutStrokeCharacter(font, ch);
            }

            gl.glPopMatrix();

            gl.glTranslated(0, -rowHeight, 0);

        }

        gl.glPopMatrix();
    }

    /**
     * Renders to the given VisRenderer
     * @param visRenderer the associated control panel
     */
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

        if (!enabled) {
            return;
        }

        int width = (int) glDrawable.getWidth();
        int height = (int) glDrawable.getHeight();
        if (width != oldWidth || height != oldHeight) {
            dirty = true;
            oldWidth = width;
            oldHeight = height;
        }

        GL gl = glDrawable.getGL();

        if (dirty || displayList == 0) {
            if (displayList != 0) {
                gl.glDeleteLists(displayList, 1);
            }
            displayList = gl.glGenLists(1);
            gl.glNewList(displayList, GL.GL_COMPILE);
            privateRender(visRenderer);
            gl.glEndList();
            dirty = false;
        }
        gl.glCallList(displayList);
    }

    private void privateRender(VisRenderer visRenderer) {
        GLAutoDrawable glDrawable = visRenderer.getGLAutoDrawable();

        GL gl = glDrawable.getGL();
        if (enabled == false)
            return;

        int glWidth = (int) glDrawable.getWidth();
        int glHeight = (int) glDrawable.getHeight();
        gl.glMatrixMode(GL.GL_PROJECTION);
        gl.glPushMatrix();
        gl.glLoadIdentity();
        gl.glOrtho(0, glWidth, 0, glHeight, -1.0f, 1.0f);

        gl.glMatrixMode(GL.GL_MODELVIEW);
        gl.glPushMatrix();

        gl.glLoadIdentity();

        gl.glDisable(GL.GL_LIGHTING);
        gl.glDisable(GL.GL_DEPTH_TEST);
        gl.glLineWidth(1.0f);

        if (visRenderer.getAntiAliasedLines()) {
            gl.glEnable(GL.GL_LINE_SMOOTH);
            gl.glEnable(GL.GL_BLEND);
            gl.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA);
            gl.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST);
        } else {
            gl.glDisable(GL.GL_LINE_SMOOTH);
            gl.glDisable(GL.GL_BLEND);
        }

        int pixelHeight = Math.min(height, height - topBottomMargin);
        int pixelWidth = width;

        gl.glTranslated(leftMargin, (glHeight / 2) - (pixelHeight / 2), 0);

        // draw the actual scale as a set of 10 blended quads
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

        // draw the upper string
        float stringwidth;
        float startx;
        float bump;

        bump = 0;
        stringwidth = getTextWidth(gl, highString);
        startx = (pixelWidth / 2) - (stringwidth / 2);
        if (startx < -leftMargin + leftTextMargin) {
            bump = -startx - leftMargin + leftTextMargin;
        }
        drawText(gl, bump + pixelWidth / 2, pixelHeight, highString, false);

        // draw the lower string
        bump = 0;
        stringwidth = getTextWidth(gl, lowString);
        startx = (pixelWidth / 2) - (stringwidth / 2);
        if (startx < -leftMargin + leftTextMargin) {
            bump = -startx - leftMargin + leftTextMargin;
        }

        drawText(gl, bump + pixelWidth / 2, 0, lowString, true);

        // rotate and draw the label
        gl.glRotatef(90, 0, 0, 1);
        drawText(gl, pixelHeight / 2, -pixelWidth, label, true);

        gl.glPopMatrix();

        gl.glMatrixMode(GL.GL_PROJECTION);
        gl.glPopMatrix();
        gl.glMatrixMode(GL.GL_MODELVIEW);

        gl.glEnable(GL.GL_LIGHTING);
        gl.glEnable(GL.GL_DEPTH_TEST);
    }

    public double getFontScale() {
        return fontScale;
    }

    public void setFontScale(double fontScale) {
        this.fontScale = fontScale;
        this.dirty = true;
    }

    public void resetCanvas() {
        dirty = true;
        displayList = 0;
    }

}
