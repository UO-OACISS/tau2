package edu.uoregon.tau.vis;

import java.awt.Color;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Observable;

import javax.media.opengl.GL;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.glu.GLU;
import javax.media.opengl.glu.GLUquadric;
import javax.swing.BorderFactory;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import com.sun.opengl.util.GLUT;

public class XmasTree implements Plot {
    private List<List<Ornament>> levels;

    // rendering details
    private int displayList;
    private boolean dirty = true;

    private float sphereSize = 0.4f;
    private int sphereDetail = 8;

    private ColorScale colorScale;

    private GLUT glut = new GLUT();
    //private int font = GLUT.STROKE_MONO_ROMAN;
    //private float stringSize = 3;

    private boolean showLabels = true;

    private double verticalSpread = 2.5;
    private double radiusMultiple = 0.5;

    public static class Ornament {
        private Object userObject;

        private List<Ornament> parents;
        private List<Ornament> children;

        //private Vec position;
        private float size;
        private float color;

        private float position;

        private Vec vec;

        private String label;

        public Ornament(String label) {
            this(label, label);
        }

        public Ornament(String label, Object userObject) {
            this.label = label;
            this.userObject = userObject;
            children = new ArrayList<Ornament>();
            parents = new ArrayList<Ornament>();
        }

        public void addChild(Ornament child) {
            children.add(child);
            //child.getParents().add(this);
        }

        public List<Ornament> getChildren() {
            return children;
        }

        public String getLabel() {
            return label;
        }

        public float getColor() {
            return color;
        }

        public void setColor(float color) {
            this.color = color;
        }

        public List<Ornament> getParents() {
            return parents;
        }

        public void setParents(List<Ornament> parents) {
            this.parents = parents;
        }

        public float getPosition() {
            return position;
        }

        public void setPosition(float position) {
            this.position = position;
        }

        public float getSize() {
            return size;
        }

        public void setSize(float size) {
            this.size = size;
        }

        public Object getUserObject() {
            return userObject;
        }

        public void setUserObject(Object userObject) {
            this.userObject = userObject;
        }

        public void setChildren(List<Ornament> children) {
            this.children = children;
        }

        public Vec getVec() {
            return vec;
        }

        public void setVec(Vec vec) {
            this.vec = vec;
        }

    }

    public XmasTree(List<List<Ornament>> levels) {
        this.levels = levels;
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

    public void render(VisRenderer visRenderer) {
        GLAutoDrawable glDrawable = visRenderer.getGLAutoDrawable();

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

//    private void drawText(GL gl, double x, double y, String text) {
//        gl.glColor3f(1, 1, 1);
//        gl.glPushMatrix();
//        gl.glScalef(stringSize / 1000, stringSize / 1000, stringSize / 1000);
//        // the text seems to be about 100 in height, so move to the middle
//        gl.glTranslatef(0.0f, -50.0f, 0.0f);
//
//        // Render The Text
//        for (int c = 0; c < text.length(); c++) {
//            char ch = text.charAt(c);
//            glut.glutStrokeCharacter(font, ch);
//        }
//        gl.glPopMatrix();
//    }

    private double setRingPosition(List<Ornament> nodes, float z, double radius) {
        double ringsize = nodes.size() - 1;
        ringsize *= radiusMultiple;
        if (ringsize <= radius) {
            ringsize = radius + 5;
        }
        int c = 0;
        for (Iterator<Ornament> it2 = nodes.iterator(); it2.hasNext();) {
            Ornament o = it2.next();

            Vec pos = new Vec(ringsize * 0.5f, 0f, 0f);

            Matrix rotate = new Matrix();

            rotate.setRotateZ(2 * Math.PI * ((float) c / nodes.size()));
            pos = rotate.transform(pos);
            pos.z = z;
            o.setVec(pos);
            c++;
        }
        return ringsize;
    }

    public void privateRender(VisRenderer visRenderer) {
        GLAutoDrawable glDrawable = visRenderer.getGLAutoDrawable();
        //Vec direction = visRenderer.getViewDirection(); //TODO: No side effects?

        GL gl = glDrawable.getGL();

        //        for (int i=0; i < 100; i++) {
        //            gl.glVertex3f(i,i,i);
        //        }
        float z = 10;
        for (Iterator<List<Ornament>> it = levels.iterator(); it.hasNext();) {
            List<Ornament> level = it.next();

            List<Ornament> nodes = new ArrayList<Ornament>();
            List<Ornament> leaves = new ArrayList<Ornament>();

            for (Iterator<Ornament> it2 = level.iterator(); it2.hasNext();) {
                Ornament o = it2.next();
                if (level.size() > 5) {
                    // split into two levels
                    if (o.children.size() > 0) {
                        nodes.add(o);
                    } else {
                        leaves.add(o);
                    }
                } else {
                    nodes.add(o);
                }
            }

            double radius = 0;

            radius = setRingPosition(nodes, z, radius);
            z -= verticalSpread / 2;

            radius = setRingPosition(leaves, z, radius);
            z -= verticalSpread;

        }

        gl.glShadeModel(GL.GL_SMOOTH);

        GLU glu = new GLU();
        gl.glEnable(GL.GL_LIGHTING);
        gl.glEnable(GL.GL_DEPTH_TEST);
        gl.glFrontFace(GL.GL_CCW);
        GLUquadric qobj = glu.gluNewQuadric();
        gl.glEnable(GL.GL_CULL_FACE);
        glu.gluQuadricDrawStyle(qobj, GLU.GLU_FILL);
        glu.gluQuadricOrientation(qobj, GLU.GLU_OUTSIDE);
        glu.gluQuadricNormals(qobj, GLU.GLU_SMOOTH);

        for (Iterator<List<Ornament>> it = levels.iterator(); it.hasNext();) {
            List<Ornament> level = it.next();
            for (Iterator<Ornament> it2 = level.iterator(); it2.hasNext();) {
                Ornament o = it2.next();
                Vec pos = o.getVec();

                gl.glPushMatrix();
                gl.glTranslatef(pos.x, pos.y, pos.z);
                if (o.children.size() > 0) {
                    gl.glColor3f(25 / 255.0f, 190 / 255.0f, 200 / 255.0f);
                } else {
                    gl.glColor3f(255 / 255.0f, 0 / 255.0f, 0 / 255.0f);
                }

                Color c = colorScale.getColor(o.getColor());
                gl.glColor3f(c.getRed() / 255.0f, c.getGreen() / 255.0f, c.getBlue() / 255.0f);

                if (sphereSize < 0.1f) {
                    gl.glDisable(GL.GL_LIGHTING);
                    gl.glPointSize(2.5f);
                    gl.glBegin(GL.GL_POINTS);
                    gl.glVertex3f(0, 0, 0);
                    gl.glEnd();
                    gl.glEnable(GL.GL_LIGHTING);
                } else {
                    glu.gluSphere(qobj, Math.max(o.getSize(), 0.1) * sphereSize, sphereDetail, sphereDetail);
                }

                gl.glPopMatrix();

            }
        }

        gl.glDisable(GL.GL_LIGHTING);

        gl.glColor3f(108 / 255.0f, 108 / 255.0f, 108 / 255.0f);

        gl.glBegin(GL.GL_LINES);
        for (Iterator<List<Ornament>> it = levels.iterator(); it.hasNext();) {
            List<Ornament> level = it.next();
            for (Iterator<Ornament> it2 = level.iterator(); it2.hasNext();) {
                Ornament o = it2.next();
                Vec source = o.getVec();

                for (Iterator<Ornament> it3 = o.getChildren().iterator(); it3.hasNext();) {
                    Ornament child = it3.next();
                    Vec dest = child.getVec();
                    gl.glVertex3f(source.x, source.y, source.z);
                    gl.glVertex3f(dest.x, dest.y, dest.z);
                }
            }
        }
        gl.glEnd();

        if (showLabels) {
            gl.glDisable(GL.GL_DEPTH_TEST);
            gl.glDisable(GL.GL_LIGHTING);
            for (Iterator<List<Ornament>> it = levels.iterator(); it.hasNext();) {
                List<Ornament> level = it.next();
                for (Iterator<Ornament> it2 = level.iterator(); it2.hasNext();) {
                    Ornament o = it2.next();
                    Vec pos = o.getVec();

                    gl.glPushMatrix();
                    gl.glTranslatef(pos.x, pos.y, pos.z);
                    //Color c = colorScale.getColor(o.getColor());
                    //gl.glColor3f(c.getRed() / 255.0f, c.getGreen() / 255.0f, c.getBlue() / 255.0f);

                    // draw the label
                    String label = o.getLabel();
                    //drawText(gl, 0, 0, label);

                    gl.glColor3f(0.8f, 0.8f, 0.8f);
                    gl.glRasterPos3d(0, 0, 0);
                    glut.glutBitmapString(GLUT.BITMAP_TIMES_ROMAN_10, label);

                    gl.glPopMatrix();

                }
            }
        }

    }

    public void update(Observable o, Object arg) {
        if (o instanceof ColorScale) {
            this.dirty = true;
        }
    }

    public JPanel getControlPanel(final VisRenderer visRenderer) {
        JPanel panel = new JPanel();

        panel.setLayout(new GridBagLayout());
        panel.setBorder(BorderFactory.createLoweredBevelBorder());

      

        final JSlider verticalSpreadSlider = new JSlider(0, 100, (int) (verticalSpread * 10));
        final JSlider sphereSizeSlider = new JSlider(0, 20, (int) (sphereSize * 10));
        final JSlider sphereDetailSlider = new JSlider(3, 30, sphereDetail);
        final JSlider radiusMultipleSlider = new JSlider(0, 10, (int) (radiusMultiple * 10));

        verticalSpreadSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    XmasTree.this.setVerticalSpread(verticalSpreadSlider.getValue() / 10.0f);
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        sphereSizeSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    XmasTree.this.setSphereSize(sphereSizeSlider.getValue() / 10.0f);
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        sphereDetailSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    XmasTree.this.setSphereDetail(sphereDetailSlider.getValue());
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        radiusMultipleSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent event) {
                try {
                    XmasTree.this.setRadiusMultiple(radiusMultipleSlider.getValue() / 10.0f);
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        final JCheckBox showLabelsBox = new JCheckBox("Show Labels", showLabels);
        showLabelsBox.addActionListener(new ActionListener() {

            public void actionPerformed(ActionEvent evt) {
                try {
                    XmasTree.this.setShowLabels(showLabelsBox.isSelected());
                    visRenderer.redraw();
                } catch (Exception e) {
                    VisTools.handleException(e);
                }
            }
        });

        
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5, 5, 5, 5);
        gbc.anchor = GridBagConstraints.WEST;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 0.0;
        gbc.weighty = 0.0;
        
        VisTools.addCompItem(panel, showLabelsBox, gbc, 0, 0, 1, 1);
        VisTools.addCompItem(panel, new JLabel("Point size"), gbc, 0, 1, 1, 1);
        VisTools.addCompItem(panel, new JLabel("Vertical spread"), gbc, 0, 2, 1, 1);
        VisTools.addCompItem(panel, new JLabel("Radius multiple"), gbc, 0, 3, 1, 1);

        gbc.weightx = 1.0;
        gbc.weighty = 1.0;

        VisTools.addCompItem(panel, sphereSizeSlider, gbc, 1, 1, 1, 1);
        VisTools.addCompItem(panel, verticalSpreadSlider, gbc, 1, 2, 1, 1);
        VisTools.addCompItem(panel, radiusMultipleSlider, gbc, 1, 3, 1, 1);
        //        VisTools.addCompItem(panel, new JLabel("Point detail"), gbc, 0, 1, 1, 1);
        //        VisTools.addCompItem(panel, sphereDetailSlider, gbc, 1, 1, 1, 1);

        return panel;

    }

    public int getSphereDetail() {
        return sphereDetail;
    }

    public void setSphereDetail(int sphereDetail) {
        this.sphereDetail = sphereDetail;
        dirty = true;
    }

    public boolean getShowLabels() {
        return showLabels;
    }

    public void setShowLabels(boolean showLabels) {
        this.showLabels = showLabels;
        dirty = true;
    }

    public void cleanUp() {
    // TODO Auto-generated method stub

    }

    public Axes getAxes() {
        // TODO Auto-generated method stub
        return null;
    }

    public float getDepth() {
        // TODO Auto-generated method stub
        return 0;
    }

    public float getHeight() {
        // TODO Auto-generated method stub
        return 0;
    }

    public String getName() {
        return "XmasTree";
    }

    public int getSelectedCol() {
        // TODO Auto-generated method stub
        return 0;
    }

    public int getSelectedRow() {
        // TODO Auto-generated method stub
        return 0;
    }

    public float getWidth() {
        // TODO Auto-generated method stub
        return 0;
    }

    public void setAxes(Axes axes) {
    // TODO Auto-generated method stub

    }

    public void setSelectedCol(int selectedCol) {
    // TODO Auto-generated method stub

    }

    public void setSelectedRow(int selectedRow) {
    // TODO Auto-generated method stub

    }

    public void setSize(float xSize, float ySize, float zSize) {
    // TODO Auto-generated method stub

    }

    /**
     * Sets the sphere size.
     * @param sphereSize the desired sphere size.
     */
    public void setSphereSize(float sphereSize) {
        this.sphereSize = sphereSize;
        this.dirty = true;
    }

    public void setVerticalSpread(double spread) {
        this.verticalSpread = spread;
        this.dirty = true;
    }

    public void setRadiusMultiple(double multiple) {
        this.radiusMultiple = multiple;
        this.dirty = true;
    }

    public void resetCanvas() {
        dirty = true;
        displayList = 0;
    }

}
