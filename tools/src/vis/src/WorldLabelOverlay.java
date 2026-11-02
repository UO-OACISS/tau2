package edu.uoregon.tau.vis;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import com.jogamp.opengl.GL2;
import com.jogamp.opengl.util.gl2.GLUT;

/** Fixed-pixel text anchored to scene-space positions. */
public final class WorldLabelOverlay implements VisRenderer.Overlay {
    public static final class Label {
        private final String text;
        private final float x, y, z;
        private final float referenceX, referenceY, referenceZ;
        private final int margin;
        private final int fallbackX, fallbackY;

        /**
         * Creates a label displaced outward from reference to anchor in screen space.
         * The fallback direction is used when both points project to the same pixel.
         */
        public Label(String text, float x, float y, float z,
                float referenceX, float referenceY, float referenceZ,
                int margin, int fallbackX, int fallbackY) {
            this.text = text;
            this.x = x; this.y = y; this.z = z;
            this.referenceX = referenceX;
            this.referenceY = referenceY;
            this.referenceZ = referenceZ;
            this.margin = margin;
            this.fallbackX = fallbackX;
            this.fallbackY = fallbackY;
        }
    }

    private final GLUT glut = new GLUT();
    private final List<Label> labels;

    public WorldLabelOverlay(List<Label> labels) {
        this.labels = Collections.unmodifiableList(new ArrayList<Label>(labels));
    }

    public void render(VisRenderer renderer, VisRenderer.View view) {
        GL2 gl = renderer.getGLAutoDrawable().getGL().getGL2();
        VisTools.glApplyInvertableColor(renderer, Color.white);
        double[] anchor = new double[3];
        double[] reference = new double[3];
        for (Label label : labels) {
            if (label.text == null || label.text.length() == 0
                    || !view.project(label.x, label.y, label.z, anchor)
                    || anchor[2] < 0.0 || anchor[2] > 1.0) continue;

            double dx = label.fallbackX;
            double dy = label.fallbackY;
            if (view.project(label.referenceX, label.referenceY,
                    label.referenceZ, reference)) {
                double projectedX = anchor[0] - reference[0];
                double projectedY = anchor[1] - reference[1];
                double length = Math.sqrt(projectedX * projectedX
                        + projectedY * projectedY);
                if (length >= 1.0) {
                    dx = projectedX / length;
                    dy = projectedY / length;
                }
            }
            double fallbackLength = Math.sqrt(dx * dx + dy * dy);
            if (fallbackLength >= 1.0) {
                dx /= fallbackLength;
                dy /= fallbackLength;
            }

            int font = GLUT.BITMAP_HELVETICA_18;
            int width = glut.glutBitmapLength(font, label.text);
            gl.glRasterPos2d(anchor[0] + label.margin * dx - width / 2.0,
                    anchor[1] + label.margin * dy - 9.0);
            for (int i = 0; i < label.text.length(); i++)
                glut.glutBitmapCharacter(font, label.text.charAt(i));
        }
    }
}
