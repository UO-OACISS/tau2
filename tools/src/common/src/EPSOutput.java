package edu.uoregon.tau.common;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Composite;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsConfiguration;
import java.awt.Image;
import java.awt.Paint;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.RenderingHints.Key;
import java.awt.Shape;
import java.awt.Stroke;
import java.awt.font.FontRenderContext;
import java.awt.font.GlyphVector;
import java.awt.font.TextLayout;
import java.awt.geom.AffineTransform;
import java.awt.geom.Arc2D;
import java.awt.geom.Ellipse2D;
import java.awt.geom.GeneralPath;
import java.awt.geom.Line2D;
import java.awt.geom.PathIterator;
import java.awt.geom.Point2D;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.BufferedImageOp;
import java.awt.image.ImageObserver;
import java.awt.image.RenderedImage;
import java.awt.image.renderable.RenderableImage;
import java.io.File;
import java.io.IOException;
import java.text.AttributedCharacterIterator;
import java.util.Map;

public class EPSOutput extends Graphics2D {

    //private int width, height;
    private Graphics2D g2d;
    private boolean drawTextAsShapes;
    private EPSWriter writer;

    private EPSOutput(EPSOutput other) {
        this.writer = other.writer;
        this.g2d = (Graphics2D) other.g2d.create();
        this.drawTextAsShapes = other.drawTextAsShapes;
    }

    public EPSOutput(String name, File file, int width, int height) {
        writer = new EPSWriter(name, file, width, height);
        //this.width = width;
        //this.height = height;
        BufferedImage bi = new BufferedImage(5, 5, BufferedImage.TYPE_INT_ARGB);
        g2d = bi.createGraphics();
        setClip(0, 0, width, height);

        setStroke(new BasicStroke());
    }

    private void output(String str) {
        writer.write(this, str);
    }
    
    private void comment(String str) {
        writer.comment(str);
    }

    public void finish() throws IOException {
        writer.finish();
    }

    public void rotate(double theta) {
        g2d.rotate(theta);
    }

    public void scale(double sx, double sy) {
        g2d.scale(sx, sy);
    }

    public void shear(double shx, double shy) {
        g2d.shear(shx, shy);
    }

    public void translate(double tx, double ty) {
        g2d.translate(tx, ty);
    }

    public void rotate(double theta, double x, double y) {
        g2d.rotate(theta, x, y);
    }

    public void translate(int x, int y) {
        translate((double) x, (double) y);
    }

    public Color getBackground() {
        return g2d.getBackground();
    }

    public void setBackground(Color color) {
        g2d.setBackground(color);
    }

    public Composite getComposite() {
        return g2d.getComposite();
    }

    public void setComposite(Composite comp) {
        g2d.setComposite(comp);
    }

    public GraphicsConfiguration getDeviceConfiguration() {
        return g2d.getDeviceConfiguration();
    }

    public Paint getPaint() {
        return g2d.getPaint();
    }

    public void setPaint(Paint paint) {
        g2d.setPaint(paint);
        if (paint instanceof Color) {
            setColor((Color) paint);
        }
    }

    public RenderingHints getRenderingHints() {
        return g2d.getRenderingHints();
    }

    public void clip(Shape s) {
        comment("clip(" + s + ")\n");
        g2d.clip(s);
        setClip(g2d.getClip());
    }

    private String transformedPoint(double x, double y) {
        Point2D p1 = new Point2D.Double(x, y);
        Point2D p2 = g2d.getTransform().transform(p1, null);
        return ((Point2D.Double) p2).x + " " + -((Point2D.Double) p2).y;
    }

    private void outputPath(Shape s) {
        output("newpath\n");

        // keep these around for quadratic beziers
        double lastx = 0, lasty = 0;

        for (PathIterator pi = s.getPathIterator(null); !pi.isDone(); pi.next()) {

            double[] coords = new double[6];
            int type = pi.currentSegment(coords);
            switch (type) {
            case PathIterator.SEG_MOVETO:
                output(transformedPoint(coords[0], coords[1]) + " moveto\n");
                lastx = coords[0];
                lasty = coords[1];
                break;
            case PathIterator.SEG_LINETO:
                output(transformedPoint(coords[0], coords[1]) + " lineto\n");
                lastx = coords[0];
                lasty = coords[1];
                break;
            case PathIterator.SEG_QUADTO:

                /* PostScript only has cubic bezier curves
                 
                 Given a quadratic spline with endpoints Q0 and Q2, and control point 
                 Q1, you can calculate the cubic control points as follows: 


                 B0 = Q0 
                 B1 = (Q0 + 2*Q1) / 3 
                 B2 = (Q2 + 2*Q1) / 3 
                 B3 = Q2
                 */
                double[] n = new double[4];

                n[0] = (lastx + 2.0f * coords[0]) / 3.0f;
                n[1] = (lasty + 2.0f * coords[1]) / 3.0f;
                n[2] = (coords[2] + 2.0f * coords[0]) / 3.0f;
                n[3] = (coords[3] + 2.0f * coords[1]) / 3.0f;

                output(transformedPoint(n[0], n[1]) + " " + transformedPoint(n[2], n[3]) + " "
                        + transformedPoint(coords[2], coords[3]) + " curveto\n");

                lastx = coords[2];
                lasty = coords[3];
                break;
            case PathIterator.SEG_CUBICTO:
                output(transformedPoint(coords[0], coords[1]) + " " + transformedPoint(coords[2], coords[3]) + " "
                        + transformedPoint(coords[4], coords[5]) + " curveto\n");

                lastx = coords[4];
                lasty = coords[5];
                break;
            case PathIterator.SEG_CLOSE:
                output("closepath\n");
                break;
            }
        }

    }

    public void draw(Shape s) {
        comment("draw(" + s + ")\n");
        outputPath(s);
        output("stroke\n");
    }

    public void fill(Shape s) {
        comment("fill(" + s + ")\n");
        outputPath(s);
        if (s.getPathIterator(null).getWindingRule() == PathIterator.WIND_EVEN_ODD) {
            output("eofill\n");
        } else {
            output("fill\n");
        }
    }

    public Stroke getStroke() {
        return g2d.getStroke();
    }

    public void setStroke(Stroke s) {
        comment("setStroke(" + s + ")\n");

        g2d.setStroke(s);

        if (s instanceof BasicStroke) {
            BasicStroke bs = (BasicStroke) s;
            output(bs.getLineWidth() + " setlinewidth\n");
            output(bs.getEndCap() + " setlinecap\n");
            output(bs.getLineJoin() + " setlinejoin\n");
            output(Math.max(bs.getMiterLimit(), 1) + " setmiterlimit\n");

            // dashes
            output("[ ");
            float[] dashes = bs.getDashArray();
            if (dashes != null) {
                for (int i = 0; i < dashes.length; i++) {
                    output((dashes[i]) + " ");
                }
            }
            output("] 0 setdash\n");
        }
    }

    public FontRenderContext getFontRenderContext() {
        return g2d.getFontRenderContext();
    }

    public void drawGlyphVector(GlyphVector g, float x, float y) {
        Shape outline = g.getOutline(x, y);
        fill(outline);
    }

    public AffineTransform getTransform() {
        return g2d.getTransform();
    }

    public void setTransform(AffineTransform Tx) {
        g2d.setTransform(Tx);
    }

    public void transform(AffineTransform Tx) {
        g2d.transform(Tx);
    }

    public void drawString(String s, float x, float y) {
        if (!drawTextAsShapes) {
            output(transformedPoint(x, y) + " moveto\n");
            output("(" + s.replaceAll("\\)", "\\\\)").replaceAll("\\(", "\\\\(") + ") show\n");
        } else {
            GlyphVector gv = getFont().createGlyphVector(getFontRenderContext(), s);
            drawGlyphVector(gv, x, y);
        }
    }

    public void drawString(String str, int x, int y) {
        drawString(str, (float) x, (float) y);
    }

    public void drawString(AttributedCharacterIterator iterator, float x, float y) {
        TextLayout layout = new TextLayout(iterator, getFontRenderContext());
        layout.draw(this, x, y);
    }

    public void drawString(AttributedCharacterIterator iterator, int x, int y) {
        TextLayout layout = new TextLayout(iterator, getFontRenderContext());
        layout.draw(this, x, y);
    }

    @SuppressWarnings("rawtypes")
	public void addRenderingHints(Map hints) {
        g2d.addRenderingHints(hints);
    }

    @SuppressWarnings("rawtypes")
	public void setRenderingHints(Map hints) {
        g2d.setRenderingHints(hints);
    }

    public boolean hit(Rectangle rect, Shape s, boolean onStroke) {
        return g2d.hit(rect, s, onStroke);
    }

    public void drawRenderedImage(RenderedImage img, AffineTransform xform) {
        // TODO Auto-generated method stub

    }

    public void drawRenderableImage(RenderableImage img, AffineTransform xform) {
        // TODO Auto-generated method stub

    }

    public void drawImage(BufferedImage img, BufferedImageOp op, int x, int y) {
        // TODO Auto-generated method stub

    }

    public Object getRenderingHint(Key hintKey) {
        return g2d.getRenderingHint(hintKey);
    }

    public void setRenderingHint(Key hintKey, Object hintValue) {
        g2d.setRenderingHint(hintKey, hintValue);
    }

    public boolean drawImage(Image img, AffineTransform xform, ImageObserver obs) {
        // TODO Auto-generated method stub
        return false;
    }

    public void dispose() {
        // TODO Auto-generated method stub
    }

    public void setPaintMode() {
        g2d.setPaintMode();
    }

    public void clearRect(int x, int y, int width, int height) {
        Color c = getColor();
        setColor(getBackground());
        outputPath(new Rectangle(x, y, width, height));
        output("fill\n");
        setColor(c);
    }

    public void clipRect(int x, int y, int width, int height) {
        this.clip(new Rectangle(x, y, width, height));
    }

    public void drawLine(int x1, int y1, int x2, int y2) {
        draw(new Line2D.Double(x1, y1, x2, y2));
    }

    public void drawOval(int x, int y, int width, int height) {
        Shape s = new Ellipse2D.Float(x, y, width, height);
        draw(s);
    }

    public void fillOval(int x, int y, int width, int height) {
        Shape s = new Ellipse2D.Float(x, y, width, height);
        fill(s);
    }

    public void fillRect(int x, int y, int width, int height) {
        fill(new Rectangle(x, y, width, height));
    }

    public void drawRect(int x, int y, int width, int height) {
        draw(new Rectangle(x, y, width, height));
    }

    public void setClip(int x, int y, int width, int height) {
        setClip(new Rectangle(x, y, width, height));
    }

    public void copyArea(int x, int y, int width, int height, int dx, int dy) {
        // TODO Auto-generated method stub
    }

    public void drawArc(int x, int y, int width, int height, int startAngle, int arcAngle) {
        Shape s = new Arc2D.Float(x, y, width, height, startAngle, arcAngle, Arc2D.OPEN);
        draw(s);
    }

    public void drawRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight) {
        Shape s = new RoundRectangle2D.Float(x, y, width, height, arcWidth, arcHeight);
        draw(s);
    }

    public void fillArc(int x, int y, int width, int height, int startAngle, int arcAngle) {
        Shape s = new Arc2D.Float(x, y, width, height, startAngle, arcAngle, Arc2D.OPEN);
        fill(s);
    }

    public void fillRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight) {
        Shape s = new RoundRectangle2D.Float(x, y, width, height, arcWidth, arcHeight);
        fill(s);
    }

    public void drawPolygon(int[] xPoints, int[] yPoints, int nPoints) {
        Shape shape = new Polygon(xPoints, yPoints, nPoints);
        draw(shape);
    }

    public void drawPolyline(int[] xPoints, int[] yPoints, int nPoints) {
        if (nPoints > 0) {
            GeneralPath path = new GeneralPath();
            path.moveTo(xPoints[0], yPoints[0]);
            for (int i = 1; i < nPoints; i++) {
                path.lineTo(xPoints[i], yPoints[i]);
            }
            draw(path);
        }
    }

    public void fillPolygon(int[] xPoints, int[] yPoints, int nPoints) {
        Shape shape = new Polygon(xPoints, yPoints, nPoints);
        fill(shape);
    }

    public Color getColor() {
        return g2d.getColor();
    }

    public void setColor(Color c) {
        comment("setColor(" + c + ")\n");
        g2d.setColor(c);
        if (c == null) {
            c = Color.BLACK;
        }
        output(c.getRed() / 255.0 + " " + c.getGreen() / 255.0 + " " + c.getBlue() / 255.0 + " setrgbcolor\n");
    }

    public void setXORMode(Color c1) {
        // TODO Auto-generated method stub
    }

    public Font getFont() {
        return g2d.getFont();
    }

    public void setFont(Font font) {
        comment("setFont(" + font + ")\n");
        g2d.setFont(font);
        output("/" + font.getPSName() + " findfont " + font.getSize() + " scalefont setfont\n");
    }

    public Graphics create() {
        return new EPSOutput(this);
    }

    public Rectangle getClipBounds() {
        return g2d.getClipBounds();
    }

    public Shape getClip() {
        return g2d.getClip();
    }

    public void setClip(Shape clip) {
        comment("setClip(" + clip + ")\n");
        comment("    old clip: " + g2d.getClip() + "\n");
        if (clip == null) {
            output("grestore\n");
        } else {
            if (g2d.getClip() != null) {
                output("grestore\n");
            }
            output("gsave\n");
            outputPath(clip);
            output("clip\n");
        }
        g2d.setClip(clip);
    }

    public FontMetrics getFontMetrics(Font f) {
        return g2d.getFontMetrics(f);
    }

    public boolean drawImage(Image img, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2,
            ImageObserver observer) {
        // TODO Auto-generated method stub
        return false;
    }

    public boolean drawImage(Image img, int x, int y, int width, int height, ImageObserver observer) {
        // TODO Auto-generated method stub
        return false;
    }

    public boolean drawImage(Image img, int x, int y, ImageObserver observer) {
        // TODO Auto-generated method stub
        return false;
    }

    public boolean drawImage(Image img, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2, Color bgcolor,
            ImageObserver observer) {
        // TODO Auto-generated method stub
        return false;
    }

    public boolean drawImage(Image img, int x, int y, int width, int height, Color bgcolor, ImageObserver observer) {
        // TODO Auto-generated method stub
        return false;
    }

    public boolean drawImage(Image img, int x, int y, Color bgcolor, ImageObserver observer) {
        // TODO Auto-generated method stub
        return false;
    }

    public boolean getDrawTextAsShapes() {
        return drawTextAsShapes;
    }

    public void setDrawTextAsShapes(boolean drawTextAsShapes) {
        this.drawTextAsShapes = drawTextAsShapes;
    }

}
