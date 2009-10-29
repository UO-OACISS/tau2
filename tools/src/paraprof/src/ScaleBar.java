package edu.uoregon.tau.paraprof;

import java.awt.*;

import javax.swing.JComponent;

/**
 * Draws a simple progressbar style line to indicate position along an axis
 *
 * <P>CVS $Id: ScaleBar.java,v 1.1 2009/10/29 23:58:22 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */
public class ScaleBar extends JComponent {
    private float ratio; // 0 to 1

    public ScaleBar() {
        int height = 18;
        setPreferredSize(new Dimension(10, height));
        setMinimumSize(new Dimension(0, height));
        setMaximumSize(new Dimension(Integer.MAX_VALUE, height));
    }

    public void setPosition(float position) {
        this.ratio = position;
        repaint();
    }

    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        // paint the background
        g.setColor(getBackground());
        g.fillRect(0, 0, getSize().width, getSize().height);

        // paint the lines
        int width = getSize().width - 1;
        int pos = (int) (width * ratio);
        g.setColor(Color.blue);
        g.fillRect(pos-2, 0, 5, getSize().height - 1);
    }
}
