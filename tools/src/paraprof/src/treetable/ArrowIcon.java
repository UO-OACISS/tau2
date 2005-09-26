package edu.uoregon.tau.paraprof.treetable;

import java.awt.Color;
import java.awt.Component;
import java.awt.Graphics;

import javax.swing.Icon;

public class ArrowIcon implements Icon {
    public static final int NONE = 0;
    public static final int DOWN = 1;
    public static final int UP = 2;

    private int direction;
    private int width = 8;
    private int height = 8;

    public ArrowIcon(int direction) {
        this.direction = direction;
    }

    public int getIconWidth() {
        return width;
    }

    public int getIconHeight() {
        return height;
    }

    public void paintIcon(Component c, Graphics g, int x, int y) {

        Color color = new Color(0, 150, 150);

        int w = width;
        int h = height;
        int m = w / 2;
        if (direction == UP) {
            g.setColor(color);
            g.drawLine(x, y, x + w, y);
            g.drawLine(x, y, x + m, y + h);
            g.setColor(color);
            g.drawLine(x + w, y, x + m, y + h);
        }
        if (direction == DOWN) {
            g.setColor(color);
            g.drawLine(x + m, y, x, y + h);
            g.setColor(color);
            g.drawLine(x, y + h, x + w, y + h);
            g.drawLine(x + m, y, x + w, y + h);
        }
    }
}
