package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;

public class WindowPlacer {

    private static Point lastGlobalDataWindowPosition;

    private static int screenHeight;
    private static int screenWidth;

    private static Point lastLocation;

    private static List visibleWindows = new ArrayList();

    private static int lastXOffset;

    private static boolean initialized = false;

    private static void checkInit() {
	if (initialized == false) {
	    Toolkit tk = Toolkit.getDefaultToolkit();
	    Dimension screenDimension = tk.getScreenSize();
	    screenHeight = screenDimension.height;
	    screenWidth = screenDimension.width;
	    initialized = true;
	}
    }

    private static int getScreenWidth() {
	checkInit();
	return screenWidth;
    }

    private static int getScreenHeight() {
	checkInit();
	return screenHeight;
    }

    public static void addToVisibleList(JFrame frame) {
        visibleWindows.add(frame);
    }
    
    public static void removeFromVisibleList(JFrame frame) {
        visibleWindows.remove(frame);
    }

    public static Point getGlobalDataWindowPosition(JFrame frame) {

        int xPosition = ParaProf.paraProfManagerWindow.getLocation().x;
        int yPosition = ParaProf.paraProfManagerWindow.getLocation().y;

        Point placement = new Point(xPosition + 75, yPosition + 110);

        if (placement.equals(lastGlobalDataWindowPosition)) {
            placement.translate(25, 25);
        }
        lastGlobalDataWindowPosition = placement;

        if (placement.x + frame.getWidth() > getScreenWidth()) {
            placement.setLocation(0, placement.y);
        }

        if (placement.y + frame.getHeight() > getScreenHeight()) {
            placement.setLocation(placement.x, 0);
        }

        sanityCheck(frame, placement);
        return placement;
    }

    private static int getProperXPosition(Component parent) {
        //        System.out.println("\n----------------------\n");
        //        System.out.println("parent = " + parent);
        while (parent != null && !(parent instanceof JFrame)) {
            parent = parent.getParent();
            //            System.out.println("parent = " + parent);
        }

        //        if (parent instanceof JFrame) {
        //            System.err.println("\nfound frame!");
        //        }

        if (parent instanceof JFrame && parent.getLocation().x >= getScreenWidth() / 2) {
            return getScreenWidth() / 2;
        }
        return 0;
    }

    private static Point getParentLocation(Component parent) {
        while (parent != null && !(parent instanceof JFrame)) {
            parent = parent.getParent();
        }
        if (parent instanceof JFrame) {
            return parent.getLocation();
        }
        return new Point(getProperXPosition(parent), 0);
    }

    private static Dimension getParentSize(Component parent) {
        while (parent != null && !(parent instanceof JFrame)) {
            parent = parent.getParent();
        }
        if (parent instanceof JFrame) {
            return parent.getSize();
        }
        return new Dimension(50, 50);
    }

    public static void sanityCheck(JFrame frame, Point p) {
        int x = p.x;
        int y = p.y;
        x = Math.min(x, getScreenWidth() - frame.getWidth());
        x = Math.max(x, 0);

        y = Math.min(y, getScreenHeight() - frame.getHeight());
        y = Math.max(y, 0);
        p.setLocation(x, y);
        

        if (ParaProf.demoMode) {
            p.setLocation(0, 0);
        }
    }

    public static Point getNewLocation(JFrame frame, Component parent) {
        if (lastLocation == null) {
            lastLocation = new Point(getProperXPosition(parent), 0);
            sanityCheck(frame, lastLocation);
            return lastLocation;
        }

        lastXOffset += 25;
        int x = getProperXPosition(parent) + lastXOffset;
        int y = lastLocation.y + 25;

        if (x + frame.getWidth() > getScreenWidth()) {
            x = getProperXPosition(parent);

            x = Math.min(x, getScreenWidth() - frame.getWidth());
            x = Math.max(x, 0);
            lastXOffset = 0;
        }

        if (y + frame.getHeight() > getScreenHeight()) {
            y = 0;
        }

        lastLocation = new Point(x, y);
        sanityCheck(frame, lastLocation);
        return lastLocation;
    }

    public static Point getNewLedgerLocation(JFrame frame, Component parent) {
        Point p = getParentLocation(parent);

        int x = p.x;
        int y = p.y + (int) (Math.random() * 50);

        //x = x + getParentSize(parent).width;
        x = x - frame.getWidth();

        p.setLocation(x, y);
        sanityCheck(frame, p);
        return p;
    }


}
