package edu.uoregon.tau.paraprof.barchart;

/**
 * Holds a rectangle, basically.  We should probably use java.awt.Rectangle.
 * 
 * <P>CVS $Id: DrawObject.java,v 1.1 2005/09/26 21:12:13 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.1 $
 */

public class DrawObject {

    int xBeg, xEnd, yBeg, yEnd;
    
    
    public DrawObject(int xBeg, int yBeg, int xEnd, int yEnd) {
        this.xBeg = xBeg;
        this.xEnd = xEnd;
        this.yBeg = yBeg;
        this.yEnd = yEnd;
    }


    public int getXBeg() {
        return xBeg;
    }


    public void setXBeg(int beg) {
        xBeg = beg;
    }


    public int getXEnd() {
        return xEnd;
    }


    public void setXEnd(int end) {
        xEnd = end;
    }


    public int getYBeg() {
        return yBeg;
    }


    public void setYBeg(int beg) {
        yBeg = beg;
    }


    public int getYEnd() {
        return yEnd;
    }


    public void setYEnd(int end) {
        yEnd = end;
    }
    
    
}
