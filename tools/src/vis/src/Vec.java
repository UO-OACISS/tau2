/*
 * Vec.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

/**
 * Utility class for vector operations.
 * 
 * Note: many of the operations exclude 'w' from processing.
 *   
 * <P>CVS $Id: Vec.java,v 1.4 2006/11/01 01:50:33 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.4 $
 * @see Matrix
 */
public class Vec {
    float x;
    float y;
    float z;
    float w;

    /**
     * Constructor with float parameters
     * @param x		value in the x direction
     * @param y		value in the y direction
     * @param z		value in the z direction
     */
    public Vec(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = 1;
    }

    /**
     * Constructor with double parameters
     * @param x		value in the x direction
     * @param y		value in the y direction
     * @param z		value in the z direction
     */
    public Vec(double x, double y, double z) {
        this.x = (float) x;
        this.y = (float) y;
        this.z = (float) z;
        this.w = 1;
    }

    /**
     * Copy constructor
     * @param rhs	Vec to copy
     */
    public Vec(Vec rhs) {
        this.x = rhs.x;
        this.y = rhs.y;
        this.z = rhs.z;
        this.w = rhs.w;
    }

    /**
     * returns this Vec's x value
     * @return this Vec's x value
     */
    public float x() {
        return x;
    }

    /**
     * returns this Vec's y value
     * @return this Vec's y value
     */
    public float y() {
        return y;
    }

    /**
     * returns this Vec's z value
     * @return this Vec's z value
     */
    public float z() {
        return z;
    }

    /**
     * returns this Vec's w value
     * @return this Vec's w value
     */
    public float w() {
        return w;
    }

    /**
     * Sets this Vec's x value
     * @param x
     */
    public void setx(double x) {
        this.x = (float) x;
    }

    /**
     * Sets this Vec's y value
     * @param y
     */
    public void sety(double y) {
        this.y = (float) y;
    }

    /**
     * Sets this Vec's z value
     * @param z
     */
    public void setz(double z) {
        this.z = (float) z;
    }

    /**
     * Sets this Vec's w value
     * @param w
     */
    public void setw(double w) {
        this.w = (float) w;
    }

    /**
     * Sets this Vec's x value
     * @param x
     */
    public void setx(float x) {
        this.x = x;
    }

    /**
     * Sets this Vec's y value
     * @param y
     */
    public void sety(float y) {
        this.y = y;
    }

    /**
     * Sets this Vec's z value
     * @param z
     */
    public void setz(float z) {
        this.z = z;
    }

    /**
     * Sets this Vec's w value
     * @param w
     */
    public void setw(float w) {
        this.w = w;
    }

    public void scale(float x, float y, float z) {
        this.x *= x;
        this.y *= y;
        this.z *= z;
    }
    
    public void scale(float value) {
        this.x *= value;
        this.y *= value;
        this.z *= value;
    }

    
    /**
     * Adds two Vecs together.  Vec adding is commutative
     * @param a		the first Vec
     * @param b		the second Vec
     * @return		the sum of the two Vecs
     */
    public static Vec add(Vec a, Vec b) {
        return new Vec(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
    }

    /**
     * Returns the length of the Vec
     * @return		the length of the Vec
     */
    public double length() {
        double l2 = (x * x) + (y * y) + (z * z);
        double l = Math.sqrt(l2);
        return l;
    }

    /**
     *	Normalizes this Vec
     */
    public void normalize() {
        double l2 = (x * x) + (y * y) + (z * z);
        double l = Math.sqrt(l2);
        x = (float) (x / l);
        y = (float) (y / l);
        z = (float) (z / l);
    }

    /**
     * Adds this Vec and another together and returns the result
     * @param rhs		the second Vec
     * @return			the result
     */
    public Vec add(Vec rhs) {
        return new Vec(this.x + rhs.x, this.y + rhs.y, this.z + rhs.z);
    }

    /**
     * Subtracts a Vec from this one.
     * @param rhs		the second Vec
     * @return			the result
     */
    public Vec subtract(Vec rhs) {
        return new Vec(this.x - rhs.x, this.y - rhs.y, this.z - rhs.z);
    }

    /**
     * Computes the cross product of two Vecs
     * @param rhs		the second Vec
     * @return			the result
     */
    public Vec cross(Vec rhs) {
        return new Vec((this.y * rhs.z) - (this.z * rhs.y), (this.z * rhs.x) - (this.x * rhs.z),
                (this.x * rhs.y) - (this.y * rhs.x));
    }

    
    /**
     * Returns a string representation of the Vec (for debug purposes)
     */
    public String toString() {
        return "(" + x + ", " + y + ", " + z + ")";
    }

}
