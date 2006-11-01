/*
 * Matrix.java
 *
 * Copyright 2005-2006                                
 * Performance Research Laboratory, University of Oregon
 */
package edu.uoregon.tau.vis;

/**
 * Utility class for matrix operations.  The matrices represented by this 
 * class are always 4x4 in size.
 *    
 * TODO : This class is complete
 *
 * <P>CVS $Id: Matrix.java,v 1.4 2006/11/01 03:20:48 amorris Exp $</P>
 * @author	Alan Morris
 * @version	$Revision: 1.4 $
 * @see Vec
 */
public class Matrix {

    private double[][] matrix = new double[4][4];

    /** 
     * Class constructor.  This matrix will be initialized to the identity matrix
     */
    public Matrix() {
        // initialize to the identity matrix;
        setIdentity();
    }

    /**
     * Sets this matrix to the identity matrix.
     */
    public void setIdentity() {
        matrix[0][0] = 1;
        matrix[0][1] = 0;
        matrix[0][2] = 0;
        matrix[0][3] = 0;
        matrix[1][0] = 0;
        matrix[1][1] = 1;
        matrix[1][2] = 0;
        matrix[1][3] = 0;
        matrix[2][0] = 0;
        matrix[2][1] = 0;
        matrix[2][2] = 1;
        matrix[2][3] = 0;
        matrix[3][0] = 0;
        matrix[3][1] = 0;
        matrix[3][2] = 0;
        matrix[3][3] = 1;
    }

    /**
     * Sets this matrix to a translation matrix
     * @param x		the amount in the x direction
     * @param y		the amount in the y direction
     * @param z		the amount in the z direction
     */
    public void setToTranslate(double x, double y, double z) {
        setIdentity();
        matrix[0][3] = x;
        matrix[1][3] = y;
        matrix[2][3] = z;
    }

    /**
     * Sets this matrix to an orthogonal rotation matrix
     * @param u		the 'u' vector
     * @param v		the 'v' vector
     * @param n		the 'n' vector
     */
    public void setOrthRotate(Vec u, Vec v, Vec n) {
        matrix[0][0] = u.x();
        matrix[0][1] = u.y();
        matrix[0][2] = u.z();
        matrix[0][3] = 0;

        matrix[1][0] = v.x();
        matrix[1][1] = v.y();
        matrix[1][2] = v.z();
        matrix[1][3] = 0;

        matrix[2][0] = n.x();
        matrix[2][1] = n.y();
        matrix[2][2] = n.z();
        matrix[2][3] = 0;

        matrix[3][0] = 0;
        matrix[3][1] = 0;
        matrix[3][2] = 0;
        matrix[3][3] = 1;
    }

    /**
     * Transposes this matrix
     */
    public void transpose() {
        // create a new matrix to place transposed values
        Matrix m = new Matrix();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                m.matrix[i][j] = matrix[j][i];
            }
        }

        // now copy back
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                matrix[i][j] = m.matrix[i][j];
            }
        }
    }

    /**
     * Transforms a given vector with this matrix
     * @param v		the vector to transform
     * @return		the transformed vector
     */
    public Vec transform(Vec v) {
        Vec out = new Vec(0.0, 0.0, 0.0);
        out.setx((v.x() * matrix[0][0]) + (v.y() * matrix[0][1]) + (v.z() * matrix[0][2]) + (v.w() * matrix[0][3]));
        out.sety((v.x() * matrix[1][0]) + (v.y() * matrix[1][1]) + (v.z() * matrix[1][2]) + (v.w() * matrix[1][3]));
        out.setz((v.x() * matrix[2][0]) + (v.y() * matrix[2][1]) + (v.z() * matrix[2][2]) + (v.w() * matrix[2][3]));
        out.setw((v.x() * matrix[3][0]) + (v.y() * matrix[3][1]) + (v.z() * matrix[3][2]) + (v.w() * matrix[3][3]));

        return out;
    }

    /**
     * Creates a rotation matrix around the X axis
     * @param theta the angle to rotate
     * @return      the transformation matrix
     */
    public static Matrix createRotateX(double theta) {
        Matrix m = new Matrix();
        m.setRotateX(theta);
        return m;
    }

    /**
     * Creates a rotation matrix around the Y axis
     * @param theta the angle to rotate
     * @return      the transformation matrix
     */
    public static Matrix createRotateY(double theta) {
        Matrix m = new Matrix();
        m.setRotateY(theta);
        return m;
    }

    /**
     * Creates a rotation matrix around the Z axis
     * @param theta the angle to rotate
     * @return      the transformation matrix
     */
    public static Matrix createRotateZ(double theta) {
        Matrix m = new Matrix();
        m.setRotateZ(theta);
        return m;
    }

    /**
     * Sets the transformation matrix to a rotation about the X
     * @param theta the angle to rotate
     */
    public void setRotateX(double theta) {
        matrix[0][0] = 1;
        matrix[0][1] = 0;
        matrix[0][2] = 0;
        matrix[0][3] = 0;
        matrix[1][0] = 0;
        matrix[1][1] = Math.cos(theta);
        matrix[1][2] = -Math.sin(theta);
        matrix[1][3] = 0;
        matrix[2][0] = 0;
        matrix[2][1] = Math.sin(theta);
        matrix[2][2] = Math.cos(theta);
        matrix[2][3] = 0;
        matrix[3][0] = 0;
        matrix[3][1] = 0;
        matrix[3][2] = 0;
        matrix[3][3] = 1;
    }

    /**
     * Sets the transformation matrix to a rotation about the Y
     * @param theta the angle to rotate
     */
    public void setRotateY(double theta) {
        matrix[0][0] = Math.cos(theta);
        matrix[0][1] = 0;
        matrix[0][2] = Math.sin(theta);
        matrix[0][3] = 0;
        matrix[1][0] = 0;
        matrix[1][1] = 1;
        matrix[1][2] = 0;
        matrix[1][3] = 0;
        matrix[2][0] = -Math.sin(theta);
        matrix[2][1] = 0;
        matrix[2][2] = Math.cos(theta);
        matrix[2][3] = 0;
        matrix[3][0] = 0;
        matrix[3][1] = 0;
        matrix[3][2] = 0;
        matrix[3][3] = 1;
    }

    /**
     * Sets the transformation matrix to a rotation about the Z
     * @param theta the angle to rotate
     */
    public void setRotateZ(double theta) {
        matrix[0][0] = Math.cos(theta);
        matrix[0][1] = -Math.sin(theta);
        matrix[0][2] = 0;
        matrix[0][3] = 0;
        matrix[1][0] = Math.sin(theta);
        matrix[1][1] = Math.cos(theta);
        matrix[1][2] = 0;
        matrix[1][3] = 0;
        matrix[2][0] = 0;
        matrix[2][1] = 0;
        matrix[2][2] = 1;
        matrix[2][3] = 0;
        matrix[3][0] = 0;
        matrix[3][1] = 0;
        matrix[3][2] = 0;
        matrix[3][3] = 1;
    }

    /**
     * Multiply two matrices (this and rhs) and return the result.
     * C = this * rhs
     * Note: this matrix and rhs are unnaffected.
     * @param rhs		the right hand side matrix
     * @return			the resulting (C) matrix
     */
    public Matrix multiply(Matrix rhs) {
        int i, j, x;
        Matrix C = new Matrix();

        for (i = 0; i < 4; i++) {
            for (j = 0; j < 4; j++) {
                C.matrix[i][j] = 0;
                for (x = 0; x < 4; x++) {
                    C.matrix[i][j] += this.matrix[i][x] * rhs.matrix[x][j];
                }
            }
        }
        return C;
    }

}
