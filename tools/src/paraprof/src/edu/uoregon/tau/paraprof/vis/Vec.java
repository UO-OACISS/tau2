package edu.uoregon.tau.paraprof.vis;

public class Vec {
    float x;
    float y;
    float z;
    float w;

    public Vec(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.w = 1;
    }

    public Vec(double x, double y, double z) {
        this.x = (float) x;
        this.y = (float) y;
        this.z = (float) z;
        this.w = 1;
    }

    // copy constructor
    public Vec(Vec rhs) {
        this.x = rhs.x;
        this.y = rhs.y;
        this.z = rhs.z;
        this.w = rhs.w;
    }

    public float x() {
        return x;
    }

    public float y() {
        return y;
    }

    public float z() {
        return z;
    }

    public float w() {
        return w;
    }

    public void setx(double x) {
        this.x = (float) x;
    }

    public void sety(double y) {
        this.y = (float) y;
    }

    public void setz(double z) {
        this.z = (float) z;
    }

    public void setw(double w) {
        this.w = (float) w;
    }

    public void setx(float x) {
        this.x = x;
    }

    public void sety(float y) {
        this.y = y;
    }

    public void setz(float z) {
        this.z = z;
    }

    public void setw(float w) {
        this.w = w;
    }

    static Vec add(Vec a, Vec b) {
        return new Vec(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
    }

    public double length() {
        double l2 = (x * x) + (y * y) + (z * z);
        double l = Math.sqrt(l2);
        return l;
    }

    public void normalize() {
        double l2 = (x * x) + (y * y) + (z * z);
        double l = Math.sqrt(l2);
        x = (float) (x / l);
        y = (float) (y / l);
        z = (float) (z / l);
    }

    public Vec add(Vec rhs) {
        return new Vec(this.x + rhs.x, this.y + rhs.y, this.z + rhs.z);
    }

    public Vec subtract(Vec rhs) {
        return new Vec(this.x - rhs.x, this.y - rhs.y, this.z - rhs.z);
    }

    public Vec cross(Vec rhs) {
        return new Vec((this.y * rhs.z) - (this.z * rhs.y), (this.z * rhs.x) - (this.x * rhs.z),
                (this.x * rhs.y) - (this.y * rhs.x));
    }

    public String toString() {
        return "(" + x + ", " + y + ", " + z + ")";
    }

}
