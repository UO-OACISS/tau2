import edu.uoregon.tau.vis.TopoPlot;

/** Headless check of hover-pick projection math and nearest/front selection (no GL context needed). */
public class TopoPlotPickTest {
    private static void check(boolean ok, String msg) {
        if (!ok) {
            throw new AssertionError(msg);
        }
    }

    // Orthographic-style column-major matrices: x,y map 1:1 to a 200x100 viewport; z is depth.
    private static double[] identity() {
        double[] m = new double[16];
        m[0] = m[5] = m[10] = m[15] = 1;
        return m;
    }

    public static void main(String[] args) {
        TopoPlot plot = new TopoPlot();
        float[] min = {0, 0, 0, 0};
        float[] max = {10, 10, 10, 1};
        plot.setMinMax(min, max);
        plot.setSize(10, 10, 10);
        float[][] values = {
            {0, 0, 0, 1}, {10, 10, 0, 1}, {5, 5, 0, 1}, {5, 5, 10, 1}, {10, 0, 5, Float.NaN}
        };
        plot.setValues(values);

        double[] model = identity();
        double[] proj = identity();
        // NDC [-1,1] -> viewport 200x100; point x=1 maps to window x=200, x=0 to 100.
        int[] vp = {0, 0, 200, 100};
        proj[0] = 0.2; // scale so coordinates 0..10 span NDC 0..2 horizontally
        proj[5] = 0.2;
        proj[10] = 0.2; // depth
        proj[11] = 0;
        proj[15] = 1;
        model[14] = 0;
        // translate so point (0,0) lands at NDC (-1,-1)
        model[12] = -5;
        model[13] = -5;

        // Window position of point 2 (5,5): NDC (0,0) -> (100,50)
        check(plot.pick(model, proj, vp, 100, 50, 8) == 2 || plot.pick(model, proj, vp, 100, 50, 8) == 3,
              "nearest at center is one of the overlapping points");
        check(plot.pick(model, proj, vp, 100, 50, 8) == 2, "front (lower depth) point wins on overlap");
        check(plot.pick(model, proj, vp, 150, 0, 8) == -1, "empty area yields no pick");
        check(plot.pick(model, proj, vp, 100 + 9, 50, 8) == -1, "outside radius yields no pick");
        check(plot.pick(model, proj, vp, 100 + 3, 50 - 2, 8) == 2, "inside radius picks");

        // NaN-valued (hidden) points are never picked.
        check(plot.pick(model, proj, vp, 200, 0, 8) == -1, "hidden point is not pickable");

        // Behind the camera is rejected.
        double[] behind = identity();
        behind[15] = -1;
        check(plot.pick(model, behind, vp, 100, 50, 8) == -1, "w<=0 is rejected");
        System.out.println("TopoPlot pick tests passed.");
    }
}
