package fa.core;

import java.util.Random;

import net.objecthunter.exp4j.Expression;

public class FunctionSampler {
    private Expression e;
    private double[] x;
    private double[] y;

    public FunctionSampler(Expression e, double xmin, double xmax, int npoints, Random rand) {
        this.e = e;
        this.x = new double[npoints];
        this.y = new double[npoints];

        this.sample(xmin, xmax, npoints, rand);
    }

    public double[] x() {
        return this.x;
    }

    public double[] y() {
        return this.y;
    }

    public void sample(double xmin, double xmax, int npoints, Random rand) {
        double range = xmax - xmin;

        for (int i = 0; i < npoints; i++) {
            x[i] = rand.nextDouble() * range + xmin;
            e.setVariable("x", x[i]);
            y[i] = e.evaluate();
        }
    }
}
