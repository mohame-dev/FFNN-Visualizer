package fa.nn.util;

import java.util.Objects;

public class Preconditions {
    public static void requireProbability(double v, String name) {
        if (!Double.isFinite(v) || v < 0.0 || v >= 1.0) {
            throw new IllegalArgumentException(name + " must be in (0,1); got " + v);
        }
    }

    public static void requirePositive(int v, String name) {
        if (v <= 0) {
            throw new IllegalArgumentException(name + " must be > 0; got " + v);
        }
    }

    public static void requirePositiveOrZero(int v, String name) {
        if (v < 0) {
            throw new IllegalArgumentException(name + " must be ≥ 0; got " + v);
        }
    }

    public static void requireMatrix(double[][] m, int rows, int cols, String name) {
        Objects.requireNonNull(m, name);
        if (m.length != rows) {
            throw new IllegalArgumentException(name + " rows " + m.length + " != expected " + rows);
        }
        for (int r = 0; r < rows; r++) {
            double[] row = Objects.requireNonNull(m[r], name + "[" + r + "]");
            if (row.length != cols) {
                throw new IllegalArgumentException(name + "[" + r + "] cols " + row.length + " != expected " + cols
                        + " (expected shape " + shape(rows, cols) + ")");
            }
        }
    }

    public static void requireVector(double[] v, int len, String name) {
        Objects.requireNonNull(v, name);
        if (v.length != len) {
            throw new IllegalArgumentException(name + " length " + v.length + " != expected " + len);
        }
    }

    public static String shape(int rows, int cols) {
        return "(" + rows + "x" + cols + ")";
    }
}
