package fa.dto;

public class ValidationResponse {
    private boolean valid;
    private double[] x;
    private double[] y;

    public ValidationResponse(boolean valid) {
        this(valid, new double[0], new double[0]);
    }

    public ValidationResponse(boolean valid, double[] x, double[] y) {
        this.valid = valid;
        this.x = x;
        this.y = y;
    }

    public boolean isValid() {
        return valid;
    }

    public double[] getX() {
        return this.x;
    }

    public double[] getY() {
        return this.y;
    }
}
