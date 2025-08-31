package fa.dto;

public class ValidationRequest {
    private String expression;
    private double xmin;
    private double xmax;
    private int npoints;
    private int epochs;
    private int interval;

    public void setExpression(String expression) {
        this.expression = expression;
    }

    public void setXmin(double xmin) {
        this.xmin = xmin;
    }

    public void setXmax(double xmax) {
        this.xmax = xmax;
    }

    public void setNpts(int npoints) {
        this.npoints = npoints;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void setInterval(int interval) {
        this.interval = interval;
    }

    public String getExpression() {
        return this.expression;
    }

    public double getXmin() {
        return this.xmin;
    }

    public double getXmax() {
        return this.xmax;
    }

    public int getNpoints() {
        return this.npoints;
    }

    public int getEpochs() {
        return this.epochs;
    }

    public int getInterval() {
        return this.interval;
    }
}
