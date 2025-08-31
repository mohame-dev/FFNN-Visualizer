package fa.dto;

public class PredictionResponse {
    private double[] x;
    private double[] y;
    private int epoch;
    private double loss;
    private double valLoss;

    public PredictionResponse(double[] x, double[] y, int epoch, double loss, double valLoss) {
        this.x = x;
        this.y = y;
        this.epoch = epoch;
        this.loss = loss;
        this.valLoss = valLoss;
    }

    public double[] getX() {
        return this.x;
    }

    public double[] getY() {
        return this.y;
    }

    public int getEpoch() {
        return this.epoch;
    }

    public double getLoss() {
        return this.loss;
    }

    public double getValLoss() {
        return this.valLoss;
    }
}
