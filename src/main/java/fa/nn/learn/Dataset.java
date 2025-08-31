package fa.nn.learn;

import java.util.Arrays;
import java.util.Random;

public class Dataset {
    private double[][] trainX;
    private double[][] trainY;
    private double[][] valX;
    private double[][] valY;
    private Random rand;

    public Dataset(double[][] x, double[][] y, double split, Random rand) {
        if (x.length != y.length) {
            throw new IllegalArgumentException();
        }

        int idx = (int) (x.length * (1 - split));

        this.trainX = Arrays.copyOfRange(x, 0, idx);
        this.trainY = Arrays.copyOfRange(y, 0, idx);
        this.valX = Arrays.copyOfRange(x, idx, x.length);
        this.valY = Arrays.copyOfRange(y, idx, y.length);

        this.rand = rand;
    }

    public double[][] trainX() {
        return trainX;
    }

    public double[][] trainY() {
        return trainY;
    }

    public double[][] valX() {
        return valX;
    }

    public double[][] valY() {
        return valY;
    }

    public void shuffle() {
        for (int i = trainX.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);
            double[] tx = trainX[i];
            trainX[i] = trainX[j];
            trainX[j] = tx;

            double[] ty = trainY[i];
            trainY[i] = trainY[j];
            trainY[j] = ty;
        }
    }

    public String toString() {
        String s = "trainX: " + Arrays.deepToString(trainX) + "\n";
        s += "trainX: " + Arrays.deepToString(trainY) + "\n";
        s += "valX: " + Arrays.deepToString(valX) + "\n";
        s += "valY: " + Arrays.deepToString(valY) + "\n";
        return s;
    }
}
