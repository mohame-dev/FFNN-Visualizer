package fa.core;

import java.util.Arrays;
import java.util.Random;

import fa.nn.Layer;
import fa.nn.NeuralNetwork;
import fa.nn.activation.Linear;
import fa.nn.activation.ReLU;
import fa.nn.learn.Adam;
import fa.nn.learn.Dataset;
import fa.nn.loss.MSE;

public class Trainer {
    private final int BATCH_SIZE = 256;
    private final double SPLIT = 0.2;

    private NeuralNetwork nn;
    private Dataset d;

    public Trainer(double[] x, double[] y, Random rand) {
        this.nn = this.initialize();

        double[][] mx = Arrays.stream(x)
                .mapToObj(v -> new double[] { v })
                .toArray(double[][]::new);

        double[][] my = Arrays.stream(y)
                .mapToObj(v -> new double[] { v })
                .toArray(double[][]::new);

        this.d = new Dataset(mx, my, SPLIT, rand);
    }

    public void next() {
        nn.fitNext(d, BATCH_SIZE);
    }

    public double[] predict(double[] x) {
        double[] predicted = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            double y = nn.predict(new double[] { x[i] })[0];
            predicted[i] = y;
        }

        return predicted;
    }

    public double trainLoss() {
        return this.nn.calculateLoss(this.d.trainX(), this.d.trainY());
    }

    public double valLoss() {
        return this.nn.calculateLoss(this.d.valX(), this.d.valY());
    }

    private NeuralNetwork initialize() {
        Layer[] layers = new Layer[] {
                new Layer(1, 32, new ReLU()),
                new Layer(32, 32, new ReLU()),
                new Layer(32, 1, new Linear())
        };

        NeuralNetwork nn = new NeuralNetwork(layers);
        nn.setup(new Adam(nn), new MSE());

        return nn;
    }
}
