package fa.nn.examples;

import java.util.Arrays;
import java.util.Random;

import fa.nn.Layer;
import fa.nn.NeuralNetwork;
import fa.nn.activation.Linear;
import fa.nn.activation.ReLU;
import fa.nn.initialize.XavierInitializer;
import fa.nn.learn.Adam;
import fa.nn.loss.MSE;

public class XSquared {
    private static final int n = 10000;
    private static final int epochs = 1000;

    private static final double[] x = new double[n];
    private static final double[] y = new double[n];

    private static Random rand = null;

    public static void main(String[] arg) {
        rand = new Random();

        NeuralNetwork nn = initialize();
        generatePoints();

        nn.fit(x, y, 0.2, epochs, 256, true, rand);

        double[] xPred = { -30, -20, -4, 4, 12, 30 };
        double[] yPred = predict(nn, xPred);

        System.out.println("x: " + Arrays.toString(xPred));
        System.out.println("y: " + Arrays.toString(yPred));
    }

    public static void generatePoints() {
        for (int i = 0; i < n; i++) {
            x[i] = rand.nextDouble() * 100 - 50;
            y[i] = x[i] * x[i];
        }
    }

    public static double[] predict(NeuralNetwork nn, double[] values) {
        double[] predicted = new double[values.length];

        for (int i = 0; i < values.length; i++) {
            double x = values[i];
            double y = nn.predict(new double[] { x })[0];
            predicted[i] = y;
        }

        return predicted;
    }

    public static NeuralNetwork initialize() {
        Layer[] layers = new Layer[] {
                new Layer(1, 32, new ReLU(), new XavierInitializer(rand)),
                new Layer(32, 32, new ReLU(), new XavierInitializer(rand)),
                new Layer(32, 1, new Linear(), new XavierInitializer(rand))
        };

        NeuralNetwork nn = new NeuralNetwork(layers);
        nn.setup(new Adam(nn), new MSE());

        return nn;
    }
}
