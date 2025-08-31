package fa.nn.loss;

public class Quadratic implements LossFunction {
    @Override
    public double loss(double[] predicted, double[] expected) {
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = predicted[i] - expected[i];
            sum += 0.5 * diff * diff;
        }
        return sum;
    }

    @Override
    public double derivative(double predicted, double expected) {
        return predicted - expected;
    }
}
