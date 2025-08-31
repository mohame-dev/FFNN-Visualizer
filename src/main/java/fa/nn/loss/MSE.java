package fa.nn.loss;

public class MSE implements LossFunction {
    @Override
    public double loss(double[] predicted, double[] expected) {
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            double diff = predicted[i] - expected[i];
            sum += 0.5 * diff * diff;
        }
        return sum / predicted.length;
    }

    @Override
    public double derivative(double predicted, double expected) {
        return predicted - expected;
    }
}
