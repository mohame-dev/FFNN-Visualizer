package fa.nn.activation;

public class Sigmoid implements ActivationFunction {
    @Override
    public double activate(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    public double derivative(double z) {
        double a = activate(z);
        return a * (1.0 - a);
    }
}
