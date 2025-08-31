package fa;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import fa.nn.Layer;
import fa.nn.NeuralNetwork;
import fa.nn.activation.ReLU;
import fa.nn.learn.Backpropagation;
import fa.nn.learn.SGD;
import fa.nn.loss.MSE;

public class BackpropagationTest {
    Backpropagation backprop;
    NeuralNetwork nn;

    @BeforeEach
    public void setup() {
        Layer l1 = new Layer(2, 2, new ReLU());

        double[][] w = { { 1.0, 0.5 }, { 2.0, 2.5 } };
        double[] b = { 0, 0 };
        l1.set(w, b);

        this.nn = new NeuralNetwork(new Layer[] { l1 });
        nn.setup(new SGD(nn, 1), new MSE());

        this.backprop = new Backpropagation(nn, new MSE());
    }

    @Test
    public void constructor_nullParameters_ExceptionThrown() {
        assertThrows(NullPointerException.class, () -> new Backpropagation(null, new MSE()));
        assertThrows(NullPointerException.class, () -> new Backpropagation(nn, null));
    }

    @Test
    public void compute_nullParameters_ExceptionThrown() {
        double[] input = new double[2];
        double[] expected = new double[2];
        double[][][] gw = new double[2][][];
        double[][] gb = new double[2][];

        assertThrows(NullPointerException.class, () -> this.backprop.compute(null, expected, gw, gb));
        assertThrows(NullPointerException.class, () -> this.backprop.compute(input, null, gw, gb));
        assertThrows(NullPointerException.class, () -> this.backprop.compute(input, expected, null, gb));
        assertThrows(NullPointerException.class, () -> this.backprop.compute(input, expected, gw, null));
    }

    @Test
    public void computeDeltas_nullParameters_ExceptionThrown() {
        assertThrows(NullPointerException.class, () -> this.backprop.computeDeltas(null));
    }

    @Test
    public void computeDeltas_invalidParameters_ExceptionThrown() {
        double[] expected = new double[3];
        assertThrows(IllegalArgumentException.class, () -> this.backprop.computeDeltas(expected));
    }

    @Test
    public void computeDeltas_validParameters_ShapeExpected() {
        double[] expected = new double[2];
        double[][] deltas = this.backprop.computeDeltas(expected);

        assertEquals(1, deltas.length);
        assertEquals(2, deltas[0].length);
    }
}
