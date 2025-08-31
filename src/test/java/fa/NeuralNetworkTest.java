package fa;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import fa.nn.Layer;
import fa.nn.NeuralNetwork;
import fa.nn.activation.ReLU;
import fa.nn.learn.Adam;
import fa.nn.learn.SGD;
import fa.nn.loss.MSE;

public class NeuralNetworkTest {
    Layer l1;
    NeuralNetwork nn;

    @BeforeEach
    void setup() {
        Layer l1 = new Layer(1, 2, new ReLU());
        double[][] w1 = { { 1.0 }, { 2.0 } };
        double[] b1 = { 0, 0 };
        l1.set(w1, b1);

        Layer l2 = new Layer(2, 1, new ReLU());
        double[][] w2 = { { 0.5, 1.0 } };
        double[] b2 = { 0 };
        l2.set(w2, b2);

        this.nn = new NeuralNetwork(new Layer[] { l1, l2 });
    }

    @Test
    public void constructor_nullParameters_ExceptionThrown() {
        assertThrows(NullPointerException.class, () -> new NeuralNetwork(null));
    }

    public void constructor_noLayers_ExceptionThrown() {
        Layer[] layers = {};
        assertThrows(IllegalArgumentException.class, () -> new NeuralNetwork(layers));
    }

    @Test
    public void constructor_layerMismatch_ExceptionThrown() {
        Layer l1 = new Layer(1, 1, new ReLU());
        Layer l2 = new Layer(5, 1, new ReLU());

        Layer[] layers = { l1, l2 };
        assertThrows(IllegalArgumentException.class, () -> new NeuralNetwork(layers));
    }

    @Test
    public void predict_nullParameters_ExceptionThrown() {
        assertThrows(NullPointerException.class, () -> this.nn.predict(null));
    }

    @Test
    public void predict_invalidLength_ExceptionThrown() {
        assertThrows(IllegalArgumentException.class, () -> this.nn.predict(new double[2]));
    }

    @Test
    public void predict_validParameters_ShapeExpected() {
        double[] x = new double[1];
        double[] yhat = this.nn.predict(x);
        assertEquals(1, yhat.length);
    }

    @Test
    public void predict_validParameters_ComputesExpected() {
        double[] x = { 1.0 };
        double[] yhat = this.nn.predict(x);

        // z[1][1] = 1.0 * 1.0 = 1.0 => a(1.0) = 1.0
        // z[1][2] = 2.0 * 1.0 = 2.0 => a(2.0) = 2.0

        // z[2][1] = 0.5 * 1.0 + 1.0 * 2.0 = 2.5 => a(2.5) = 2.5
        double[] expected = { 2.5 };
        assertArrayEquals(expected, yhat);
    }

    @Test
    public void setup_nullParameters_ExceptionThrown() {
        assertThrows(NullPointerException.class, () -> this.nn.setup(new Adam(nn), null));
        assertThrows(NullPointerException.class, () -> this.nn.setup(null, new MSE()));
    }

    @Test
    public void fit_nullParameters_ExceptionThrown() {
        double[] x = new double[1];
        double[] y = new double[1];

        this.nn.setup(new Adam(nn), new MSE());

        assertThrows(NullPointerException.class, () -> this.nn.fit(x, null, 0.1, 1, 1, false, null));
        assertThrows(NullPointerException.class, () -> this.nn.fit(null, y, 0.1, 1, 1, false, null));

    }

    @Test
    public void fit_invalidParameters_ExceptionThrown() {
        double[] x = new double[1];
        double[] y = new double[1];

        this.nn.setup(new Adam(nn), new MSE());

        assertThrows(IllegalArgumentException.class,
                () -> this.nn.fit(x, y, -1, 1, 1, false, null));
        assertThrows(IllegalArgumentException.class,
                () -> this.nn.fit(x, y, 0.1, -1, 1, false, null));
        assertThrows(IllegalArgumentException.class,
                () -> this.nn.fit(x, y, 0.1, 1, -1, false, null));
        assertThrows(IllegalArgumentException.class,
                () -> this.nn.fit(new double[2], y, 0.1, 1, 1, false, null));
        assertThrows(IllegalArgumentException.class,
                () -> this.nn.fit(new double[4][1], new double[4][2], 0.1, 1, 1, false, null));
    }

    @Test
    public void fit_setupNotCalledBeforeFitting_ExceptionThrown() {
        double[] x = new double[1];
        double[] y = new double[1];

        assertThrows(IllegalStateException.class, () -> this.nn.fit(x, y, 0.1, 1, 1, false, null));
    }

    @Test
    public void fit_validParameters_ComputesExpected() {
        Layer l1 = new Layer(2, 2, new ReLU());

        double[][] w = { { 1.0, 0.5 }, { 2.0, 2.5 } };
        double[] b = { 0, 0 };
        l1.set(w, b);

        NeuralNetwork nn = new NeuralNetwork(new Layer[] { l1 });
        nn.setup(new SGD(nn, 1), new MSE());

        double[][] x = { { 0.5, 1.0 } };
        double[][] y = { { 1.0, 0.5 } };

        nn.fit(x, y, 0.0, 1, 1, false, null);
        double[] predicted = nn.predict(x[0]);

        assertArrayEquals(new double[] { 1.0, 0.0 }, predicted);
    }
}
