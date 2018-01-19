import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.*;

public class SimuEngineTest {

    @Test
    public void run() {
        SimuEngine engine = new SimuEngine();
        Experiment exp = new Experiment(5);
        engine.initialize(exp);
        ArrayList<Integer> action = new ArrayList<Integer>() {{
            add(0);
            add(1);
            add(3);
            add(2);
            add(4);
            add(0);
        }};
        exp.setAction(action);
        engine.run();
        System.out.println(exp.getReward());
    }
}