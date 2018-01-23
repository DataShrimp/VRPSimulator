import org.junit.Test;

import java.util.ArrayList;

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
        exp.setAllActions(action);
        engine.run(-1);
        System.out.println(exp.getReward());
    }

    @Test
    public void run2() {
        SimuEngine engine = new SimuEngine();
        Experiment exp = new Experiment(5);
        engine.initialize(exp);
        System.out.println(engine.run(1));
        System.out.println(engine.run(3));
        System.out.println(engine.run(2));
        System.out.println(engine.run(4));
        System.out.println(engine.run(0));
        System.out.println(engine.run(2));
        System.out.println(exp.getReward());
    }
}